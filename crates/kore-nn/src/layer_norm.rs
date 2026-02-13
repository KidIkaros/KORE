//! Layer Normalization.
//!
//! Normalizes across the last dimension: y = (x - mean) / sqrt(var + eps) * gamma + beta

use kore_core::{DType, KoreError, Tensor};
use crate::module::Module;

/// Layer Normalization over the last dimension.
pub struct LayerNorm {
    normalized_shape: usize,
    eps: f32,
    gamma: Tensor,
    beta: Tensor,
    training: bool,
}

impl LayerNorm {
    /// Create a new LayerNorm layer.
    pub fn new(normalized_shape: usize, eps: f32) -> Self {
        let mut gamma = Tensor::ones(&[normalized_shape]);
        gamma.set_requires_grad(true);
        let mut beta = Tensor::zeros(&[normalized_shape], DType::F32);
        beta.set_requires_grad(true);

        Self {
            normalized_shape,
            eps,
            gamma,
            beta,
            training: true,
        }
    }

    /// Create from existing gamma and beta tensors (preserves trained params).
    pub fn from_weight(gamma: Tensor, beta: Tensor, eps: f32) -> Self {
        let normalized_shape = gamma.numel();
        Self { gamma, beta, normalized_shape, eps, training: true }
    }

    /// Create with default eps (1e-5).
    pub fn default_new(normalized_shape: usize) -> Self {
        Self::new(normalized_shape, 1e-5)
    }

    /// Get gamma (weight).
    pub fn gamma(&self) -> &Tensor {
        &self.gamma
    }

    /// Get beta (bias).
    pub fn beta(&self) -> &Tensor {
        &self.beta
    }

    /// Normalized shape dimension.
    pub fn normalized_shape(&self) -> usize {
        self.normalized_shape
    }

    /// Epsilon value.
    pub fn eps(&self) -> f32 {
        self.eps
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> kore_core::Result<Tensor> {
        let data = input.contiguous();
        let dims = data.shape().dims().to_vec();
        let slice = data.as_f32_slice().ok_or_else(|| {
            KoreError::UnsupportedDType(input.dtype())
        })?;

        let last_dim = *dims.last().ok_or_else(|| {
            KoreError::StorageError("LayerNorm: empty shape".into())
        })?;
        if last_dim != self.normalized_shape {
            return Err(KoreError::ShapeMismatch {
                expected: vec![self.normalized_shape],
                got: vec![last_dim],
            });
        }

        let batch_size = data.numel() / last_dim;
        let gamma = self.gamma.as_f32_slice().unwrap();
        let beta = self.beta.as_f32_slice().unwrap();

        let mut result = vec![0.0f32; data.numel()];

        for b in 0..batch_size {
            let start = b * last_dim;
            let end = start + last_dim;
            let row = &slice[start..end];

            // Mean
            let mean: f32 = row.iter().sum::<f32>() / last_dim as f32;

            // Variance
            let var: f32 = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / last_dim as f32;

            let inv_std = 1.0 / (var + self.eps).sqrt();

            for i in 0..last_dim {
                result[start + i] = (row[i] - mean) * inv_std * gamma[i] + beta[i];
            }
        }

        Ok(Tensor::from_f32(&result, &dims))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.gamma, &self.beta]
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        vec![("gamma".into(), &self.gamma), ("beta".into(), &self.beta)]
    }

    fn set_parameters(&mut self, params: &[Tensor]) -> usize {
        self.gamma = params[0].clone();
        self.beta = params[1].clone();
        2
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_shape() {
        let ln = LayerNorm::default_new(4);
        let input = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
        let output = ln.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 4]);
    }

    #[test]
    fn test_layer_norm_zero_mean() {
        let ln = LayerNorm::new(4, 1e-5);
        let input = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let output = ln.forward(&input).unwrap();
        let data = output.as_f32_slice().unwrap();

        // With default gamma=1, beta=0, output should have ~zero mean
        let mean: f32 = data.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean should be ~0, got {}", mean);
    }

    #[test]
    fn test_layer_norm_unit_variance() {
        let ln = LayerNorm::new(4, 1e-5);
        let input = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let output = ln.forward(&input).unwrap();
        let data = output.as_f32_slice().unwrap();

        let mean: f32 = data.iter().sum::<f32>() / 4.0;
        let var: f32 = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 0.1, "variance should be ~1, got {}", var);
    }

    #[test]
    fn test_layer_norm_parameters() {
        let ln = LayerNorm::default_new(8);
        assert_eq!(ln.parameters().len(), 2);
        assert_eq!(ln.gamma().shape().dims(), &[8]);
        assert_eq!(ln.beta().shape().dims(), &[8]);
    }

    #[test]
    fn test_layer_norm_batched() {
        let ln = LayerNorm::default_new(3);
        let input = Tensor::from_f32(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0], &[2, 3]);
        let output = ln.forward(&input).unwrap();
        let data = output.as_f32_slice().unwrap();

        // Each row should be independently normalized
        let mean1: f32 = data[0..3].iter().sum::<f32>() / 3.0;
        let mean2: f32 = data[3..6].iter().sum::<f32>() / 3.0;
        assert!(mean1.abs() < 1e-5);
        assert!(mean2.abs() < 1e-5);
    }
}
