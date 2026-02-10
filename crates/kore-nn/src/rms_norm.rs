//! RMSNorm: Root Mean Square Layer Normalization.
//!
//! Simpler and faster than LayerNorm — no mean subtraction, no bias.
//!
//! `y = x / sqrt(mean(x^2) + eps) * gamma`
//!
//! Used in LLaMA, Mistral, and most modern transformer architectures.

use std::collections::HashMap;

use kore_core::{DType, KoreError, Tensor};
use crate::module::Module;

/// Root Mean Square Layer Normalization.
///
/// Normalizes the last dimension of the input tensor using RMS statistics.
/// Learnable scale parameter `gamma` (no bias).
pub struct RMSNorm {
    gamma: Tensor,
    dim: usize,
    eps: f32,
    training: bool,
}

impl RMSNorm {
    /// Create a new RMSNorm layer.
    ///
    /// # Arguments
    /// * `dim` - Size of the last dimension to normalize over
    /// * `eps` - Small constant for numerical stability (default: 1e-6)
    pub fn new(dim: usize, eps: f32) -> Self {
        let mut gamma = Tensor::ones(&[dim]);
        gamma.set_requires_grad(true);
        Self {
            gamma,
            dim,
            eps,
            training: true,
        }
    }

    /// Dimension being normalized.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Epsilon value.
    pub fn eps(&self) -> f32 {
        self.eps
    }
}

impl std::fmt::Display for RMSNorm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RMSNorm(dim={}, eps={})", self.dim, self.eps)
    }
}

impl Module for RMSNorm {
    fn forward(&self, input: &Tensor) -> kore_core::Result<Tensor> {
        if input.dtype() != DType::F32 {
            return Err(KoreError::UnsupportedDType(input.dtype()));
        }

        let dims = input.shape().dims();
        let ndim = dims.len();
        if ndim == 0 || dims[ndim - 1] != self.dim {
            return Err(KoreError::ShapeMismatch {
                expected: vec![self.dim],
                got: dims.to_vec(),
            });
        }

        let data = input.contiguous();
        let x = data.as_f32_slice()
            .ok_or_else(|| KoreError::UnsupportedDType(input.dtype()))?;
        let gamma = self.gamma.as_f32_slice()
            .ok_or_else(|| KoreError::UnsupportedDType(self.gamma.dtype()))?;

        let batch: usize = dims[..ndim - 1].iter().product();
        let d = self.dim;
        if d == 0 {
            return Err(KoreError::ShapeMismatch {
                expected: vec![1],  // dim must be >= 1
                got: vec![0],
            });
        }
        let mut output = vec![0.0f32; x.len()];

        for b in 0..batch {
            let row = &x[b * d..(b + 1) * d];
            let out_row = &mut output[b * d..(b + 1) * d];

            // Compute RMS: sqrt(mean(x^2) + eps)
            let sq_mean: f32 = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
            let rms = (sq_mean + self.eps).sqrt();
            let inv_rms = 1.0 / rms;

            for i in 0..d {
                out_row[i] = row[i] * inv_rms * gamma[i];
            }
        }

        Ok(Tensor::from_f32(&output, dims))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.gamma]
    }

    fn named_parameters(&self) -> Vec<(&str, &Tensor)> {
        vec![("gamma", &self.gamma)]
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();
        sd.insert("gamma".to_string(), self.gamma.clone());
        sd
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_shape() {
        let norm = RMSNorm::new(8, 1e-6);
        let input = Tensor::ones(&[2, 8]);
        let output = norm.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 8]);
    }

    #[test]
    fn test_rms_norm_unit_input() {
        let norm = RMSNorm::new(4, 1e-6);
        let input = Tensor::ones(&[1, 4]);
        let output = norm.forward(&input).unwrap();
        let data = output.as_f32_slice().unwrap();
        // RMS of all-ones = 1.0, so output ≈ 1.0 * gamma(=1.0) = 1.0
        for &v in data {
            assert!((v - 1.0).abs() < 1e-3, "got {}", v);
        }
    }

    #[test]
    fn test_rms_norm_finite() {
        let norm = RMSNorm::new(16, 1e-6);
        let input = Tensor::randn(&[4, 16]);
        let output = norm.forward(&input).unwrap();
        let data = output.as_f32_slice().unwrap();
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_rms_norm_parameters() {
        let norm = RMSNorm::new(32, 1e-5);
        let params = norm.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].numel(), 32);
    }

    #[test]
    fn test_rms_norm_3d_input() {
        let norm = RMSNorm::new(8, 1e-6);
        let input = Tensor::ones(&[2, 3, 8]);
        let output = norm.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 3, 8]);
    }

    #[test]
    fn test_rms_norm_display() {
        let norm = RMSNorm::new(64, 1e-5);
        let s = format!("{}", norm);
        assert!(s.contains("RMSNorm"));
        assert!(s.contains("64"));
    }
}
