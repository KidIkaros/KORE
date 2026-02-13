//! RMSNorm: Root Mean Square Layer Normalization.
//!
//! Simpler and faster than LayerNorm — no mean subtraction, no bias.
//!
//! `y = x / sqrt(mean(x^2) + eps) * gamma`
//!
//! Used in LLaMA, Mistral, and most modern transformer architectures.

use std::collections::HashMap;
use std::sync::Arc;

use kore_core::{DType, KoreError, Tensor};
use kore_core::autograd::{self, GradFn, GradNode};
use crate::module::Module;

/// Root Mean Square Layer Normalization.
///
/// Normalizes the last dimension of the input tensor using RMS statistics.
/// Learnable scale parameter `gamma` (no bias).
#[derive(Clone)]
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

    /// Create an RMSNorm layer from a pre-trained gamma (scale) tensor.
    ///
    /// # Arguments
    /// * `gamma` - Scale parameter tensor of shape `[dim]`
    /// * `eps` - Small constant for numerical stability
    pub fn from_weight(gamma: Tensor, eps: f32) -> Self {
        let dim = gamma.numel();
        Self {
            gamma,
            dim,
            eps,
            training: false,
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

    /// Scale parameter (gamma).
    pub fn gamma(&self) -> &Tensor {
        &self.gamma
    }
}

impl std::fmt::Display for RMSNorm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RMSNorm(dim={}, eps={})", self.dim, self.eps)
    }
}

impl Module for RMSNorm {
    fn clone_box(&self) -> Box<dyn Module> { Box::new(self.clone()) }

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

        if self.dim == 0 {
            return Err(KoreError::ShapeMismatch {
                expected: vec![1],
                got: vec![0],
            });
        }

        // Use fused kernel for forward pass
        let result = kore_kernels::cpu_fused::fused_rms_norm(input, &self.gamma, self.eps)?;

        // Wire into autograd graph if tracking gradients
        let tracks = autograd::is_grad_enabled()
            && (input.tracks_grad() || self.gamma.tracks_grad());
        if tracks {
            // Order must match fused_rms_norm_backward return: (dx, dgamma)
            let mut inputs = Vec::new();
            if let Some(n) = input.grad_node() { inputs.push(Arc::clone(n)); }  // idx 0 → dx
            if let Some(n) = self.gamma.grad_node() { inputs.push(Arc::clone(n)); }  // idx 1 → dgamma
            let grad_fn = Box::new(FusedRMSNormBackward {
                input: input.clone(),
                gamma: self.gamma.clone(),
                eps: self.eps,
            });
            let node = GradNode::with_grad_fn(grad_fn, inputs);
            Ok(result.with_grad_node(node))
        } else {
            Ok(result)
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.gamma]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.gamma]
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        vec![("gamma".into(), &self.gamma)]
    }

    fn set_parameters(&mut self, params: &[Tensor]) -> usize {
        self.gamma = params[0].clone();
        1
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

// ============================================================================
// Fused backward GradFn
// ============================================================================

/// GradFn wrapper that delegates to `kore_kernels::cpu_fused_backward::fused_rms_norm_backward`.
struct FusedRMSNormBackward {
    input: Tensor,
    gamma: Tensor,
    eps: f32,
}

impl GradFn for FusedRMSNormBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        match kore_kernels::cpu_fused_backward::fused_rms_norm_backward(
            grad_output, &self.input, &self.gamma, self.eps,
        ) {
            Ok((dx, dgamma)) => vec![Some(dx), Some(dgamma)],
            Err(e) => {
                eprintln!("FusedRMSNormBackward failed: {e}");
                vec![None, None]
            }
        }
    }

    fn name(&self) -> &str { "FusedRMSNormBackward" }
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

    #[test]
    fn test_rms_norm_backward_produces_grads() {
        let norm = RMSNorm::new(4, 1e-6);
        let mut input = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
        input.set_requires_grad(true);
        let input_node = kore_core::autograd::GradNode::leaf();
        let input = input.with_grad_node(std::sync::Arc::clone(&input_node));

        let output = norm.forward(&input).unwrap();
        assert!(output.grad_node().is_some(), "output should have grad node");

        // Backward with ones
        let go = Tensor::ones(&[2, 4]);
        kore_core::autograd::backward(output.grad_node().unwrap(), go);

        let g = input_node.get_grad();
        assert!(g.is_some(), "input should have gradient after backward");
        let gd = g.unwrap();
        assert_eq!(gd.shape().dims(), &[2, 4]);
        assert!(gd.as_f32_slice().unwrap().iter().all(|v| v.is_finite()));
    }
}
