//! SwiGLU Feed-Forward Network (used in LLaMA, Mistral, etc.)
//!
//! FFN(x) = (swish(x @ W1) * (x @ W3)) @ W2
//! where swish(x) = x * sigmoid(x)
//!
//! Uses `kore_kernels::cpu_matmul::matmul_f32` for SIMD-accelerated projections.

use kore_core::{KoreError, Tensor};
use kore_kernels::cpu_matmul::matmul_f32;

/// SwiGLU feed-forward block.
pub struct FeedForward {
    /// Gate projection: [d_model, d_ff]
    pub w1: Tensor,
    /// Down projection: [d_ff, d_model]
    pub w2: Tensor,
    /// Up projection: [d_model, d_ff]
    pub w3: Tensor,
    pub d_model: usize,
    pub d_ff: usize,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let scale = (1.0 / d_model as f64).sqrt() as f32;
        Self {
            w1: random_tensor(d_model, d_ff, scale, 0),
            w2: random_tensor(d_ff, d_model, scale, 1),
            w3: random_tensor(d_model, d_ff, scale, 2),
            d_model,
            d_ff,
        }
    }

    /// Forward pass: x is [seq_len, d_model] → [seq_len, d_model]
    ///
    /// W1 is [d_model, d_ff], W2 is [d_ff, d_model], W3 is [d_model, d_ff]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, KoreError> {
        let seq_len = x.shape().dims()[0];
        let ff = self.d_ff;

        // gate = x @ W1 → [seq_len, d_ff]
        let gate_tensor = matmul_f32(x, &self.w1)?;

        // up = x @ W3 → [seq_len, d_ff]
        let up_tensor = matmul_f32(x, &self.w3)?;

        // swiglu = swish(gate) * up
        let gate = gate_tensor.as_f32_slice().unwrap();
        let up = up_tensor.as_f32_slice().unwrap();
        let mut hidden = vec![0.0f32; seq_len * ff];
        for i in 0..hidden.len() {
            let g = gate[i];
            let swish = g * sigmoid(g);
            hidden[i] = swish * up[i];
        }

        // out = hidden @ W2 → [seq_len, d_model]
        let hidden_tensor = Tensor::from_f32(&hidden, &[seq_len, ff]);
        matmul_f32(&hidden_tensor, &self.w2)
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn random_tensor(rows: usize, cols: usize, scale: f32, seed: usize) -> Tensor {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| {
            let x = (((i + seed * 999983) * 2654435761 + 1013904223) & 0xFFFFFF) as f32
                / 0xFFFFFF as f32;
            (x * 2.0 - 1.0) * scale
        })
        .collect();
    Tensor::from_f32(&data, &[rows, cols])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffn_shape() {
        let ffn = FeedForward::new(64, 128);
        let x = Tensor::from_f32(&vec![0.1; 8 * 64], &[8, 64]);
        let out = ffn.forward(&x).unwrap();
        assert_eq!(out.shape().dims(), &[8, 64]);
    }

    #[test]
    fn test_ffn_single_token() {
        let ffn = FeedForward::new(32, 64);
        let x = Tensor::from_f32(&vec![0.5; 32], &[1, 32]);
        let out = ffn.forward(&x).unwrap();
        assert_eq!(out.shape().dims(), &[1, 32]);
    }

    #[test]
    fn test_ffn_nonzero_output() {
        let ffn = FeedForward::new(16, 32);
        let x = Tensor::from_f32(&vec![1.0; 2 * 16], &[2, 16]);
        let out = ffn.forward(&x).unwrap();
        let data = out.as_f32_slice().unwrap();
        // At least some outputs should be nonzero with random weights
        assert!(data.iter().any(|&v| v.abs() > 1e-6));
    }
}
