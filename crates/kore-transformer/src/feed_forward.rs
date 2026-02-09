//! SwiGLU Feed-Forward Network (used in LLaMA, Mistral, etc.)
//!
//! FFN(x) = (swish(x @ W1) * (x @ W3)) @ W2
//! where swish(x) = x * sigmoid(x)

use kore_core::{KoreError, Tensor};

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
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, KoreError> {
        let x_data = x.as_f32_slice().ok_or(KoreError::StorageError("expected f32 tensor".into()))?;
        let seq_len = x.shape().dims()[0];
        let d = self.d_model;
        let ff = self.d_ff;

        let w1 = self.w1.as_f32_slice().unwrap();
        let w2 = self.w2.as_f32_slice().unwrap();
        let w3 = self.w3.as_f32_slice().unwrap();

        // gate = x @ W1^T → [seq_len, d_ff]
        let gate = matmul_ab_t(x_data, w1, seq_len, d, ff);
        // up = x @ W3^T → [seq_len, d_ff]
        let up = matmul_ab_t(x_data, w3, seq_len, d, ff);

        // swiglu = swish(gate) * up
        let mut hidden = vec![0.0f32; seq_len * ff];
        for i in 0..hidden.len() {
            let g = gate[i];
            let swish = g * sigmoid(g);
            hidden[i] = swish * up[i];
        }

        // out = hidden @ W2^T → [seq_len, d_model]
        let out = matmul_ab_t(&hidden, w2, seq_len, ff, d);

        Ok(Tensor::from_f32(&out, &[seq_len, d]))
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

/// C = A @ B^T where A is [m, k], B is [n, k] → C is [m, n]
fn matmul_ab_t(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[j * k + p];
            }
            c[i * n + j] = acc;
        }
    }
    c
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
