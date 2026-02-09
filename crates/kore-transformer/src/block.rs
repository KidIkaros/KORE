//! Single transformer decoder block (pre-norm architecture).
//!
//! Block(x) = x + MHA(RMSNorm(x)) + FFN(RMSNorm(x + attn_out))

use kore_core::{KoreError, Tensor};
use crate::rms_norm::RMSNorm;
use crate::mha::MultiHeadAttention;
use crate::feed_forward::FeedForward;

/// A single decoder transformer block.
pub struct TransformerBlock {
    pub attn_norm: RMSNorm,
    pub attn: MultiHeadAttention,
    pub ffn_norm: RMSNorm,
    pub ffn: FeedForward,
    pub d_model: usize,
}

impl TransformerBlock {
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize, norm_eps: f32) -> Self {
        Self {
            attn_norm: RMSNorm::new(d_model, norm_eps),
            attn: MultiHeadAttention::new(d_model, n_heads),
            ffn_norm: RMSNorm::new(d_model, norm_eps),
            ffn: FeedForward::new(d_model, d_ff),
            d_model,
        }
    }

    /// Forward pass.
    ///
    /// `x`: [seq_len, d_model]
    /// `mask`: optional attention mask
    /// `use_cache`: whether to use KV cache
    ///
    /// Returns: [seq_len, d_model]
    pub fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        use_cache: bool,
    ) -> Result<Tensor, KoreError> {
        let x_data = x.as_f32_slice().ok_or(KoreError::StorageError("expected f32 tensor".into()))?;
        let n = x_data.len();

        // Pre-norm attention: h = x + MHA(RMSNorm(x))
        let normed = self.attn_norm.forward(x)?;
        let attn_out = self.attn.forward(&normed, mask, use_cache)?;
        let attn_data = attn_out.as_f32_slice().unwrap();

        let mut h = vec![0.0f32; n];
        for i in 0..n {
            h[i] = x_data[i] + attn_data[i];
        }
        let h_tensor = Tensor::from_f32(&h, x.shape().dims());

        // Pre-norm FFN: out = h + FFN(RMSNorm(h))
        let normed2 = self.ffn_norm.forward(&h_tensor)?;
        let ffn_out = self.ffn.forward(&normed2)?;
        let ffn_data = ffn_out.as_f32_slice().unwrap();

        let mut out = vec![0.0f32; n];
        for i in 0..n {
            out[i] = h[i] + ffn_data[i];
        }

        Ok(Tensor::from_f32(&out, x.shape().dims()))
    }

    /// Reset KV cache.
    pub fn reset_cache(&mut self) {
        self.attn.reset_cache();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_shape() {
        let mut block = TransformerBlock::new(64, 4, 128, 1e-5);
        let x = Tensor::from_f32(&vec![0.1; 8 * 64], &[8, 64]);
        let out = block.forward(&x, None, false).unwrap();
        assert_eq!(out.shape().dims(), &[8, 64]);
    }

    #[test]
    fn test_block_residual() {
        let mut block = TransformerBlock::new(32, 4, 64, 1e-5);
        let x = Tensor::from_f32(&vec![1.0; 2 * 32], &[2, 32]);
        let out = block.forward(&x, None, false).unwrap();
        let data = out.as_f32_slice().unwrap();
        // Residual connection means output shouldn't be zero
        assert!(data.iter().any(|&v| v.abs() > 0.1));
    }

    #[test]
    fn test_block_with_cache() {
        let mut block = TransformerBlock::new(32, 4, 64, 1e-5);
        let x1 = Tensor::from_f32(&vec![0.1; 1 * 32], &[1, 32]);
        let _ = block.forward(&x1, None, true).unwrap();
        let x2 = Tensor::from_f32(&vec![0.2; 1 * 32], &[1, 32]);
        let out = block.forward(&x2, None, true).unwrap();
        assert_eq!(out.shape().dims(), &[1, 32]);
    }
}
