//! Multi-Head Self-Attention with optional KV-cache.
//!
//! Uses `kore_kernels::cpu_matmul::matmul_f32` for SIMD-accelerated projections.

use kore_core::{KoreError, Tensor};
use kore_attention::kv_cache::KvCache;
use kore_kernels::cpu_matmul::matmul_f32;
use crate::rope::RopeTable;

/// Multi-Head Attention layer with Grouped-Query Attention (GQA) support.
///
/// Projects input into Q, K, V heads, runs scaled dot-product attention,
/// then projects back to d_model. Optionally applies RoPE to Q and K.
///
/// When `n_kv_heads < n_heads`, uses GQA: each KV head is shared across
/// `n_heads / n_kv_heads` query heads. When `n_kv_heads == 1`, this is MQA.
pub struct MultiHeadAttention {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub d_model: usize,
    pub d_head: usize,
    /// Q projection: [d_model, n_heads * d_head]
    pub wq: Tensor,
    /// K projection: [d_model, n_kv_heads * d_head]
    pub wk: Tensor,
    /// V projection: [d_model, n_kv_heads * d_head]
    pub wv: Tensor,
    /// Output projection: [n_heads * d_head, d_model]
    pub wo: Tensor,
    /// Per-KV-head caches (n_kv_heads caches, not n_heads)
    pub kv_caches: Vec<KvCache>,
    /// RoPE frequency table (None = no RoPE, use sinusoidal from embedding)
    pub rope: Option<RopeTable>,
    /// Current sequence offset for KV-cache (tracks how many tokens processed)
    pub seq_offset: usize,
}

impl MultiHeadAttention {
    /// Standard MHA (n_kv_heads == n_heads, no RoPE).
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        Self::new_full(d_model, n_heads, n_heads, false, 4096, 10000.0)
    }

    /// MHA with RoPE (n_kv_heads == n_heads).
    pub fn new_with_rope(d_model: usize, n_heads: usize, use_rope: bool, max_seq_len: usize, rope_base: f32) -> Self {
        Self::new_full(d_model, n_heads, n_heads, use_rope, max_seq_len, rope_base)
    }

    /// Full constructor with GQA + RoPE support.
    pub fn new_full(d_model: usize, n_heads: usize, n_kv_heads: usize, use_rope: bool, max_seq_len: usize, rope_base: f32) -> Self {
        assert!(d_model % n_heads == 0, "d_model must be divisible by n_heads");
        assert!(n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads");
        let d_head = d_model / n_heads;
        let scale = (1.0 / d_model as f64).sqrt() as f32;

        let q_dim = n_heads * d_head;      // == d_model
        let kv_dim = n_kv_heads * d_head;

        let wq = random_tensor(d_model, q_dim, scale);
        let wk = random_tensor(d_model, kv_dim, scale);
        let wv = random_tensor(d_model, kv_dim, scale);
        let wo = random_tensor(q_dim, d_model, scale);

        let kv_caches = (0..n_kv_heads).map(|_| KvCache::new(d_head, d_head, max_seq_len)).collect();

        let rope = if use_rope {
            Some(RopeTable::new(d_head, max_seq_len, rope_base))
        } else {
            None
        };

        Self { n_heads, n_kv_heads, d_model, d_head, wq, wk, wv, wo, kv_caches, rope, seq_offset: 0 }
    }

    /// Forward pass.
    ///
    /// `x`: [seq_len, d_model]
    /// `mask`: optional [seq_len, seq_len] additive mask
    /// `use_cache`: if true, appends to KV cache and uses full cached K/V
    ///
    /// Returns: [seq_len, d_model]
    pub fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        use_cache: bool,
    ) -> Result<Tensor, KoreError> {
        let dims = x.shape().dims();
        let seq_len = dims[0];
        let d = dims[1];

        if d != self.d_model {
            return Err(KoreError::StorageError(
                format!("input d={} != d_model={}", d, self.d_model),
            ));
        }

        let dh = self.d_head;
        let nh = self.n_heads;
        let nkv = self.n_kv_heads;
        let heads_per_kv = nh / nkv;
        let q_dim = nh * dh;
        let kv_dim = nkv * dh;

        // Project Q: x @ Wq → [seq_len, q_dim]
        let q_tensor = matmul_f32(x, &self.wq)?;
        let q_all = q_tensor.as_f32_slice().unwrap();

        // Project K, V: x @ Wk → [seq_len, kv_dim]
        let k_tensor = matmul_f32(x, &self.wk)?;
        let v_tensor = matmul_f32(x, &self.wv)?;
        let k_all = k_tensor.as_f32_slice().unwrap();
        let v_all = v_tensor.as_f32_slice().unwrap();

        let mut head_outputs = vec![0.0f32; seq_len * q_dim];

        // Iterate over KV head groups: update cache once per KV head,
        // then process all query heads that share this KV head.
        for kv_h in 0..nkv {
            // Extract KV head from k_all/v_all [seq_len, kv_dim]
            let mut k_h = extract_head(k_all, seq_len, kv_dim, kv_h, dh);
            let v_h = extract_head(v_all, seq_len, kv_dim, kv_h, dh);

            // Apply RoPE to K (Q gets its own per query head below)
            if let Some(ref rope) = self.rope {
                rope.apply_single(&mut k_h, self.seq_offset, seq_len);
            }

            // Update KV cache once for this KV head
            let (k_full_tensor, v_full_tensor) = if use_cache {
                let cache = &mut self.kv_caches[kv_h];
                let kt = Tensor::from_f32(&k_h, &[seq_len, dh]);
                let vt = Tensor::from_f32(&v_h, &[seq_len, dh]);
                cache.update(&kt, &vt)?
            } else {
                (
                    Tensor::from_f32(&k_h, &[seq_len, dh]),
                    Tensor::from_f32(&v_h, &[seq_len, dh]),
                )
            };

            let kv_seq_len = k_full_tensor.shape().dims()[0];
            let k_t_transposed = k_full_tensor.transpose()?;

            // Process each query head in this KV group
            for qh_offset in 0..heads_per_kv {
                let h = kv_h * heads_per_kv + qh_offset;

                // Extract Q head h from q_all [seq_len, q_dim]
                let mut q_h = extract_head(q_all, seq_len, q_dim, h, dh);

                // Apply RoPE to Q
                if let Some(ref rope) = self.rope {
                    rope.apply_single(&mut q_h, self.seq_offset, seq_len);
                }

                // scores = Q_h @ K_full^T / sqrt(d_head)  → [seq_len, kv_seq_len]
                let q_t = Tensor::from_f32(&q_h, &[seq_len, dh]);
                let scores_tensor = matmul_f32(&q_t, &k_t_transposed)?;
                let mut scores = scores_tensor.as_f32_slice().unwrap().to_vec();

                let scale = 1.0 / (dh as f32).sqrt();
                for s in scores.iter_mut() {
                    *s *= scale;
                }

                // Apply mask (if provided and dimensions match)
                if let Some(m) = mask {
                    let m_data = m.as_f32_slice().ok_or(KoreError::StorageError("expected f32 mask".into()))?;
                    let m_dims = m.shape().dims();
                    if m_dims[0] >= seq_len && m_dims[1] >= kv_seq_len {
                        for i in 0..seq_len {
                            for j in 0..kv_seq_len {
                                scores[i * kv_seq_len + j] += m_data[i * m_dims[1] + j];
                            }
                        }
                    }
                }

                // Softmax over last dim
                for i in 0..seq_len {
                    let row = &mut scores[i * kv_seq_len..(i + 1) * kv_seq_len];
                    let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for v in row.iter_mut() {
                        *v = (*v - max_val).exp();
                        sum += *v;
                    }
                    if sum > 0.0 {
                        for v in row.iter_mut() {
                            *v /= sum;
                        }
                    }
                }

                // attn_out = scores @ V_full → [seq_len, d_head]
                let scores_t = Tensor::from_f32(&scores, &[seq_len, kv_seq_len]);
                let attn_out_tensor = matmul_f32(&scores_t, &v_full_tensor)?;
                let attn_out = attn_out_tensor.as_f32_slice().unwrap();

                // Write head output back into concatenated position
                for i in 0..seq_len {
                    for j in 0..dh {
                        head_outputs[i * q_dim + h * dh + j] = attn_out[i * dh + j];
                    }
                }
            }
        }

        // Output projection: [seq_len, q_dim] @ Wo → [seq_len, d_model]
        let head_tensor = Tensor::from_f32(&head_outputs, &[seq_len, q_dim]);
        let final_out = matmul_f32(&head_tensor, &self.wo)?;

        // Advance sequence offset for RoPE
        if use_cache {
            self.seq_offset += seq_len;
        }

        Ok(final_out)
    }

    /// Reset all KV caches and sequence offset (call between sequences).
    pub fn reset_cache(&mut self) {
        for cache in &mut self.kv_caches {
            cache.clear();
        }
        self.seq_offset = 0;
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn random_tensor(rows: usize, cols: usize, scale: f32) -> Tensor {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| {
            let x = ((i * 2654435761 + 1013904223) & 0xFFFFFF) as f32 / 0xFFFFFF as f32;
            (x * 2.0 - 1.0) * scale
        })
        .collect();
    Tensor::from_f32(&data, &[rows, cols])
}

fn extract_head(data: &[f32], seq_len: usize, d_model: usize, head: usize, d_head: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * d_head];
    for i in 0..seq_len {
        let src_start = i * d_model + head * d_head;
        let dst_start = i * d_head;
        out[dst_start..dst_start + d_head].copy_from_slice(&data[src_start..src_start + d_head]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mha_shape() {
        let mut mha = MultiHeadAttention::new(64, 4);
        let x = Tensor::from_f32(&vec![0.1; 8 * 64], &[8, 64]);
        let out = mha.forward(&x, None, false).unwrap();
        assert_eq!(out.shape().dims(), &[8, 64]);
    }

    #[test]
    fn test_mha_with_mask() {
        let mut mha = MultiHeadAttention::new(32, 4);
        let x = Tensor::from_f32(&vec![0.1; 4 * 32], &[4, 32]);
        let mask = kore_attention::mask::causal_mask(4);
        let out = mha.forward(&x, Some(&mask), false).unwrap();
        assert_eq!(out.shape().dims(), &[4, 32]);
    }

    #[test]
    fn test_mha_kv_cache() {
        let mut mha = MultiHeadAttention::new(32, 4);
        // First token
        let x1 = Tensor::from_f32(&vec![0.1; 1 * 32], &[1, 32]);
        let out1 = mha.forward(&x1, None, true).unwrap();
        assert_eq!(out1.shape().dims(), &[1, 32]);

        // Second token — KV cache should have seq_len=2 now
        let x2 = Tensor::from_f32(&vec![0.2; 1 * 32], &[1, 32]);
        let out2 = mha.forward(&x2, None, true).unwrap();
        assert_eq!(out2.shape().dims(), &[1, 32]);

        // Reset and verify it works again
        mha.reset_cache();
        let out3 = mha.forward(&x1, None, true).unwrap();
        assert_eq!(out3.shape().dims(), &[1, 32]);
    }

    #[test]
    fn test_gqa_shape() {
        // 8 query heads, 2 KV heads → 4 query heads per KV head
        let mut mha = MultiHeadAttention::new_full(64, 8, 2, false, 128, 10000.0);
        assert_eq!(mha.n_heads, 8);
        assert_eq!(mha.n_kv_heads, 2);
        assert_eq!(mha.d_head, 8); // 64 / 8
        assert_eq!(mha.kv_caches.len(), 2);

        let x = Tensor::from_f32(&vec![0.1; 4 * 64], &[4, 64]);
        let out = mha.forward(&x, None, false).unwrap();
        assert_eq!(out.shape().dims(), &[4, 64]);
    }

    #[test]
    fn test_gqa_with_cache() {
        // 4 query heads, 2 KV heads
        let mut mha = MultiHeadAttention::new_full(32, 4, 2, true, 128, 10000.0);
        let x1 = Tensor::from_f32(&vec![0.1; 1 * 32], &[1, 32]);
        let out1 = mha.forward(&x1, None, true).unwrap();
        assert_eq!(out1.shape().dims(), &[1, 32]);

        let x2 = Tensor::from_f32(&vec![0.2; 1 * 32], &[1, 32]);
        let out2 = mha.forward(&x2, None, true).unwrap();
        assert_eq!(out2.shape().dims(), &[1, 32]);
    }

    #[test]
    fn test_mqa_shape() {
        // Multi-Query Attention: 8 query heads, 1 KV head
        let mut mha = MultiHeadAttention::new_full(64, 8, 1, false, 128, 10000.0);
        assert_eq!(mha.kv_caches.len(), 1);

        let x = Tensor::from_f32(&vec![0.1; 4 * 64], &[4, 64]);
        let out = mha.forward(&x, None, false).unwrap();
        assert_eq!(out.shape().dims(), &[4, 64]);
    }
}
