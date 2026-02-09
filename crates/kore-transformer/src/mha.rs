//! Multi-Head Self-Attention with optional KV-cache.

use kore_core::{KoreError, Tensor};
use kore_attention::kv_cache::KvCache;

/// Multi-Head Attention layer.
///
/// Projects input into Q, K, V heads, runs scaled dot-product attention,
/// then projects back to d_model.
pub struct MultiHeadAttention {
    pub n_heads: usize,
    pub d_model: usize,
    pub d_head: usize,
    /// [d_model, d_model] — Q projection
    pub wq: Tensor,
    /// [d_model, d_model] — K projection
    pub wk: Tensor,
    /// [d_model, d_model] — V projection
    pub wv: Tensor,
    /// [d_model, d_model] — output projection
    pub wo: Tensor,
    /// Per-head KV caches (for autoregressive generation)
    pub kv_caches: Vec<KvCache>,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        assert!(d_model % n_heads == 0, "d_model must be divisible by n_heads");
        let d_head = d_model / n_heads;
        let scale = (1.0 / d_model as f64).sqrt() as f32;

        let wq = random_tensor(d_model, d_model, scale);
        let wk = random_tensor(d_model, d_model, scale);
        let wv = random_tensor(d_model, d_model, scale);
        let wo = random_tensor(d_model, d_model, scale);

        let kv_caches = (0..n_heads).map(|_| KvCache::new(d_head, d_head, 4096)).collect();

        Self { n_heads, d_model, d_head, wq, wk, wv, wo, kv_caches }
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
        let x_data = x.as_f32_slice().ok_or(KoreError::StorageError("expected f32 tensor".into()))?;
        let dims = x.shape().dims();
        let seq_len = dims[0];
        let d = dims[1];

        if d != self.d_model {
            return Err(KoreError::StorageError(
                format!("input d={} != d_model={}", d, self.d_model),
            ));
        }

        // Project: Q, K, V = x @ W^T  → [seq_len, d_model]
        let q_all = matmul_2d(x_data, self.wq.as_f32_slice().unwrap(), seq_len, d, d);
        let k_all = matmul_2d(x_data, self.wk.as_f32_slice().unwrap(), seq_len, d, d);
        let v_all = matmul_2d(x_data, self.wv.as_f32_slice().unwrap(), seq_len, d, d);

        let dh = self.d_head;
        let nh = self.n_heads;
        let mut head_outputs = vec![0.0f32; seq_len * d];

        for h in 0..nh {
            // Extract head h: [seq_len, d_head]
            let q_h = extract_head(&q_all, seq_len, d, h, dh);
            let k_h = extract_head(&k_all, seq_len, d, h, dh);
            let v_h = extract_head(&v_all, seq_len, d, h, dh);

            // Optionally use KV cache
            let (k_full, v_full, kv_seq_len) = if use_cache {
                let cache = &mut self.kv_caches[h];
                let k_tensor = Tensor::from_f32(&k_h, &[seq_len, dh]);
                let v_tensor = Tensor::from_f32(&v_h, &[seq_len, dh]);
                let (ck, cv) = cache.update(&k_tensor, &v_tensor)?;
                let ksl = ck.shape().dims()[0];
                let ck_data = ck.as_f32_slice().unwrap().to_vec();
                let cv_data = cv.as_f32_slice().unwrap().to_vec();
                (ck_data, cv_data, ksl)
            } else {
                (k_h.clone(), v_h.clone(), seq_len)
            };

            // Scaled dot-product attention for this head
            // scores = Q @ K^T / sqrt(d_head)  → [seq_len, kv_seq_len]
            let scale = 1.0 / (dh as f32).sqrt();
            let mut scores = vec![0.0f32; seq_len * kv_seq_len];
            for i in 0..seq_len {
                for j in 0..kv_seq_len {
                    let mut dot = 0.0f32;
                    for k in 0..dh {
                        dot += q_h[i * dh + k] * k_full[j * dh + k];
                    }
                    scores[i * kv_seq_len + j] = dot * scale;
                }
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

            // attn_out = scores @ V → [seq_len, d_head]
            let mut attn_out = vec![0.0f32; seq_len * dh];
            for i in 0..seq_len {
                for j in 0..dh {
                    let mut acc = 0.0f32;
                    for k in 0..kv_seq_len {
                        acc += scores[i * kv_seq_len + k] * v_full[k * dh + j];
                    }
                    attn_out[i * dh + j] = acc;
                }
            }

            // Write head output back into concatenated position
            for i in 0..seq_len {
                for j in 0..dh {
                    head_outputs[i * d + h * dh + j] = attn_out[i * dh + j];
                }
            }
        }

        // Output projection: [seq_len, d_model] @ Wo^T → [seq_len, d_model]
        let wo_data = self.wo.as_f32_slice().unwrap();
        let final_out = matmul_2d(&head_outputs, wo_data, seq_len, d, d);

        Ok(Tensor::from_f32(&final_out, &[seq_len, d]))
    }

    /// Reset all KV caches (call between sequences).
    pub fn reset_cache(&mut self) {
        for cache in &mut self.kv_caches {
            cache.clear();
        }
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

/// C = A @ B^T where A is [m, k], B is [n, k] → C is [m, n]
fn matmul_2d(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
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
}
