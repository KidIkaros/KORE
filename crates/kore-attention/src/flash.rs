//! Flash Attention v2 — memory-efficient tiled attention.
//!
//! Computes attention in O(n) memory instead of O(n²) by processing
//! blocks of Q, K, V and maintaining running softmax statistics.
//!
//! Reference: Dao et al., "FlashAttention-2: Faster Attention with Better
//! Parallelism and Work Partitioning" (2023).

use kore_core::{DType, KoreError, Tensor};

/// Block size for tiled computation.
/// Chosen to fit in L1/L2 cache for typical d_k values.
const BLOCK_SIZE: usize = 64;

/// Flash Attention: memory-efficient scaled dot-product attention.
///
/// Instead of materializing the full [seq_q, seq_k] attention matrix,
/// processes blocks and maintains running softmax statistics (online softmax).
///
/// # Arguments
/// * `query`  - [seq_q, d_k]
/// * `key`    - [seq_k, d_k]
/// * `value`  - [seq_k, d_v]
/// * `causal` - Whether to apply causal masking
///
/// # Returns
/// * `output` - [seq_q, d_v]
pub fn flash_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    causal: bool,
) -> Result<Tensor, KoreError> {
    if query.dtype() != DType::F32 {
        return Err(KoreError::UnsupportedDType(query.dtype()));
    }

    let q = query.contiguous();
    let k = key.contiguous();
    let v = value.contiguous();

    let q_dims = q.shape().dims();
    let k_dims = k.shape().dims();
    let v_dims = v.shape().dims();

    if q_dims.len() != 2 || k_dims.len() != 2 || v_dims.len() != 2 {
        return Err(KoreError::ShapeMismatch {
            expected: vec![0, 0],
            got: q_dims.to_vec(),
        });
    }

    let seq_q = q_dims[0];
    let d_k = q_dims[1];
    let seq_k = k_dims[0];
    let d_v = v_dims[1];

    if d_k != k_dims[1] {
        return Err(KoreError::MatmulDimMismatch {
            m: seq_q, k1: d_k, k2: k_dims[1], n: seq_k,
        });
    }

    let scale = 1.0 / (d_k as f32).sqrt();

    let q_data = q.as_f32_slice().unwrap();
    let k_data = k.as_f32_slice().unwrap();
    let v_data = v.as_f32_slice().unwrap();

    // Output accumulator
    let mut output = vec![0.0f32; seq_q * d_v];
    // Running softmax statistics per query row
    let mut row_max = vec![f32::NEG_INFINITY; seq_q];   // m_i
    let mut row_sum = vec![0.0f32; seq_q];               // l_i

    // Process K/V in blocks
    for kv_start in (0..seq_k).step_by(BLOCK_SIZE) {
        let kv_end = (kv_start + BLOCK_SIZE).min(seq_k);
        let kv_block_size = kv_end - kv_start;

        // Process Q in blocks
        for q_start in (0..seq_q).step_by(BLOCK_SIZE) {
            let q_end = (q_start + BLOCK_SIZE).min(seq_q);

            for qi in q_start..q_end {
                // Compute scores for this query row against the KV block
                let mut block_scores = vec![0.0f32; kv_block_size];
                let mut block_max = f32::NEG_INFINITY;

                for (bj, kj) in (kv_start..kv_end).enumerate() {
                    // Causal mask: skip future positions
                    if causal && kj > qi {
                        block_scores[bj] = f32::NEG_INFINITY;
                        continue;
                    }

                    let mut dot = 0.0f32;
                    for p in 0..d_k {
                        dot += q_data[qi * d_k + p] * k_data[kj * d_k + p];
                    }
                    let score = dot * scale;
                    block_scores[bj] = score;
                    if score > block_max {
                        block_max = score;
                    }
                }

                // Online softmax update
                let prev_max = row_max[qi];
                let new_max = prev_max.max(block_max);

                // Rescale previous accumulator
                let prev_scale = (prev_max - new_max).exp();
                row_sum[qi] *= prev_scale;
                for j in 0..d_v {
                    output[qi * d_v + j] *= prev_scale;
                }

                // Add contribution from this block
                let mut block_sum = 0.0f32;
                for bj in 0..kv_block_size {
                    let w = (block_scores[bj] - new_max).exp();
                    block_sum += w;

                    let kj = kv_start + bj;
                    for j in 0..d_v {
                        output[qi * d_v + j] += w * v_data[kj * d_v + j];
                    }
                }

                row_max[qi] = new_max;
                row_sum[qi] += block_sum;
            }
        }
    }

    // Final normalization
    for qi in 0..seq_q {
        let inv_sum = if row_sum[qi] > 0.0 { 1.0 / row_sum[qi] } else { 0.0 };
        for j in 0..d_v {
            output[qi * d_v + j] *= inv_sum;
        }
    }

    Ok(Tensor::from_f32(&output, &[seq_q, d_v]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scaled_dot::scaled_dot_product_attention;
    use crate::mask::causal_mask;

    #[test]
    fn test_flash_matches_standard() {
        // Flash attention should produce the same results as standard attention
        let seq = 16;
        let d = 8;
        let data: Vec<f32> = (0..seq * d).map(|i| ((i * 7 + 3) % 13) as f32 * 0.1 - 0.6).collect();

        let q = Tensor::from_f32(&data, &[seq, d]);
        let k = Tensor::from_f32(&data, &[seq, d]);
        let v = Tensor::from_f32(&data, &[seq, d]);

        let standard = scaled_dot_product_attention(&q, &k, &v, None, None).unwrap();
        let flash = flash_attention(&q, &k, &v, false).unwrap();

        let std_data = standard.as_f32_slice().unwrap();
        let flash_data = flash.as_f32_slice().unwrap();

        for i in 0..seq * d {
            assert!(
                (std_data[i] - flash_data[i]).abs() < 1e-4,
                "Mismatch at {}: std={}, flash={}",
                i, std_data[i], flash_data[i]
            );
        }
    }

    #[test]
    fn test_flash_causal_matches_standard() {
        let seq = 16;
        let d = 8;
        let data: Vec<f32> = (0..seq * d).map(|i| ((i * 7 + 3) % 13) as f32 * 0.1 - 0.6).collect();

        let q = Tensor::from_f32(&data, &[seq, d]);
        let k = Tensor::from_f32(&data, &[seq, d]);
        let v = Tensor::from_f32(&data, &[seq, d]);

        let mask = causal_mask(seq);
        let standard = scaled_dot_product_attention(&q, &k, &v, Some(&mask), None).unwrap();
        let flash = flash_attention(&q, &k, &v, true).unwrap();

        let std_data = standard.as_f32_slice().unwrap();
        let flash_data = flash.as_f32_slice().unwrap();

        for i in 0..seq * d {
            assert!(
                (std_data[i] - flash_data[i]).abs() < 1e-4,
                "Mismatch at {}: std={}, flash={}",
                i, std_data[i], flash_data[i]
            );
        }
    }

    #[test]
    fn test_flash_larger_than_block() {
        // Test with sequence longer than BLOCK_SIZE
        let seq = BLOCK_SIZE + 32;
        let d = 4;
        let data: Vec<f32> = (0..seq * d).map(|i| (i % 5) as f32 * 0.2 - 0.4).collect();

        let q = Tensor::from_f32(&data, &[seq, d]);
        let k = Tensor::from_f32(&data, &[seq, d]);
        let v = Tensor::from_f32(&data, &[seq, d]);

        let out = flash_attention(&q, &k, &v, false).unwrap();
        assert_eq!(out.shape().dims(), &[seq, d]);

        let out_data = out.as_f32_slice().unwrap();
        assert!(out_data.iter().all(|v| v.is_finite()));
    }
}
