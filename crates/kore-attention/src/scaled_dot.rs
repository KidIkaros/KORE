//! Standard scaled dot-product attention.
//!
//! Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k) + mask) @ V
//!
//! This is the baseline implementation. For memory-efficient attention,
//! see the `flash` module.

use kore_core::{DType, KoreError, Tensor};

/// Scaled dot-product attention.
///
/// # Arguments
/// * `query`  - [batch, seq_q, d_k] or [seq_q, d_k]
/// * `key`    - [batch, seq_k, d_k] or [seq_k, d_k]
/// * `value`  - [batch, seq_k, d_v] or [seq_k, d_v]
/// * `mask`   - Optional [seq_q, seq_k] additive mask (-inf for masked positions)
/// * `scale`  - Optional scaling factor (default: 1/sqrt(d_k))
///
/// # Returns
/// * `output` - [batch, seq_q, d_v] or [seq_q, d_v]
pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<&Tensor>,
    scale: Option<f32>,
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

    // Support both 2D [seq, d] and 3D [batch, seq, d]
    let (batch, seq_q, d_k, seq_k, d_v) = match (q_dims.len(), k_dims.len(), v_dims.len()) {
        (2, 2, 2) => (1, q_dims[0], q_dims[1], k_dims[0], v_dims[1]),
        (3, 3, 3) => {
            if q_dims[0] != k_dims[0] || q_dims[0] != v_dims[0] {
                return Err(KoreError::ShapeMismatch {
                    expected: q_dims.to_vec(),
                    got: k_dims.to_vec(),
                });
            }
            (q_dims[0], q_dims[1], q_dims[2], k_dims[1], v_dims[2])
        }
        _ => {
            return Err(KoreError::ShapeMismatch {
                expected: vec![0, 0, 0],
                got: q_dims.to_vec(),
            });
        }
    };

    if q_dims[q_dims.len() - 1] != k_dims[k_dims.len() - 1] {
        return Err(KoreError::MatmulDimMismatch {
            m: seq_q,
            k1: d_k,
            k2: k_dims[k_dims.len() - 1],
            n: seq_k,
        });
    }

    let scale_factor = scale.unwrap_or(1.0 / (d_k as f32).sqrt());

    let q_data = q.as_f32_slice().unwrap();
    let k_data = k.as_f32_slice().unwrap();
    let v_data = v.as_f32_slice().unwrap();

    let mask_data = mask.map(|m| m.contiguous());
    let mask_slice = mask_data.as_ref().and_then(|m| m.as_f32_slice());

    let mut output = vec![0.0f32; batch * seq_q * d_v];

    for b in 0..batch {
        let q_off = b * seq_q * d_k;
        let k_off = b * seq_k * d_k;
        let v_off = b * seq_k * d_v;
        let o_off = b * seq_q * d_v;

        // scores = Q @ K^T * scale  [seq_q, seq_k]
        let mut scores = vec![0.0f32; seq_q * seq_k];
        for i in 0..seq_q {
            for j in 0..seq_k {
                let mut dot = 0.0f32;
                for p in 0..d_k {
                    dot += q_data[q_off + i * d_k + p] * k_data[k_off + j * d_k + p];
                }
                scores[i * seq_k + j] = dot * scale_factor;
            }
        }

        // Apply mask (additive)
        if let Some(m) = mask_slice {
            for i in 0..seq_q {
                for j in 0..seq_k {
                    scores[i * seq_k + j] += m[i * seq_k + j];
                }
            }
        }

        // Softmax over last dimension (seq_k)
        for i in 0..seq_q {
            let row_start = i * seq_k;
            let row = &mut scores[row_start..row_start + seq_k];

            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            let inv_sum = 1.0 / sum;
            for v in row.iter_mut() {
                *v *= inv_sum;
            }
        }

        // output = attn_weights @ V  [seq_q, d_v]
        for i in 0..seq_q {
            for j in 0..d_v {
                let mut acc = 0.0f32;
                for p in 0..seq_k {
                    acc += scores[i * seq_k + p] * v_data[v_off + p * d_v + j];
                }
                output[o_off + i * d_v + j] = acc;
            }
        }
    }

    if q_dims.len() == 2 {
        Ok(Tensor::from_f32(&output, &[seq_q, d_v]))
    } else {
        Ok(Tensor::from_f32(&output, &[batch, seq_q, d_v]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mask::causal_mask;

    #[test]
    fn test_self_attention_2d() {
        // Simple self-attention: Q=K=V
        let x = Tensor::from_f32(
            &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            &[3, 2],
        );
        let out = scaled_dot_product_attention(&x, &x, &x, None, None).unwrap();
        assert_eq!(out.shape().dims(), &[3, 2]);

        // Output should be finite
        let data = out.as_f32_slice().unwrap();
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_attention_with_causal_mask() {
        let seq_len = 4;
        let d = 2;
        let q_data: Vec<f32> = (0..seq_len * d).map(|i| i as f32 * 0.1).collect();
        let q = Tensor::from_f32(&q_data, &[seq_len, d]);
        let mask = causal_mask(seq_len);

        let out = scaled_dot_product_attention(&q, &q, &q, Some(&mask), None).unwrap();
        assert_eq!(out.shape().dims(), &[seq_len, d]);

        let data = out.as_f32_slice().unwrap();
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_attention_3d_batched() {
        let batch = 2;
        let seq = 3;
        let d = 4;
        let data: Vec<f32> = (0..batch * seq * d).map(|i| (i % 7) as f32 * 0.1).collect();

        let q = Tensor::from_f32(&data, &[batch, seq, d]);
        let k = Tensor::from_f32(&data, &[batch, seq, d]);
        let v = Tensor::from_f32(&data, &[batch, seq, d]);

        let out = scaled_dot_product_attention(&q, &k, &v, None, None).unwrap();
        assert_eq!(out.shape().dims(), &[batch, seq, d]);
    }

    #[test]
    fn test_attention_uniform_weights() {
        // With identical Q and K, attention should be roughly uniform
        let q = Tensor::ones(&[4, 2]);
        let k = Tensor::ones(&[4, 2]);
        let v = Tensor::from_f32(
            &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            &[4, 2],
        );

        let out = scaled_dot_product_attention(&q, &k, &v, None, None).unwrap();
        let data = out.as_f32_slice().unwrap();

        // All rows should be the same (mean of V rows)
        let expected_0 = (1.0 + 0.0 + 1.0 + 0.0) / 4.0;
        let expected_1 = (0.0 + 1.0 + 1.0 + 0.0) / 4.0;
        assert!((data[0] - expected_0).abs() < 1e-5);
        assert!((data[1] - expected_1).abs() < 1e-5);
    }
}
