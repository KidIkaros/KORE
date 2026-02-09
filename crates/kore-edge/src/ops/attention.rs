//! Fused attention operator on raw slices.
//!
//! Single-head attention: softmax(Q·K^T / sqrt(d)) · V
//! Supports causal masking and multi-head via caller slicing.

use super::activation::softmax;

/// Single-head scaled dot-product attention.
///
/// `q`: [seq_q, d] query
/// `k`: [seq_k, d] key
/// `v`: [seq_k, d] value
/// `output`: [seq_q, d] result buffer
/// `scores_buf`: [seq_q, seq_k] scratch buffer for attention scores
/// `causal`: if true, apply causal mask (future positions = -inf)
pub fn attention_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    scores_buf: &mut [f32],
    seq_q: usize,
    seq_k: usize,
    d: usize,
    causal: bool,
) {
    let scale = 1.0 / (d as f32).sqrt();

    // scores = Q @ K^T / sqrt(d)
    for i in 0..seq_q {
        for j in 0..seq_k {
            let mut dot = 0.0f32;
            for p in 0..d {
                dot += q[i * d + p] * k[j * d + p];
            }
            scores_buf[i * seq_k + j] = dot * scale;
        }
    }

    // Causal mask: set future positions to -inf
    if causal {
        for i in 0..seq_q {
            for j in (i + 1)..seq_k {
                scores_buf[i * seq_k + j] = f32::NEG_INFINITY;
            }
        }
    }

    // Softmax over each row
    softmax(scores_buf, seq_k);

    // output = scores @ V
    for i in 0..seq_q {
        for p in 0..d {
            let mut acc = 0.0f32;
            for j in 0..seq_k {
                acc += scores_buf[i * seq_k + j] * v[j * d + p];
            }
            output[i * d + p] = acc;
        }
    }
}

/// Multi-head attention dispatching to per-head attention_head.
///
/// `q`: [seq_q, n_heads * head_dim]
/// `k`: [seq_k, n_kv_heads * head_dim]
/// `v`: [seq_k, n_kv_heads * head_dim]
/// `output`: [seq_q, n_heads * head_dim]
/// `scores_buf`: [n_heads, seq_q, seq_k] scratch
/// Supports GQA: n_heads can be a multiple of n_kv_heads.
pub fn multi_head_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    scores_buf: &mut [f32],
    seq_q: usize,
    seq_k: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    causal: bool,
) {
    let heads_per_kv = n_heads / n_kv_heads;
    let q_stride = n_heads * head_dim;
    let k_stride = n_kv_heads * head_dim;
    let score_size = seq_q * seq_k;

    for h in 0..n_heads {
        let kv_h = h / heads_per_kv;

        // Extract per-head slices by gathering from interleaved layout
        // For simplicity, we work with strided access
        let score_slice = &mut scores_buf[h * score_size..(h + 1) * score_size];

        // Q·K^T for this head
        let scale = 1.0 / (head_dim as f32).sqrt();
        for i in 0..seq_q {
            for j in 0..seq_k {
                let mut dot = 0.0f32;
                for p in 0..head_dim {
                    dot += q[i * q_stride + h * head_dim + p]
                        * k[j * k_stride + kv_h * head_dim + p];
                }
                score_slice[i * seq_k + j] = dot * scale;
            }
        }

        // Causal mask
        if causal {
            for i in 0..seq_q {
                for j in (i + 1)..seq_k {
                    score_slice[i * seq_k + j] = f32::NEG_INFINITY;
                }
            }
        }

        // Softmax
        softmax(score_slice, seq_k);

        // scores @ V → output for this head
        for i in 0..seq_q {
            for p in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..seq_k {
                    acc += score_slice[i * seq_k + j]
                        * v[j * k_stride + kv_h * head_dim + p];
                }
                output[i * q_stride + h * head_dim + p] = acc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_head_shape() {
        let d = 4;
        let seq = 3;
        let q = vec![1.0f32; seq * d];
        let k = vec![1.0f32; seq * d];
        let v = vec![1.0f32; seq * d];
        let mut output = vec![0.0f32; seq * d];
        let mut scores = vec![0.0f32; seq * seq];

        attention_head(&q, &k, &v, &mut output, &mut scores, seq, seq, d, false);

        // With uniform Q,K,V, output should be all 1.0
        for &v in &output {
            assert!((v - 1.0).abs() < 1e-4, "got {}", v);
        }
    }

    #[test]
    fn test_attention_causal() {
        let d = 2;
        let seq = 3;
        let q = vec![1.0; seq * d];
        let k = vec![1.0; seq * d];
        let v: Vec<f32> = (0..seq * d).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; seq * d];
        let mut scores = vec![0.0f32; seq * seq];

        attention_head(&q, &k, &v, &mut output, &mut scores, seq, seq, d, true);

        // First token can only attend to itself
        assert!((output[0] - v[0]).abs() < 1e-4);
        assert!((output[1] - v[1]).abs() < 1e-4);
    }

    #[test]
    fn test_multi_head_attention() {
        let seq = 2;
        let n_heads = 2;
        let n_kv_heads = 2;
        let head_dim = 2;
        let full_dim = n_heads * head_dim;

        let q = vec![1.0f32; seq * full_dim];
        let k = vec![1.0f32; seq * full_dim];
        let v = vec![1.0f32; seq * full_dim];
        let mut output = vec![0.0f32; seq * full_dim];
        let mut scores = vec![0.0f32; n_heads * seq * seq];

        multi_head_attention(
            &q, &k, &v, &mut output, &mut scores,
            seq, seq, n_heads, n_kv_heads, head_dim, false,
        );

        for &v in &output {
            assert!((v - 1.0).abs() < 1e-4);
        }
    }
}
