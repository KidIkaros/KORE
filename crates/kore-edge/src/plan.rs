//! Execution plan — memory planning and operator fusion for edge inference.
//!
//! Analyzes the model graph to compute peak memory usage and determine
//! which operations can be fused or executed in-place.

use crate::format::KorefHeader;

/// Memory requirement for a single buffer in the execution plan.
#[derive(Debug, Clone)]
pub struct BufferSlot {
    pub name: String,
    pub size_bytes: usize,
    pub offset: usize,
}

/// A fused operation group (multiple ops executed as one).
#[derive(Debug, Clone)]
pub enum FusedOp {
    /// Embedding lookup
    Embedding { weight_name: String, dim: usize },
    /// Linear projection (matmul + optional bias)
    Linear { weight_name: String, bias_name: Option<String>, m: usize, n: usize, k: usize },
    /// RMS normalization
    RMSNorm { gamma_name: String, dim: usize, eps: f32 },
    /// RoPE application
    RoPE { n_heads: usize, head_dim: usize, base: f32 },
    /// Multi-head attention (fused QKV → scores → output)
    Attention { n_heads: usize, n_kv_heads: usize, head_dim: usize, causal: bool },
    /// SwiGLU feed-forward: silu(x @ w1) * (x @ w3) @ w2
    FeedForward {
        w1_name: String,
        w2_name: String,
        w3_name: String,
        d_model: usize,
        d_ff: usize,
    },
    /// Residual add (in-place)
    ResidualAdd,
    /// Final linear projection to vocab logits
    FinalProj { weight_name: String, vocab_size: usize, d_model: usize },
}

/// Complete execution plan for a transformer model.
pub struct ExecutionPlan {
    pub ops: Vec<FusedOp>,
    pub peak_memory: usize,
    pub kv_cache_size: usize,
    pub n_layers: usize,
}

impl ExecutionPlan {
    /// Build an execution plan from a .koref header.
    pub fn from_header(header: &KorefHeader) -> Self {
        let d = header.d_model;
        let n_heads = header.n_heads;
        let n_kv_heads = header.n_kv_heads;
        let head_dim = d / n_heads;
        let d_ff = header.d_ff;
        let n_layers = header.n_layers;
        let eps = header.norm_eps;
        let rope_base = header.rope_base;

        let mut ops = Vec::new();

        // Embedding
        ops.push(FusedOp::Embedding {
            weight_name: "model.embed_tokens.weight".into(),
            dim: d,
        });

        // Transformer blocks
        for layer in 0..n_layers {
            let prefix = format!("model.layers.{}", layer);

            // Attention norm
            ops.push(FusedOp::RMSNorm {
                gamma_name: format!("{}.input_layernorm.weight", prefix),
                dim: d,
                eps,
            });

            // QKV projections (3 separate linears)
            ops.push(FusedOp::Linear {
                weight_name: format!("{}.self_attn.q_proj.weight", prefix),
                bias_name: None,
                m: 0, // seq_len determined at runtime
                n: n_heads * head_dim,
                k: d,
            });
            ops.push(FusedOp::Linear {
                weight_name: format!("{}.self_attn.k_proj.weight", prefix),
                bias_name: None,
                m: 0,
                n: n_kv_heads * head_dim,
                k: d,
            });
            ops.push(FusedOp::Linear {
                weight_name: format!("{}.self_attn.v_proj.weight", prefix),
                bias_name: None,
                m: 0,
                n: n_kv_heads * head_dim,
                k: d,
            });

            // RoPE
            ops.push(FusedOp::RoPE { n_heads, head_dim, base: rope_base });

            // Attention
            ops.push(FusedOp::Attention {
                n_heads,
                n_kv_heads,
                head_dim,
                causal: true,
            });

            // Output projection
            ops.push(FusedOp::Linear {
                weight_name: format!("{}.self_attn.o_proj.weight", prefix),
                bias_name: None,
                m: 0,
                n: d,
                k: n_heads * head_dim,
            });

            // Residual
            ops.push(FusedOp::ResidualAdd);

            // FFN norm
            ops.push(FusedOp::RMSNorm {
                gamma_name: format!("{}.post_attention_layernorm.weight", prefix),
                dim: d,
                eps,
            });

            // Feed-forward (SwiGLU)
            ops.push(FusedOp::FeedForward {
                w1_name: format!("{}.mlp.gate_proj.weight", prefix),
                w2_name: format!("{}.mlp.down_proj.weight", prefix),
                w3_name: format!("{}.mlp.up_proj.weight", prefix),
                d_model: d,
                d_ff,
            });

            // Residual
            ops.push(FusedOp::ResidualAdd);
        }

        // Final norm
        ops.push(FusedOp::RMSNorm {
            gamma_name: "model.norm.weight".into(),
            dim: d,
            eps,
        });

        // LM head
        ops.push(FusedOp::FinalProj {
            weight_name: "lm_head.weight".into(),
            vocab_size: header.vocab_size,
            d_model: d,
        });

        // Estimate peak memory (for single-token generation)
        // Activations: max(d_model, d_ff, vocab_size) * 4 bytes * 2 (double buffer)
        let max_dim = d.max(d_ff).max(header.vocab_size);
        let activation_mem = max_dim * 4 * 2;
        // Attention scores: n_heads * max_seq * 4
        let score_mem = n_heads * header.max_seq_len * 4;
        // KV cache: 2 * n_layers * n_kv_heads * head_dim * max_seq * 4
        let kv_cache_size = 2 * n_layers * n_kv_heads * head_dim * header.max_seq_len * 4;

        let peak_memory = activation_mem + score_mem + kv_cache_size;

        ExecutionPlan {
            ops,
            peak_memory,
            kv_cache_size,
            n_layers,
        }
    }

    /// Number of operations in the plan.
    pub fn num_ops(&self) -> usize {
        self.ops.len()
    }

    /// Estimated peak memory in bytes.
    pub fn peak_memory_bytes(&self) -> usize {
        self.peak_memory
    }

    /// Estimated peak memory in MB.
    pub fn peak_memory_mb(&self) -> f32 {
        self.peak_memory as f32 / (1024.0 * 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn tiny_header() -> KorefHeader {
        KorefHeader {
            model_type: "llama".into(),
            vocab_size: 256,
            d_model: 64,
            n_heads: 4,
            n_kv_heads: 4,
            n_layers: 2,
            d_ff: 128,
            max_seq_len: 128,
            norm_eps: 1e-5,
            rope_base: 10000.0,
            tensors: HashMap::new(),
            ops: Vec::new(),
        }
    }

    #[test]
    fn test_plan_from_header() {
        let header = tiny_header();
        let plan = ExecutionPlan::from_header(&header);

        // Should have ops for: embed + 2 layers * (norm + 3 proj + rope + attn + o_proj + res + norm + ffn + res) + final_norm + lm_head
        assert!(plan.num_ops() > 10);
        assert!(plan.peak_memory_bytes() > 0);
        assert_eq!(plan.n_layers, 2);
    }

    #[test]
    fn test_plan_memory_estimate() {
        let header = tiny_header();
        let plan = ExecutionPlan::from_header(&header);
        // Should be reasonable for a tiny model
        assert!(plan.peak_memory_mb() < 10.0, "peak={}MB", plan.peak_memory_mb());
    }
}
