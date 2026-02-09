//! Inference session — loads a .koref model and runs forward passes.
//!
//! Manages KV-cache, arena allocation, and operator dispatch.

use crate::arena::Arena;
use crate::format::KorefModel;
use crate::ops::{activation, attention, elementwise, embedding, matmul, norm, rope};
use crate::plan::ExecutionPlan;

/// KV-cache for autoregressive generation.
struct KvCache {
    /// keys[layer]: [max_seq, n_kv_heads * head_dim]
    keys: Vec<Vec<f32>>,
    /// values[layer]: [max_seq, n_kv_heads * head_dim]
    values: Vec<Vec<f32>>,
    /// Current sequence length stored in cache.
    len: usize,
    _max_seq: usize,
    kv_dim: usize,
}

impl KvCache {
    fn new(n_layers: usize, max_seq: usize, kv_dim: usize) -> Self {
        Self {
            keys: (0..n_layers).map(|_| vec![0.0f32; max_seq * kv_dim]).collect(),
            values: (0..n_layers).map(|_| vec![0.0f32; max_seq * kv_dim]).collect(),
            len: 0,
            _max_seq: max_seq,
            kv_dim,
        }
    }

    fn append_kv(&mut self, layer: usize, k: &[f32], v: &[f32], seq_len: usize) {
        let start = self.len * self.kv_dim;
        let end = start + seq_len * self.kv_dim;
        self.keys[layer][start..end].copy_from_slice(&k[..seq_len * self.kv_dim]);
        self.values[layer][start..end].copy_from_slice(&v[..seq_len * self.kv_dim]);
    }

    fn advance(&mut self, seq_len: usize) {
        self.len += seq_len;
    }

    fn clear(&mut self) {
        self.len = 0;
    }
}

/// Inference session for a loaded .koref model.
pub struct Session {
    model: KorefModel,
    plan: ExecutionPlan,
    kv_cache: KvCache,
    arena: Arena,
    // Model config cached for fast access
    d_model: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    d_ff: usize,
    vocab_size: usize,
    max_seq_len: usize,
}

impl Session {
    /// Create a new inference session from a loaded model.
    pub fn new(model: KorefModel) -> Self {
        let plan = ExecutionPlan::from_header(&model.header);
        let d_model = model.header.d_model;
        let n_heads = model.header.n_heads;
        let n_kv_heads = model.header.n_kv_heads;
        let head_dim = d_model / n_heads;
        let d_ff = model.header.d_ff;
        let vocab_size = model.header.vocab_size;
        let max_seq_len = model.header.max_seq_len;
        let n_layers = model.header.n_layers;
        let kv_dim = n_kv_heads * head_dim;

        // Arena: enough for activations + scratch
        let arena_size = plan.peak_memory.max(1024 * 1024); // at least 1MB
        let arena = Arena::new(arena_size);

        let kv_cache = KvCache::new(n_layers, max_seq_len, kv_dim);

        Self {
            model,
            plan,
            kv_cache,
            arena,
            d_model,
            n_heads,
            n_kv_heads,
            head_dim,
            d_ff,
            vocab_size,
            max_seq_len,
        }
    }

    /// Run a single forward pass on input token IDs.
    /// Returns logits [vocab_size] for the last token position.
    pub fn forward(&mut self, input_ids: &[u32]) -> Vec<f32> {
        let seq_len = input_ids.len();
        let d = self.d_model;
        let n_layers = self.plan.n_layers;
        let pos = self.kv_cache.len;

        self.arena.reset();

        // Embedding lookup → hidden [seq_len, d_model]
        let mut hidden = vec![0.0f32; seq_len * d];
        if let Some(embed_w) = self.model.tensor_f32("model.embed_tokens.weight") {
            embedding::embedding_lookup(embed_w, input_ids, &mut hidden, self.vocab_size, d);
        }

        // Transformer layers
        for layer in 0..n_layers {
            let prefix = format!("model.layers.{}", layer);

            // Save residual
            let residual: Vec<f32> = hidden.clone();

            // Attention norm
            if let Some(gamma) = self.model.tensor_f32(&format!("{}.input_layernorm.weight", prefix)) {
                norm::rms_norm(&mut hidden, gamma, d, self.model.header.norm_eps);
            }

            // QKV projections
            let mut q = vec![0.0f32; seq_len * self.n_heads * self.head_dim];
            let mut k = vec![0.0f32; seq_len * self.n_kv_heads * self.head_dim];
            let mut v = vec![0.0f32; seq_len * self.n_kv_heads * self.head_dim];

            self.project(&hidden, &mut q, seq_len, self.n_heads * self.head_dim, d,
                &format!("{}.self_attn.q_proj.weight", prefix));
            self.project(&hidden, &mut k, seq_len, self.n_kv_heads * self.head_dim, d,
                &format!("{}.self_attn.k_proj.weight", prefix));
            self.project(&hidden, &mut v, seq_len, self.n_kv_heads * self.head_dim, d,
                &format!("{}.self_attn.v_proj.weight", prefix));

            // RoPE
            rope::apply_rope(&mut q, seq_len, self.n_heads, self.head_dim, pos, self.model.header.rope_base);
            rope::apply_rope(&mut k, seq_len, self.n_kv_heads, self.head_dim, pos, self.model.header.rope_base);

            // Update KV cache
            self.kv_cache.append_kv(layer, &k, &v, seq_len);
            let _total_seq = self.kv_cache.len + seq_len; // will be advanced after all layers

            // Attention (using full KV cache)
            let kv_len = pos + seq_len;
            let kv_dim = self.n_kv_heads * self.head_dim;
            let mut attn_out = vec![0.0f32; seq_len * self.n_heads * self.head_dim];
            let mut scores = vec![0.0f32; self.n_heads * seq_len * kv_len];

            attention::multi_head_attention(
                &q,
                &self.kv_cache.keys[layer][..kv_len * kv_dim],
                &self.kv_cache.values[layer][..kv_len * kv_dim],
                &mut attn_out,
                &mut scores,
                seq_len,
                kv_len,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                true,
            );

            // Output projection
            let mut attn_proj = vec![0.0f32; seq_len * d];
            self.project(&attn_out, &mut attn_proj, seq_len, d, self.n_heads * self.head_dim,
                &format!("{}.self_attn.o_proj.weight", prefix));

            // Residual add
            elementwise::residual_add(&residual, &attn_proj, &mut hidden);

            // FFN
            let residual2: Vec<f32> = hidden.clone();

            // FFN norm
            if let Some(gamma) = self.model.tensor_f32(&format!("{}.post_attention_layernorm.weight", prefix)) {
                norm::rms_norm(&mut hidden, gamma, d, self.model.header.norm_eps);
            }

            // SwiGLU: silu(x @ w1) * (x @ w3) @ w2
            let mut gate = vec![0.0f32; seq_len * self.d_ff];
            let mut up = vec![0.0f32; seq_len * self.d_ff];

            self.project(&hidden, &mut gate, seq_len, self.d_ff, d,
                &format!("{}.mlp.gate_proj.weight", prefix));
            self.project(&hidden, &mut up, seq_len, self.d_ff, d,
                &format!("{}.mlp.up_proj.weight", prefix));

            activation::silu(&mut gate);
            let mut gate_up = vec![0.0f32; seq_len * self.d_ff];
            elementwise::mul(&gate, &up, &mut gate_up);

            let mut ffn_out = vec![0.0f32; seq_len * d];
            self.project(&gate_up, &mut ffn_out, seq_len, d, self.d_ff,
                &format!("{}.mlp.down_proj.weight", prefix));

            elementwise::residual_add(&residual2, &ffn_out, &mut hidden);
        }

        // Advance KV cache
        self.kv_cache.advance(seq_len);

        // Final norm
        if let Some(gamma) = self.model.tensor_f32("model.norm.weight") {
            norm::rms_norm(&mut hidden, gamma, d, self.model.header.norm_eps);
        }

        // LM head: last token only → logits [vocab_size]
        let last_hidden = &hidden[(seq_len - 1) * d..seq_len * d];
        let mut logits = vec![0.0f32; self.vocab_size];

        if let Some(lm_w) = self.model.tensor_f32("lm_head.weight") {
            // lm_head.weight is [vocab_size, d_model], compute last_hidden @ W^T
            for v in 0..self.vocab_size {
                let mut acc = 0.0f32;
                for j in 0..d {
                    acc += last_hidden[j] * lm_w[v * d + j];
                }
                logits[v] = acc;
            }
        }

        logits
    }

    /// Autoregressive text generation.
    pub fn generate(&mut self, input_ids: &[u32], max_new_tokens: usize) -> Vec<u32> {
        self.kv_cache.clear();

        let mut output = input_ids.to_vec();

        // Prefill: process all input tokens at once
        let logits = self.forward(input_ids);
        let next_token = argmax(&logits);
        output.push(next_token);

        // Decode: one token at a time
        for _ in 1..max_new_tokens {
            let logits = self.forward(&[*output.last().unwrap()]);
            let next_token = argmax(&logits);
            output.push(next_token);
        }

        output
    }

    /// Reset the session (clear KV cache).
    pub fn reset(&mut self) {
        self.kv_cache.clear();
    }

    /// Model info string.
    pub fn info(&self) -> String {
        format!(
            "{}  d={} heads={} kv_heads={} layers={} ff={} vocab={} max_seq={}",
            self.model.header.model_type,
            self.d_model,
            self.n_heads,
            self.n_kv_heads,
            self.plan.n_layers,
            self.d_ff,
            self.vocab_size,
            self.max_seq_len,
        )
    }

    // Internal: project input through a weight matrix.
    // Supports f32 and ternary weights.
    fn project(&self, input: &[f32], output: &mut [f32], m: usize, n: usize, k: usize, weight_name: &str) {
        let entry = self.model.header.tensors.get(weight_name);

        match entry.map(|e| e.dtype.as_str()) {
            Some("f32") => {
                if let Some(w) = self.model.tensor_f32(weight_name) {
                    // W is [n, k] (row-major), compute input[m,k] @ W^T → output[m,n]
                    for i in 0..m {
                        for j in 0..n {
                            let mut acc = 0.0f32;
                            for p in 0..k {
                                acc += input[i * k + p] * w[j * k + p];
                            }
                            output[i * n + j] = acc;
                        }
                    }
                }
            }
            Some("ternary") => {
                if let (Some(data), Some(scales)) = (
                    self.model.tensor_data(weight_name),
                    self.model.tensor_scales(weight_name),
                ) {
                    matmul::matmul_ternary(data, scales, input, output, n, m, k);
                }
            }
            Some("quaternary") => {
                if let (Some(data), Some(scales)) = (
                    self.model.tensor_data(weight_name),
                    self.model.tensor_scales(weight_name),
                ) {
                    matmul::matmul_quaternary(data, scales, input, output, n, m, k);
                }
            }
            _ => {
                // Weight not found or unsupported dtype — output stays zero
            }
        }
    }
}

/// Argmax over a slice.
fn argmax(data: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in data.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::KorefBuilder;

    fn build_tiny_model() -> KorefModel {
        let vocab = 8;
        let d = 4;
        let n_heads = 2;
        let n_kv_heads = 2;
        let n_layers = 1;
        let d_ff = 8;
        let _head_dim = d / n_heads;

        let mut builder = KorefBuilder::new("test", vocab, d, n_heads, n_kv_heads, n_layers, d_ff, 32, 1e-5, 10000.0);

        // Embedding
        let embed: Vec<f32> = (0..vocab * d).map(|i| (i as f32 * 0.01) - 0.1).collect();
        builder.add_f32("model.embed_tokens.weight", &[vocab, d], &embed);

        // Layer 0
        let prefix = "model.layers.0";

        // Norms (gamma = 1)
        let ones = vec![1.0f32; d];
        builder.add_f32(&format!("{}.input_layernorm.weight", prefix), &[d], &ones);
        builder.add_f32(&format!("{}.post_attention_layernorm.weight", prefix), &[d], &ones);

        // Attention projections (small identity-like)
        let q_w: Vec<f32> = (0..d * d).map(|i| if i / d == i % d { 1.0 } else { 0.0 }).collect();
        builder.add_f32(&format!("{}.self_attn.q_proj.weight", prefix), &[d, d], &q_w);
        builder.add_f32(&format!("{}.self_attn.k_proj.weight", prefix), &[d, d], &q_w);
        builder.add_f32(&format!("{}.self_attn.v_proj.weight", prefix), &[d, d], &q_w);
        builder.add_f32(&format!("{}.self_attn.o_proj.weight", prefix), &[d, d], &q_w);

        // FFN
        let gate_w: Vec<f32> = (0..d_ff * d).map(|i| (i as f32 * 0.01) - 0.05).collect();
        let up_w: Vec<f32> = (0..d_ff * d).map(|i| (i as f32 * 0.01) + 0.01).collect();
        let down_w: Vec<f32> = (0..d * d_ff).map(|i| (i as f32 * 0.01) - 0.02).collect();
        builder.add_f32(&format!("{}.mlp.gate_proj.weight", prefix), &[d_ff, d], &gate_w);
        builder.add_f32(&format!("{}.mlp.up_proj.weight", prefix), &[d_ff, d], &up_w);
        builder.add_f32(&format!("{}.mlp.down_proj.weight", prefix), &[d, d_ff], &down_w);

        // Final norm + LM head
        builder.add_f32("model.norm.weight", &[d], &ones);
        let lm_w: Vec<f32> = (0..vocab * d).map(|i| (i as f32 * 0.02) - 0.1).collect();
        builder.add_f32("lm_head.weight", &[vocab, d], &lm_w);

        builder.build()
    }

    #[test]
    fn test_session_creation() {
        let model = build_tiny_model();
        let session = Session::new(model);
        assert!(session.info().contains("test"));
    }

    #[test]
    fn test_forward_produces_logits() {
        let model = build_tiny_model();
        let mut session = Session::new(model);
        let logits = session.forward(&[0, 1, 2]);
        assert_eq!(logits.len(), 8); // vocab_size
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_generate() {
        let model = build_tiny_model();
        let mut session = Session::new(model);
        let output = session.generate(&[0, 1], 3);
        assert_eq!(output.len(), 5); // 2 input + 3 generated
    }

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[-1.0, -2.0, -0.5]), 2);
    }
}
