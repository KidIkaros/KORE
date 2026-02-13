//! Layered inference engine — layer-by-layer forward pass for large models.
//!
//! This is the main orchestrator for sharded inference. It loads one transformer
//! layer at a time, runs the forward pass, and frees memory before moving on.
//! Combined with async prefetching and LRU caching, this enables running models
//! far larger than available VRAM.

use std::sync::Arc;

use kore_core::{DType, Tensor};
use kore_core::storage::Storage;
use kore_nn::embedding::Embedding;
use kore_nn::linear::Linear;
use kore_nn::module::Module;
use kore_nn::rms_norm::RMSNorm;
use kore_nn::sampler::{self, Rng, SamplerConfig};

use crate::state::InferenceModel;
use super::cache::{LayerCache, LayerWeights};
use super::config::LayeredConfig;
use super::prefetcher::LayerPrefetcher;

/// KV cache for autoregressive generation.
///
/// Uses pre-allocated buffers sized to `max_seq_len` so that appending new
/// key/value data on each decode step is a `memcpy` into the existing buffer
/// rather than a fresh allocation + copy via `Tensor::cat`.
#[derive(Debug, Clone)]
pub struct KvCache {
    layers: Vec<KvCacheLayer>,
    max_seq_len: usize,
}

/// Per-layer pre-allocated key/value buffers.
#[derive(Debug, Clone)]
struct KvCacheLayer {
    k_buf: Vec<f32>,
    v_buf: Vec<f32>,
    /// Number of sequence positions currently filled.
    len: usize,
    /// Dimension per position (`n_kv_heads * head_dim`). Set on first update.
    dim: usize,
}

impl KvCacheLayer {
    fn new() -> Self {
        Self { k_buf: Vec::new(), v_buf: Vec::new(), len: 0, dim: 0 }
    }

    /// Ensure the buffer is allocated for the given max_seq_len and dim.
    /// Only allocates on the first call; subsequent calls are no-ops.
    fn ensure_capacity(&mut self, max_seq_len: usize, dim: usize) {
        if self.dim == 0 {
            self.dim = dim;
            let capacity = max_seq_len * dim;
            self.k_buf.resize(capacity, 0.0);
            self.v_buf.resize(capacity, 0.0);
        }
    }

    /// Append new_tokens positions of key/value data into the pre-allocated buffer.
    fn append(&mut self, k_data: &[f32], v_data: &[f32], new_tokens: usize) -> Result<(), String> {
        let n = new_tokens * self.dim;
        let offset = self.len * self.dim;
        let end = offset + n;
        if end > self.k_buf.len() {
            return Err(format!(
                "KV cache overflow: need {} but capacity is {} (max_seq_len exceeded)",
                end, self.k_buf.len()
            ));
        }
        self.k_buf[offset..end].copy_from_slice(&k_data[..n]);
        self.v_buf[offset..end].copy_from_slice(&v_data[..n]);
        self.len += new_tokens;
        Ok(())
    }

    /// Return the full cached K tensor `[len, dim]`.
    fn k_tensor(&self) -> Tensor {
        Tensor::from_f32(&self.k_buf[..self.len * self.dim], &[self.len, self.dim])
    }

    /// Return the full cached V tensor `[len, dim]`.
    fn v_tensor(&self) -> Tensor {
        Tensor::from_f32(&self.v_buf[..self.len * self.dim], &[self.len, self.dim])
    }

    fn clear(&mut self) {
        self.len = 0;
        // Keep the buffer allocated for reuse — just reset the length.
    }
}

impl KvCache {
    /// Create a KV cache for `num_layers` layers with room for `max_seq_len` positions.
    ///
    /// Actual buffer allocation is deferred until the first `update` call
    /// (when the KV dimension is known).
    pub fn new(num_layers: usize, max_seq_len: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| KvCacheLayer::new()).collect(),
            max_seq_len,
        }
    }

    /// Append new key/value tensors for a given layer.
    ///
    /// On the first call for a layer the buffer is pre-allocated to hold
    /// `max_seq_len` positions. Subsequent calls copy into the existing
    /// buffer without allocating.
    ///
    /// Returns tensors covering all cached positions for this layer.
    pub fn update(&mut self, layer: usize, new_k: Tensor, new_v: Tensor) -> Result<(Tensor, Tensor), String> {
        let dims = new_k.shape().dims();
        if dims.len() != 2 {
            return Err(format!(
                "KV cache: expected 2D tensor [seq_len, kv_dim], got shape {:?}",
                dims
            ));
        }
        let new_tokens = dims[0];
        let dim = dims[1];

        let entry = &mut self.layers[layer];
        entry.ensure_capacity(self.max_seq_len, dim);

        let k_data = new_k.as_f32_slice().ok_or("KV cache: K must be F32")?;
        let v_data = new_v.as_f32_slice().ok_or("KV cache: V must be F32")?;
        entry.append(k_data, v_data, new_tokens)?;

        Ok((entry.k_tensor(), entry.v_tensor()))
    }

    /// Clear all layers (call between independent prompts).
    /// Buffers remain allocated for reuse.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    /// Number of cached positions (from the first populated layer).
    pub fn seq_len(&self) -> usize {
        self.layers.iter().map(|l| l.len).max().unwrap_or(0)
    }
}

/// Layered inference engine for running large models on limited VRAM.
///
/// Instead of loading the entire model into memory, this engine loads one
/// transformer layer at a time, computes the forward pass for that layer,
/// and releases the weights before loading the next layer.
///
/// # Example
///
/// ```rust,no_run
/// use kore_serve::layered::{LayeredEngine, LayeredConfig};
/// use std::path::PathBuf;
///
/// let config = LayeredConfig::llama(
///     PathBuf::from("./model_shards"),
///     32, 4096, 32000, 32, 32, 11008,
/// );
/// let engine = LayeredEngine::new(config).expect("failed to create engine");
/// ```
#[derive(Debug)]
pub struct LayeredEngine {
    config: LayeredConfig,
    cache: Arc<LayerCache>,
    /// Persistent KV cache used across generation steps.
    kv_cache: KvCache,
    /// Pre-computed RoPE inverse frequencies: `1 / (base^(2i / head_dim))`
    /// for `i` in `0..head_dim/2`. Computed once; reused every token.
    rope_inv_freq: Vec<f32>,
}

impl LayeredEngine {
    /// Create a new layered inference engine.
    ///
    /// Initializes the LRU layer cache based on `config.cache_fraction`.
    pub fn new(config: LayeredConfig) -> Result<Self, String> {
        if !config.shard_dir.exists() {
            return Err(format!(
                "shard directory does not exist: {}",
                config.shard_dir.display()
            ));
        }

        let cache = if config.cache_fraction > 0.0 {
            LayerCache::adaptive(config.cache_fraction)
        } else {
            LayerCache::new(0)
        };

        tracing::info!(
            "LayeredEngine: {} layers, d_model={}, vocab={}, shards={}",
            config.num_layers,
            config.d_model,
            config.vocab_size,
            config.shard_dir.display(),
        );

        let kv_cache = KvCache::new(config.num_layers, config.max_seq_len);

        let head_dim = config.d_model / config.num_heads;
        let rope_inv_freq = precompute_rope_inv_freq(head_dim, config.rope_theta);

        Ok(Self {
            config,
            cache: Arc::new(cache),
            kv_cache,
            rope_inv_freq,
        })
    }

    /// Get the engine configuration.
    pub fn config(&self) -> &LayeredConfig {
        &self.config
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> super::cache::CacheStats {
        self.cache.stats()
    }

    /// Run a full forward pass through the model, producing logits.
    ///
    /// Uses the internal KV cache — call [`Self::clear_kv_cache`] between
    /// independent prompts. For the first call the full sequence is processed
    /// (prefill); subsequent calls only process new tokens.
    ///
    /// This is the core layer-by-layer inference loop:
    /// 1. Embed input tokens
    /// 2. For each transformer layer: load weights → compute → release
    /// 3. Apply final norm
    /// 4. Project to vocabulary (lm_head)
    pub fn forward(&mut self, input_ids: &[usize]) -> Result<Vec<f32>, String> {
        let start_pos = self.kv_cache.seq_len();
        self.forward_inner(input_ids, start_pos)
    }

    /// Clear the KV cache (call between independent prompts).
    pub fn clear_kv_cache(&mut self) {
        self.kv_cache.clear();
    }

    fn forward_inner(&mut self, input_ids: &[usize], start_pos: usize) -> Result<Vec<f32>, String> {
        if input_ids.is_empty() {
            return Err("input_ids must not be empty".to_string());
        }

        let layer_names = self.config.layer_names();

        let prefetcher = LayerPrefetcher::new(
            self.config.shard_dir.clone(),
            layer_names.clone(),
            self.config.prefetch_lookahead,
            self.cache.clone(),
            false,
        );
        prefetcher.start();

        // ── 1. Embedding ──────────────────────────────────────────
        let embed_weights = load_layer_sync(&prefetcher, 0)?;
        let hidden = apply_embedding(&embed_weights, input_ids, self.config.d_model)?;
        drop(embed_weights);
        prefetcher.release(0);

        tracing::debug!("embed → shape {:?}, start_pos={}", hidden.shape().dims(), start_pos);

        // ── 2. Transformer layers ─────────────────────────────────
        let mut hidden = hidden;
        for i in 0..self.config.num_layers {
            let layer_idx = i + 1; // offset by 1 (embed is index 0)
            let layer_weights = load_layer_sync(&prefetcher, layer_idx)?;

            hidden = apply_transformer_block(
                &layer_weights,
                &hidden,
                &self.config,
                i,
                start_pos,
                &mut self.kv_cache,
                &self.rope_inv_freq,
            )?;

            drop(layer_weights);
            prefetcher.release(layer_idx);

            tracing::trace!("layer {}/{} done", i + 1, self.config.num_layers);
        }

        // ── 3. Final norm ─────────────────────────────────────────
        let norm_idx = self.config.num_layers + 1;
        let norm_weights = load_layer_sync(&prefetcher, norm_idx)?;
        hidden = apply_rms_norm_from_weights(&norm_weights, &hidden, self.config.norm_eps)?;
        drop(norm_weights);
        prefetcher.release(norm_idx);

        // ── 4. LM head ───────────────────────────────────────────
        let head_idx = self.config.num_layers + 2;
        let head_weights = load_layer_sync(&prefetcher, head_idx)?;
        let logits = apply_lm_head(&head_weights, &hidden)?;
        drop(head_weights);
        prefetcher.release(head_idx);

        // Return logits for the last token position
        let logits_data = logits.as_f32_slice()
            .ok_or("failed to read logits as f32")?;

        let seq_len = input_ids.len();
        let vocab = self.config.vocab_size;

        if logits_data.len() < vocab {
            return Err(format!(
                "logits too small: expected at least {} (vocab), got {}",
                vocab,
                logits_data.len()
            ));
        }

        // Last token's logits
        let start = (seq_len - 1) * vocab;
        let end = start + vocab;
        if end > logits_data.len() {
            Ok(logits_data[logits_data.len() - vocab..].to_vec())
        } else {
            Ok(logits_data[start..end].to_vec())
        }
    }

    /// Autoregressive generation loop using the KV cache.
    ///
    /// On the first step the entire prompt is processed (prefill).
    /// Each subsequent step processes only the newly generated token,
    /// reusing cached key/value tensors from previous positions.
    fn generate_tokens(
        &mut self,
        prompt_tokens: &[usize],
        max_tokens: usize,
        config: &SamplerConfig,
        rng: &mut Rng,
    ) -> Result<Vec<usize>, String> {
        self.clear_kv_cache();
        let mut tokens = prompt_tokens.to_vec();

        // Prefill: process full prompt
        let logits = self.forward(&tokens)?;
        let mut next_token = sampler::sample(&logits, &tokens, config, rng);

        for step in 0..max_tokens {
            // Check for EOS
            if let Some(eos) = config.eos_token_id {
                if next_token == eos {
                    tracing::debug!("EOS at step {}", step);
                    break;
                }
            }

            tokens.push(next_token);

            // Decode: only process the new token (KV cache has the rest)
            let logits = self.forward(&[next_token])?;
            next_token = sampler::sample(&logits, &tokens, config, rng);
        }

        Ok(tokens)
    }
}

impl InferenceModel for LayeredEngine {
    fn generate_with_config(
        &mut self,
        prompt_tokens: &[usize],
        max_tokens: usize,
        config: &SamplerConfig,
        rng: &mut Rng,
    ) -> Result<Vec<usize>, String> {
        self.generate_tokens(prompt_tokens, max_tokens, config, rng)
    }
}

// ============================================================================
// Helper functions — weight deserialization and layer application
// ============================================================================

/// Synchronously load a layer from the prefetcher (blocks on the async channel).
///
/// If a Tokio runtime is available, uses `block_in_place` to avoid blocking
/// the async executor. Otherwise, creates a temporary runtime for the load.
fn load_layer_sync(prefetcher: &LayerPrefetcher, idx: usize) -> Result<LayerWeights, String> {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            tokio::task::block_in_place(|| handle.block_on(prefetcher.get_layer(idx)))
        }
        Err(_) => {
            // No active runtime — spin up a lightweight one for this call.
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| format!("failed to create tokio runtime: {e}"))?;
            rt.block_on(prefetcher.get_layer(idx))
        }
    }
}

/// Convert raw weight bytes (F32 little-endian) into a kore Tensor.
fn bytes_to_tensor_f32(data: &[u8], shape: &[usize]) -> Result<Tensor, String> {
    let expected_numel: usize = shape.iter().product();
    let expected_bytes = expected_numel * 4; // f32 = 4 bytes

    if data.len() != expected_bytes {
        return Err(format!(
            "weight size mismatch: expected {} bytes ({} f32s for shape {:?}), got {}",
            expected_bytes, expected_numel, shape, data.len()
        ));
    }

    let storage = Storage::from_bytes(DType::F32, expected_numel, data.to_vec())
        .map_err(|e| format!("storage error: {e}"))?;
    Ok(Tensor::from_storage(storage, shape))
}

/// Find a weight tensor by suffix in layer weights (e.g. "weight", "self_attn.q_proj.weight").
fn find_weight<'a>(weights: &'a LayerWeights, suffix: &str) -> Option<&'a Vec<u8>> {
    weights.iter()
        .find(|(name, _)| name.ends_with(suffix))
        .map(|(_, data)| data)
}

/// Apply embedding lookup from raw weight bytes.
fn apply_embedding(
    weights: &LayerWeights,
    input_ids: &[usize],
    d_model: usize,
) -> Result<Tensor, String> {
    let w_data = find_weight(weights, "weight")
        .ok_or("embedding: missing 'weight' tensor")?;

    let num_embeddings = w_data.len() / (d_model * 4); // f32 = 4 bytes
    let w_tensor = bytes_to_tensor_f32(w_data, &[num_embeddings, d_model])?;
    let embed = Embedding::from_weight(w_tensor);
    Ok(embed.lookup(input_ids))
}

/// Apply a single transformer block (LLaMA-style):
/// pre-norm → multi-head GQA attention with RoPE → residual → pre-norm → SwiGLU MLP → residual.
fn apply_transformer_block(
    weights: &LayerWeights,
    hidden: &Tensor,
    config: &LayeredConfig,
    layer_idx: usize,
    start_pos: usize,
    kv_cache: &mut KvCache,
    rope_inv_freq: &[f32],
) -> Result<Tensor, String> {
    let d = config.d_model;
    let eps = config.norm_eps;
    let n_heads = config.num_heads;
    let n_kv_heads = config.num_kv_heads;
    let head_dim = d / n_heads;
    let intermediate = config.intermediate_size;

    // ── Pre-attention norm ─────────────────────────────────────
    let attn_norm = build_rms_norm(weights, "input_layernorm.weight", d, eps)?;
    let normed = attn_norm.forward(hidden).map_err(|e| format!("attn norm: {e}"))?;

    // ── Self-attention with multi-head GQA, RoPE, and KV cache ──
    let q_proj = build_linear(weights, "self_attn.q_proj.weight", "self_attn.q_proj.bias", d, n_heads * head_dim)?;
    let k_proj = build_linear(weights, "self_attn.k_proj.weight", "self_attn.k_proj.bias", d, n_kv_heads * head_dim)?;
    let v_proj = build_linear(weights, "self_attn.v_proj.weight", "self_attn.v_proj.bias", d, n_kv_heads * head_dim)?;
    let o_proj = build_linear(weights, "self_attn.o_proj.weight", "self_attn.o_proj.bias", n_heads * head_dim, d)?;

    // Project: [seq_len, d] → Q:[seq_len, n_heads*hd], K:[seq_len, n_kv*hd], V:[seq_len, n_kv*hd]
    let q_full = q_proj.forward(&normed).map_err(|e| format!("q_proj: {e}"))?;
    let k_new = k_proj.forward(&normed).map_err(|e| format!("k_proj: {e}"))?;
    let v_new = v_proj.forward(&normed).map_err(|e| format!("v_proj: {e}"))?;

    let seq_len = hidden.shape().dims()[0];

    // Split into per-head tensors: each [seq_len, head_dim]
    let q_heads = split_heads(&q_full, n_heads, head_dim, seq_len);
    let k_new_heads = split_heads(&k_new, n_kv_heads, head_dim, seq_len);
    let v_new_heads = split_heads(&v_new, n_kv_heads, head_dim, seq_len);

    // Apply RoPE to Q and K heads (using pre-computed inverse frequencies)
    let q_heads: Vec<Tensor> = q_heads.into_iter()
        .map(|h| apply_rope(&h, start_pos, head_dim, rope_inv_freq))
        .collect();
    let k_new_heads: Vec<Tensor> = k_new_heads.into_iter()
        .map(|h| apply_rope(&h, start_pos, head_dim, rope_inv_freq))
        .collect();

    // Reassemble K for KV cache update: [seq_len, n_kv_heads * head_dim]
    let k_new_assembled = concat_heads(&k_new_heads, seq_len, n_kv_heads, head_dim);
    let v_new_assembled = concat_heads(&v_new_heads, seq_len, n_kv_heads, head_dim);

    // Update KV cache: returns [full_seq_len, n_kv_heads * head_dim]
    let (k_full, v_full) = kv_cache.update(layer_idx, k_new_assembled, v_new_assembled)?;
    let full_seq_len = k_full.shape().dims()[0];

    // Re-split full K and V into per-head tensors
    let k_heads = split_heads(&k_full, n_kv_heads, head_dim, full_seq_len);
    let v_heads = split_heads(&v_full, n_kv_heads, head_dim, full_seq_len);

    // Compute attention per head with GQA
    let heads_per_kv = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_outputs = Vec::with_capacity(n_heads);

    for (h, q_h) in q_heads.iter().enumerate() {
        let kv_idx = h / heads_per_kv;
        let k_h = &k_heads[kv_idx];    // [full_seq_len, head_dim]
        let v_h = &v_heads[kv_idx];    // [full_seq_len, head_dim]

        // scores = Q · K^T / sqrt(head_dim)  →  [seq_len, full_seq_len]
        let k_t = k_h.transpose().map_err(|e| format!("k transpose h{h}: {e}"))?;
        let scores = q_h.matmul(&k_t).map_err(|e| format!("QK h{h}: {e}"))?;
        let scores = scores.mul_scalar(scale).map_err(|e| format!("scale h{h}: {e}"))?;

        // Causal mask (only needed during prefill when seq_len > 1)
        let scores = if seq_len > 1 {
            apply_causal_mask(&scores, start_pos)?
        } else {
            scores
        };

        let weights = scores.softmax(-1).map_err(|e| format!("softmax h{h}: {e}"))?;
        let out = weights.matmul(v_h).map_err(|e| format!("attn·V h{h}: {e}"))?;
        attn_outputs.push(out);
    }

    // Concat heads → [seq_len, n_heads * head_dim]
    let attn_concat = concat_heads(&attn_outputs, seq_len, n_heads, head_dim);

    // Output projection
    let attn_projected = o_proj.forward(&attn_concat).map_err(|e| format!("o_proj: {e}"))?;

    // Residual connection
    let hidden_post_attn = hidden.add(&attn_projected).map_err(|e| format!("attn residual: {e}"))?;

    // ── Post-attention norm + SwiGLU MLP ──────────────────────
    let mlp_norm = build_rms_norm(weights, "post_attention_layernorm.weight", d, eps)?;
    let normed_mlp = mlp_norm.forward(&hidden_post_attn).map_err(|e| format!("mlp norm: {e}"))?;

    let gate_proj = build_linear(weights, "mlp.gate_proj.weight", "mlp.gate_proj.bias", d, intermediate)?;
    let up_proj = build_linear(weights, "mlp.up_proj.weight", "mlp.up_proj.bias", d, intermediate)?;
    let down_proj = build_linear(weights, "mlp.down_proj.weight", "mlp.down_proj.bias", intermediate, d)?;

    let gate = gate_proj.forward(&normed_mlp).map_err(|e| format!("gate: {e}"))?;
    let up = up_proj.forward(&normed_mlp).map_err(|e| format!("up: {e}"))?;

    let gate_activated = silu(&gate);
    let mlp_inner = gate_activated.mul(&up).map_err(|e| format!("gate*up: {e}"))?;
    let mlp_out = down_proj.forward(&mlp_inner).map_err(|e| format!("down: {e}"))?;

    hidden_post_attn.add(&mlp_out).map_err(|e| format!("mlp residual: {e}"))
}

// ============================================================================
// Multi-head attention helpers
// ============================================================================

/// Split a projected tensor `[seq_len, n_heads * head_dim]` into per-head
/// tensors, each of shape `[seq_len, head_dim]`.
fn split_heads(tensor: &Tensor, n_heads: usize, head_dim: usize, seq_len: usize) -> Vec<Tensor> {
    let data = tensor.as_f32_slice().expect("split_heads: F32 required");
    let total_dim = n_heads * head_dim;
    (0..n_heads).map(|h| {
        let mut head_data = Vec::with_capacity(seq_len * head_dim);
        for pos in 0..seq_len {
            let start = pos * total_dim + h * head_dim;
            head_data.extend_from_slice(&data[start..start + head_dim]);
        }
        Tensor::from_f32(&head_data, &[seq_len, head_dim])
    }).collect()
}

/// Concatenate per-head tensors `[seq_len, head_dim]` back into
/// `[seq_len, n_heads * head_dim]`.
fn concat_heads(heads: &[Tensor], seq_len: usize, n_heads: usize, head_dim: usize) -> Tensor {
    let total_dim = n_heads * head_dim;
    let mut result = vec![0.0f32; seq_len * total_dim];
    for (h, head) in heads.iter().enumerate() {
        let data = head.as_f32_slice().expect("concat_heads: F32 required");
        for pos in 0..seq_len {
            let src = pos * head_dim;
            let dst = pos * total_dim + h * head_dim;
            result[dst..dst + head_dim].copy_from_slice(&data[src..src + head_dim]);
        }
    }
    Tensor::from_f32(&result, &[seq_len, total_dim])
}

/// Pre-compute the RoPE inverse frequency table for a given `head_dim` and `base`.
///
/// Returns `head_dim / 2` frequencies: `1 / (base^(2i / head_dim))`.
/// This only needs to be computed once per engine lifetime.
fn precompute_rope_inv_freq(head_dim: usize, base: f32) -> Vec<f32> {
    (0..head_dim / 2)
        .map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32))
        .collect()
}

/// Apply Rotary Position Embedding (RoPE) to a single head tensor
/// of shape `[seq_len, head_dim]`.
///
/// Uses a pre-computed `inv_freq` table (from [`precompute_rope_inv_freq`])
/// so that no `powf` calls happen on the hot path — only `sin`/`cos` of
/// the position-dependent angles.
fn apply_rope(x: &Tensor, start_pos: usize, head_dim: usize, inv_freq: &[f32]) -> Tensor {
    let data = x.as_f32_slice().expect("apply_rope: F32 required");
    let seq_len = x.shape().dims()[0];
    let mut result = vec![0.0f32; data.len()];

    for pos_idx in 0..seq_len {
        let abs_pos = (start_pos + pos_idx) as f32;
        for (pair, &freq) in inv_freq.iter().enumerate() {
            let angle = abs_pos * freq;
            let cos_val = angle.cos();
            let sin_val = angle.sin();

            let i = pos_idx * head_dim + 2 * pair;
            let x0 = data[i];
            let x1 = data[i + 1];
            result[i] = x0 * cos_val - x1 * sin_val;
            result[i + 1] = x0 * sin_val + x1 * cos_val;
        }
    }

    Tensor::from_f32(&result, x.shape().dims())
}

/// Build a causal attention mask and apply it to `scores`.
///
/// `scores` has shape `[q_len, kv_len]` where `kv_len = start_pos + q_len`.
/// Position `q_i` (absolute position `start_pos + q_i`) may only attend to
/// key positions `≤ start_pos + q_i`, so the mask sets future entries to `-inf`.
fn apply_causal_mask(scores: &Tensor, start_pos: usize) -> Result<Tensor, String> {
    let dims = scores.shape().dims();
    let q_len = dims[0];
    let kv_len = dims[1];
    let data = scores.as_f32_slice().ok_or("causal mask: F32 required")?;
    let mut masked = data.to_vec();

    for qi in 0..q_len {
        let abs_pos = start_pos + qi;
        for ki in 0..kv_len {
            if ki > abs_pos {
                masked[qi * kv_len + ki] = f32::NEG_INFINITY;
            }
        }
    }

    Ok(Tensor::from_f32(&masked, dims))
}

/// Apply RMS norm from raw weight bytes.
fn apply_rms_norm_from_weights(
    weights: &LayerWeights,
    hidden: &Tensor,
    eps: f32,
) -> Result<Tensor, String> {
    let d = hidden.shape().dims().last()
        .copied()
        .ok_or("empty hidden shape")?;
    let norm = build_rms_norm(weights, "weight", d, eps)?;
    norm.forward(hidden).map_err(|e| format!("final norm: {e}"))
}

/// Apply lm_head linear projection from raw weight bytes.
fn apply_lm_head(
    weights: &LayerWeights,
    hidden: &Tensor,
) -> Result<Tensor, String> {
    let w_data = find_weight(weights, "weight")
        .ok_or("lm_head: missing 'weight' tensor")?;

    let d_model = hidden.shape().dims().last()
        .copied()
        .ok_or("empty hidden shape for lm_head")?;
    let vocab_size = w_data.len() / (d_model * 4);

    let w = bytes_to_tensor_f32(w_data, &[vocab_size, d_model])?;
    let bias_data = find_weight(weights, "bias");
    let bias = match bias_data {
        Some(b) => Some(bytes_to_tensor_f32(b, &[vocab_size])?),
        None => None,
    };

    let linear = Linear::from_weight(w, bias);
    linear.forward(hidden).map_err(|e| format!("lm_head forward: {e}"))
}

// ============================================================================
// Builder helpers
// ============================================================================

fn build_rms_norm(
    weights: &LayerWeights,
    suffix: &str,
    dim: usize,
    eps: f32,
) -> Result<RMSNorm, String> {
    let w_data = find_weight(weights, suffix)
        .ok_or_else(|| format!("missing norm weight '{suffix}'"))?;
    let gamma = bytes_to_tensor_f32(w_data, &[dim])?;
    Ok(RMSNorm::from_weight(gamma, eps))
}

fn build_linear(
    weights: &LayerWeights,
    weight_suffix: &str,
    bias_suffix: &str,
    in_features: usize,
    out_features: usize,
) -> Result<Linear, String> {
    let w_data = find_weight(weights, weight_suffix)
        .ok_or_else(|| format!("missing weight '{weight_suffix}'"))?;
    let w = bytes_to_tensor_f32(w_data, &[out_features, in_features])?;

    let bias = find_weight(weights, bias_suffix)
        .and_then(|b| bytes_to_tensor_f32(b, &[out_features]).ok());

    Ok(Linear::from_weight(w, bias))
}

/// SiLU activation: x * sigmoid(x), numerically stable for large |x|.
fn silu(x: &Tensor) -> Tensor {
    let data = x.as_f32_slice().expect("silu: input must be F32");
    let result: Vec<f32> = data.iter()
        .map(|&v| {
            if v >= 0.0 {
                v / (1.0 + (-v).exp())
            } else {
                let exp_v = v.exp();
                (v * exp_v) / (1.0 + exp_v)
            }
        })
        .collect();
    Tensor::from_f32(&result, x.shape().dims())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_engine_rejects_missing_shard_dir() {
        let config = LayeredConfig::llama(
            PathBuf::from("/nonexistent/path"),
            32, 4096, 32000, 32, 32, 11008,
        );
        let result = LayeredEngine::new(config);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("does not exist"));
    }

    #[test]
    fn test_forward_rejects_empty_input() {
        let dir = std::env::temp_dir().join("kore_empty_input_test");
        let _ = std::fs::create_dir_all(&dir);
        let config = LayeredConfig::llama(dir.clone(), 1, 64, 100, 2, 2, 128);
        let mut engine = LayeredEngine::new(config).unwrap();
        let result = engine.forward(&[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must not be empty"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_silu() {
        let x = Tensor::from_f32(&[0.0, 1.0, -1.0], &[3]);
        let y = silu(&x);
        let data = y.as_f32_slice().unwrap();
        // silu(0) = 0, silu(1) ≈ 0.731, silu(-1) ≈ -0.269
        assert!((data[0] - 0.0).abs() < 1e-5);
        assert!((data[1] - 0.7311).abs() < 1e-3);
        assert!((data[2] - (-0.2689)).abs() < 1e-3);
    }

    #[test]
    fn test_bytes_to_tensor() {
        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        let t = bytes_to_tensor_f32(&data, &[2, 2]).unwrap();
        assert_eq!(t.shape().dims(), &[2, 2]);
        let slice = t.as_f32_slice().unwrap();
        assert!((slice[0] - 1.0).abs() < 1e-6);
        assert!((slice[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_bytes_to_tensor_size_mismatch() {
        let data = vec![0u8; 12]; // 3 f32s
        let result = bytes_to_tensor_f32(&data, &[2, 2]); // expects 4 f32s
        assert!(result.is_err());
    }

    #[test]
    fn test_split_concat_heads_roundtrip() {
        // [2 positions, 4 heads * 3 head_dim = 12]
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = Tensor::from_f32(&data, &[2, 12]);

        let heads = split_heads(&t, 4, 3, 2);
        assert_eq!(heads.len(), 4);
        for h in &heads {
            assert_eq!(h.shape().dims(), &[2, 3]);
        }
        // Head 0 should have cols [0..3] from each row
        let h0 = heads[0].as_f32_slice().unwrap();
        assert_eq!(h0, &[0.0, 1.0, 2.0, 12.0, 13.0, 14.0]);

        let reconstructed = concat_heads(&heads, 2, 4, 3);
        assert_eq!(reconstructed.shape().dims(), &[2, 12]);
        let r = reconstructed.as_f32_slice().unwrap();
        assert_eq!(r, &data);
    }

    #[test]
    fn test_rope_identity_at_pos_zero() {
        // At position 0, all angles are 0 → cos=1, sin=0 → no rotation
        let inv_freq = precompute_rope_inv_freq(4, 10000.0);
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let rotated = apply_rope(&x, 0, 4, &inv_freq);
        let data = rotated.as_f32_slice().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);
        assert!((data[2] - 3.0).abs() < 1e-5);
        assert!((data[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_rope_preserves_norm() {
        // RoPE is a rotation, so it should preserve the L2 norm
        let inv_freq = precompute_rope_inv_freq(4, 10000.0);
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let norm_before: f32 = x.as_f32_slice().unwrap().iter().map(|v| v * v).sum::<f32>().sqrt();

        let rotated = apply_rope(&x, 42, 4, &inv_freq);
        let norm_after: f32 = rotated.as_f32_slice().unwrap().iter().map(|v| v * v).sum::<f32>().sqrt();

        assert!((norm_before - norm_after).abs() < 1e-4, "RoPE should preserve norm: {} vs {}", norm_before, norm_after);
    }

    #[test]
    fn test_rope_different_positions_differ() {
        let inv_freq = precompute_rope_inv_freq(4, 10000.0);
        let x = Tensor::from_f32(&[1.0, 0.0, 1.0, 0.0], &[1, 4]);
        let r0 = apply_rope(&x, 0, 4, &inv_freq);
        let r5 = apply_rope(&x, 5, 4, &inv_freq);
        let d0 = r0.as_f32_slice().unwrap();
        let d5 = r5.as_f32_slice().unwrap();
        // They should differ (rotation at different positions)
        assert!((d0[0] - d5[0]).abs() > 1e-4 || (d0[1] - d5[1]).abs() > 1e-4);
    }

    #[test]
    fn test_precompute_rope_inv_freq() {
        let inv_freq = precompute_rope_inv_freq(8, 10000.0);
        assert_eq!(inv_freq.len(), 4); // head_dim/2
        // First freq = 1 / 10000^0 = 1.0
        assert!((inv_freq[0] - 1.0).abs() < 1e-6);
        // Each subsequent freq should be smaller
        for i in 1..inv_freq.len() {
            assert!(inv_freq[i] < inv_freq[i - 1]);
        }
    }

    #[test]
    fn test_causal_mask_blocks_future() {
        // scores: [3, 3] — 3 query positions, 3 key positions, start_pos=0
        let scores = Tensor::from_f32(&[1.0; 9], &[3, 3]);
        let masked = apply_causal_mask(&scores, 0).unwrap();
        let data = masked.as_f32_slice().unwrap();
        // Row 0 (pos 0): can see [0], not [1,2]
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], f32::NEG_INFINITY);
        assert_eq!(data[2], f32::NEG_INFINITY);
        // Row 1 (pos 1): can see [0,1], not [2]
        assert_eq!(data[3], 1.0);
        assert_eq!(data[4], 1.0);
        assert_eq!(data[5], f32::NEG_INFINITY);
        // Row 2 (pos 2): can see [0,1,2]
        assert_eq!(data[6], 1.0);
        assert_eq!(data[7], 1.0);
        assert_eq!(data[8], 1.0);
    }

    #[test]
    fn test_causal_mask_with_start_pos() {
        // 1 query at absolute position 5, 6 keys (positions 0..5)
        let scores = Tensor::from_f32(&[1.0; 6], &[1, 6]);
        let masked = apply_causal_mask(&scores, 5).unwrap();
        let data = masked.as_f32_slice().unwrap();
        // Position 5 can attend to all positions 0..5
        assert!(data.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_kv_cache_update_and_concat() {
        let mut kv = KvCache::new(2, 128);
        assert_eq!(kv.seq_len(), 0);

        // First update: 3 positions
        let k1 = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let v1 = Tensor::from_f32(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[3, 2]);
        let (fk, fv) = kv.update(0, k1, v1).unwrap();
        assert_eq!(fk.shape().dims(), &[3, 2]);
        assert_eq!(fv.shape().dims(), &[3, 2]);
        assert_eq!(kv.seq_len(), 3);

        // Second update: 1 more position
        let k2 = Tensor::from_f32(&[7.0, 8.0], &[1, 2]);
        let v2 = Tensor::from_f32(&[70.0, 80.0], &[1, 2]);
        let (fk, fv) = kv.update(0, k2, v2).unwrap();
        assert_eq!(fk.shape().dims(), &[4, 2]);
        assert_eq!(fv.shape().dims(), &[4, 2]);
        assert_eq!(kv.seq_len(), 4);

        // Verify concatenated data
        let kd = fk.as_f32_slice().unwrap();
        assert_eq!(kd, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // Clear
        kv.clear();
        assert_eq!(kv.seq_len(), 0);
    }

    #[test]
    fn test_gqa_head_mapping() {
        // n_heads=4, n_kv_heads=2 → heads_per_kv=2
        // Q heads 0,1 should map to KV head 0; Q heads 2,3 to KV head 1
        let n_heads = 4;
        let n_kv_heads = 2;
        let heads_per_kv = n_heads / n_kv_heads;
        assert_eq!(0 / heads_per_kv, 0);
        assert_eq!(1 / heads_per_kv, 0);
        assert_eq!(2 / heads_per_kv, 1);
        assert_eq!(3 / heads_per_kv, 1);
    }
}
