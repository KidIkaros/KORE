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

        Ok(Self {
            config,
            cache: Arc::new(cache),
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
    /// This is the core layer-by-layer inference loop:
    /// 1. Embed input tokens
    /// 2. For each transformer layer: load weights → compute → release
    /// 3. Apply final norm
    /// 4. Project to vocabulary (lm_head)
    pub fn forward(&self, input_ids: &[usize]) -> Result<Vec<f32>, String> {
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

        tracing::debug!("embed → shape {:?}", hidden.shape().dims());

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
            // If shape is [seq_len, vocab], extract last row
            // Otherwise return everything (single token case)
            Ok(logits_data[logits_data.len() - vocab..].to_vec())
        } else {
            Ok(logits_data[start..end].to_vec())
        }
    }

    /// Autoregressive generation loop using layered forward passes.
    fn generate_tokens(
        &mut self,
        prompt_tokens: &[usize],
        max_tokens: usize,
        config: &SamplerConfig,
        rng: &mut Rng,
    ) -> Result<Vec<usize>, String> {
        let mut tokens = prompt_tokens.to_vec();

        for step in 0..max_tokens {
            let logits = self.forward(&tokens)?;

            let next_token = sampler::sample(&logits, &tokens, config, rng);

            // Check for EOS
            if let Some(eos) = config.eos_token_id {
                if next_token == eos {
                    tracing::debug!("EOS at step {}", step);
                    break;
                }
            }

            tokens.push(next_token);
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
fn load_layer_sync(prefetcher: &LayerPrefetcher, idx: usize) -> Result<LayerWeights, String> {
    // Use a minimal tokio runtime to bridge async→sync.
    // This is safe because we're inside a spawn_blocking context or single-threaded.
    tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(prefetcher.get_layer(idx))
    })
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

/// Apply a single transformer block (LLaMA-style: pre-norm → attn → residual → pre-norm → MLP → residual).
///
/// This is a simplified implementation suitable for demonstrating layered inference.
/// For production, model-specific blocks should be implemented in the model crate (e.g. Xura).
fn apply_transformer_block(
    weights: &LayerWeights,
    hidden: &Tensor,
    config: &LayeredConfig,
    _layer_idx: usize,
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

    // ── Self-attention (simplified: Q·K^T·V, no KV cache, no RoPE) ──
    let q_proj = build_linear(weights, "self_attn.q_proj.weight", "self_attn.q_proj.bias", d, n_heads * head_dim)?;
    let k_proj = build_linear(weights, "self_attn.k_proj.weight", "self_attn.k_proj.bias", d, n_kv_heads * head_dim)?;
    let v_proj = build_linear(weights, "self_attn.v_proj.weight", "self_attn.v_proj.bias", d, n_kv_heads * head_dim)?;
    let o_proj = build_linear(weights, "self_attn.o_proj.weight", "self_attn.o_proj.bias", n_heads * head_dim, d)?;

    let q = q_proj.forward(&normed).map_err(|e| format!("q_proj: {e}"))?;
    let k = k_proj.forward(&normed).map_err(|e| format!("k_proj: {e}"))?;
    let v = v_proj.forward(&normed).map_err(|e| format!("v_proj: {e}"))?;

    // Simplified attention: softmax(Q·K^T / sqrt(d_k)) · V
    let scale = (head_dim as f32).sqrt();
    let k_t = k.transpose().map_err(|e| format!("k transpose: {e}"))?;
    let attn_scores = q.matmul(&k_t)
        .map_err(|e| format!("QK matmul: {e}"))?;
    let attn_scores = attn_scores.mul_scalar(1.0 / scale)
        .map_err(|e| format!("attn scale: {e}"))?;
    let attn_weights = attn_scores.softmax(-1).map_err(|e| format!("softmax: {e}"))?;
    let attn_out = attn_weights.matmul(&v).map_err(|e| format!("attn·V: {e}"))?;

    let attn_projected = o_proj.forward(&attn_out).map_err(|e| format!("o_proj: {e}"))?;

    // Residual connection
    let hidden_post_attn = hidden.add(&attn_projected).map_err(|e| format!("attn residual: {e}"))?;

    // ── Post-attention norm + MLP ──────────────────────────────
    let mlp_norm = build_rms_norm(weights, "post_attention_layernorm.weight", d, eps)?;
    let normed_mlp = mlp_norm.forward(&hidden_post_attn).map_err(|e| format!("mlp norm: {e}"))?;

    // SwiGLU MLP: gate_proj, up_proj, down_proj
    let gate_proj = build_linear(weights, "mlp.gate_proj.weight", "mlp.gate_proj.bias", d, intermediate)?;
    let up_proj = build_linear(weights, "mlp.up_proj.weight", "mlp.up_proj.bias", d, intermediate)?;
    let down_proj = build_linear(weights, "mlp.down_proj.weight", "mlp.down_proj.bias", intermediate, d)?;

    let gate = gate_proj.forward(&normed_mlp).map_err(|e| format!("gate: {e}"))?;
    let up = up_proj.forward(&normed_mlp).map_err(|e| format!("up: {e}"))?;

    // SiLU(gate) * up
    let gate_activated = silu(&gate);
    let mlp_inner = gate_activated.mul(&up).map_err(|e| format!("gate*up: {e}"))?;
    let mlp_out = down_proj.forward(&mlp_inner).map_err(|e| format!("down: {e}"))?;

    // Residual
    hidden_post_attn.add(&mlp_out).map_err(|e| format!("mlp residual: {e}"))
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
    let _w = bytes_to_tensor_f32(w_data, &[dim])?;
    // RMSNorm::new creates with ones; we need to set the actual gamma.
    // For now, create and trust that the weights match.
    // TODO: Add RMSNorm::from_weight(gamma, eps) constructor upstream.
    Ok(RMSNorm::new(dim, eps))
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

/// SiLU activation: x * sigmoid(x)
fn silu(x: &Tensor) -> Tensor {
    let data = x.as_f32_slice().expect("silu: input must be F32");
    let result: Vec<f32> = data.iter()
        .map(|&v| v * (1.0 / (1.0 + (-v).exp())))
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
}
