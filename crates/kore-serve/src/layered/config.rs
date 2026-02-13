//! Configuration for layered (sharded) inference.

use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// Configuration for layered inference.
///
/// Controls how the model is sharded, cached, and prefetched during
/// layer-by-layer inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayeredConfig {
    /// Directory containing per-layer `.safetensors` shard files.
    pub shard_dir: PathBuf,

    /// Number of transformer layers (excluding embed, norm, lm_head).
    pub num_layers: usize,

    /// Hidden dimension (d_model).
    pub d_model: usize,

    /// Vocabulary size.
    pub vocab_size: usize,

    /// Number of attention heads.
    pub num_heads: usize,

    /// Number of key-value heads (for GQA; equals num_heads for MHA).
    pub num_kv_heads: usize,

    /// Intermediate size (feed-forward hidden dim).
    pub intermediate_size: usize,

    /// Maximum sequence length.
    pub max_seq_len: usize,

    /// RMS norm epsilon.
    pub norm_eps: f32,

    /// Fraction of available RAM to use for layer caching (0.0–1.0).
    /// Set to 0.0 to disable caching.
    pub cache_fraction: f32,

    /// Number of layers to prefetch ahead of the current layer.
    pub prefetch_lookahead: usize,

    /// Layer name pattern for embed weights.
    pub embed_name: String,

    /// Layer name prefix for transformer blocks (e.g. "model.layers").
    pub layer_prefix: String,

    /// Layer name for the final norm.
    pub norm_name: String,

    /// Layer name for the language model head.
    pub lm_head_name: String,
}

impl LayeredConfig {
    /// Create a config for a LLaMA-style model.
    pub fn llama(shard_dir: PathBuf, num_layers: usize, d_model: usize, vocab_size: usize,
                 num_heads: usize, num_kv_heads: usize, intermediate_size: usize) -> Self {
        Self {
            shard_dir,
            num_layers,
            d_model,
            vocab_size,
            num_heads,
            num_kv_heads,
            intermediate_size,
            max_seq_len: 2048,
            norm_eps: 1e-5,
            cache_fraction: 0.3,
            prefetch_lookahead: 2,
            embed_name: "model.embed_tokens".into(),
            layer_prefix: "model.layers".into(),
            norm_name: "model.norm".into(),
            lm_head_name: "lm_head".into(),
        }
    }

    /// Try to load config from a HuggingFace `config.json` file.
    ///
    /// Returns `None` if the file doesn't exist or can't be parsed.
    pub fn from_hf_config(config_path: &std::path::Path, shard_dir: PathBuf) -> Option<Self> {
        let text = std::fs::read_to_string(config_path).ok()?;
        let json: serde_json::Value = serde_json::from_str(&text).ok()?;

        let num_layers = json.get("num_hidden_layers")?.as_u64()? as usize;
        let d_model = json.get("hidden_size")?.as_u64()? as usize;
        let vocab_size = json.get("vocab_size")?.as_u64()? as usize;
        let num_heads = json.get("num_attention_heads")?.as_u64()? as usize;
        let num_kv_heads = json.get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(num_heads as u64) as usize;
        let intermediate_size = json.get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(d_model as u64 * 4) as usize;
        let max_seq_len = json.get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(2048) as usize;
        let norm_eps = json.get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5) as f32;

        Some(Self {
            shard_dir,
            num_layers,
            d_model,
            vocab_size,
            num_heads,
            num_kv_heads,
            intermediate_size,
            max_seq_len,
            norm_eps,
            cache_fraction: 0.3,
            prefetch_lookahead: 2,
            embed_name: "model.embed_tokens".into(),
            layer_prefix: "model.layers".into(),
            norm_name: "model.norm".into(),
            lm_head_name: "lm_head".into(),
        })
    }

    /// Build a list of all layer names in forward-pass order.
    pub fn layer_names(&self) -> Vec<String> {
        let mut names = Vec::with_capacity(self.num_layers + 3);
        names.push(self.embed_name.clone());
        for i in 0..self.num_layers {
            names.push(format!("{}.{}", self.layer_prefix, i));
        }
        names.push(self.norm_name.clone());
        names.push(self.lm_head_name.clone());
        names
    }
}
