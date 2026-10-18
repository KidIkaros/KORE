//! Direct GGUF layer loader for layered inference.
//!
//! Enables running any LLaMA-family GGUF model directly via `LayeredEngine`
//! with zero offline sharding or disk duplication. Tensors are memory-mapped
//! and dequantized on-the-fly (AVX2/SIMD) layer-by-layer.

use std::path::Path;
use std::sync::Arc;

use oxillama_gguf::GgufModel;
use parking_lot::Mutex;

use super::cache::LayerWeights;
use super::config::LayeredConfig;

static GGUF_CACHE: Mutex<Option<(std::path::PathBuf, Arc<GgufModel>)>> = Mutex::new(None);

/// Get or load a cached memory-mapped GGUF model instance.
pub fn get_or_load_gguf(path: &Path) -> Result<Arc<GgufModel>, String> {
    let mut guard = GGUF_CACHE.lock();
    if let Some((ref cached_path, ref model)) = *guard {
        if cached_path == path {
            return Ok(model.clone());
        }
    }

    let model = GgufModel::load(path)
        .map_err(|e| format!("failed to load GGUF model from {}: {e}", path.display()))?;
    let arc_model = Arc::new(model);
    *guard = Some((path.to_path_buf(), arc_model.clone()));
    Ok(arc_model)
}

/// Create a `LayeredConfig` directly from GGUF metadata.
pub fn config_from_gguf(path: &Path) -> Result<LayeredConfig, String> {
    let model = get_or_load_gguf(path)?;
    let meta = &model.file.metadata;

    let num_layers = meta
        .get("llama.block_count")
        .and_then(|v| v.as_u32())
        .ok_or_else(|| "missing 'llama.block_count' in GGUF metadata".to_string())?
        as usize;

    let d_model = meta
        .get("llama.embedding_length")
        .and_then(|v| v.as_u32())
        .ok_or_else(|| "missing 'llama.embedding_length' in GGUF metadata".to_string())?
        as usize;

    let vocab_size = meta
        .get("llama.vocab_size")
        .and_then(|v| v.as_u32())
        .map(|v| v as usize)
        .or_else(|| {
            meta.get("tokenizer.ggml.tokens")
                .and_then(|v| v.as_array().map(|a| a.len()))
        })
        .unwrap_or(32000);

    let num_heads = meta
        .get("llama.attention.head_count")
        .and_then(|v| v.as_u32())
        .ok_or_else(|| "missing 'llama.attention.head_count' in GGUF metadata".to_string())?
        as usize;

    let num_kv_heads = meta
        .get("llama.attention.head_count_kv")
        .and_then(|v| v.as_u32())
        .map(|v| v as usize)
        .unwrap_or(num_heads);

    let intermediate_size = meta
        .get("llama.feed_forward_length")
        .and_then(|v| v.as_u32())
        .map(|v| v as usize)
        .unwrap_or(d_model * 4);

    let max_seq_len = meta
        .get("llama.context_length")
        .and_then(|v| v.as_u32())
        .map(|v| v as usize)
        .unwrap_or(2048);

    let norm_eps = meta
        .get("llama.attention.layer_norm_rms_epsilon")
        .and_then(|v| v.as_f32())
        .unwrap_or(1e-5);

    let rope_theta = meta
        .get("llama.rope.freq_base")
        .and_then(|v| v.as_f32())
        .unwrap_or(10000.0);

    Ok(LayeredConfig {
        shard_dir: path.to_path_buf(),
        num_layers,
        d_model,
        vocab_size,
        num_heads,
        num_kv_heads,
        intermediate_size,
        max_seq_len,
        norm_eps,
        rope_theta,
        cache_fraction: 0.0,
        prefetch_lookahead: 0,
        embed_name: "model.embed_tokens".into(),
        layer_prefix: "model.layers".into(),
        norm_name: "model.norm".into(),
        lm_head_name: "lm_head".into(),
    })
}

/// Dequantize a named GGUF tensor to a raw little-endian `f32` byte buffer.
fn dequantize_tensor(model: &GgufModel, tensor_name: &str) -> Result<Vec<u8>, String> {
    let info = model
        .file
        .tensors
        .get(tensor_name)
        .map_err(|e| format!("missing tensor '{tensor_name}': {e}"))?;

    let raw_bytes = model
        .tensor_data(tensor_name)
        .map_err(|e| format!("failed to read tensor data for '{tensor_name}': {e}"))?;

    let n_elements: usize = info.dimensions.iter().map(|&d| d as usize).product();
    let f32_data = oxillama_quant::dequantize_to_f32(raw_bytes, info.tensor_type, n_elements)
        .map_err(|e| {
            format!(
                "failed to dequantize '{tensor_name}' ({:?}): {e}",
                info.tensor_type
            )
        })?;

    Ok(bytemuck::cast_slice::<f32, u8>(&f32_data).to_vec())
}

/// Load a layer from a GGUF file and return its parameter weights in the naming
/// format expected by `LayeredEngine`.
pub fn load_layer_from_gguf(gguf_path: &Path, layer_name: &str) -> Result<LayerWeights, String> {
    let model = get_or_load_gguf(gguf_path)?;
    let mut weights = Vec::new();

    if layer_name == "model.embed_tokens" {
        let bytes = dequantize_tensor(&model, "token_embd.weight")?;
        weights.push(("weight".to_string(), bytes));
        return Ok(Arc::new(weights));
    }

    if layer_name == "model.norm" {
        let bytes = dequantize_tensor(&model, "output_norm.weight")?;
        weights.push(("weight".to_string(), bytes));
        return Ok(Arc::new(weights));
    }

    if layer_name == "lm_head" {
        let bytes = if model.file.tensors.get("output.weight").is_ok() {
            dequantize_tensor(&model, "output.weight")?
        } else {
            // Weight tying: fallback to embedding table
            dequantize_tensor(&model, "token_embd.weight")?
        };
        weights.push(("weight".to_string(), bytes));
        return Ok(Arc::new(weights));
    }

    // Transformer block layers: "model.layers.{i}"
    let prefix = "model.layers.";
    if let Some(rest) = layer_name.strip_prefix(prefix) {
        let layer_idx: usize = rest
            .parse()
            .map_err(|e| format!("invalid layer index in '{layer_name}': {e}"))?;

        let mapping = [
            ("attn_norm.weight", "input_layernorm.weight"),
            ("attn_q.weight", "self_attn.q_proj.weight"),
            ("attn_k.weight", "self_attn.k_proj.weight"),
            ("attn_v.weight", "self_attn.v_proj.weight"),
            ("attn_output.weight", "self_attn.o_proj.weight"),
            ("ffn_norm.weight", "post_attention_layernorm.weight"),
            ("ffn_gate.weight", "mlp.gate_proj.weight"),
            ("ffn_up.weight", "mlp.up_proj.weight"),
            ("ffn_down.weight", "mlp.down_proj.weight"),
        ];

        for (gguf_suffix, kore_name) in mapping {
            let gguf_tensor = format!("blk.{layer_idx}.{gguf_suffix}");
            let bytes = dequantize_tensor(&model, &gguf_tensor)?;
            weights.push((kore_name.to_string(), bytes));
        }

        // Check for optional biases (e.g. q_proj.bias if present)
        let bias_mapping = [
            ("attn_q.bias", "self_attn.q_proj.bias"),
            ("attn_k.bias", "self_attn.k_proj.bias"),
            ("attn_v.bias", "self_attn.v_proj.bias"),
            ("attn_output.bias", "self_attn.o_proj.bias"),
        ];
        for (gguf_suffix, kore_name) in bias_mapping {
            let gguf_tensor = format!("blk.{layer_idx}.{gguf_suffix}");
            if model.file.tensors.get(&gguf_tensor).is_ok() {
                let bytes = dequantize_tensor(&model, &gguf_tensor)?;
                weights.push((kore_name.to_string(), bytes));
            }
        }

        return Ok(Arc::new(weights));
    }

    Err(format!("unrecognized layer name for GGUF: '{layer_name}'"))
}
