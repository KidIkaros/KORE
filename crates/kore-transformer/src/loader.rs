//! Safetensors model loading.
//!
//! Loads transformer weights from `.safetensors` files into a `Transformer` model.
//! Supports both single-file and sharded (multi-file) models.

use std::collections::HashMap;
use std::path::Path;

use kore_core::{KoreError, Tensor};
use safetensors::SafeTensors;
use serde::Deserialize;

use crate::model::{Transformer, TransformerConfig};

// ============================================================================
// Config loading (HuggingFace-style config.json)
// ============================================================================

/// Subset of HuggingFace `config.json` fields we care about.
#[derive(Deserialize, Debug)]
pub struct HfConfig {
    #[serde(alias = "hidden_size")]
    pub d_model: Option<usize>,
    #[serde(alias = "num_attention_heads")]
    pub n_heads: Option<usize>,
    #[serde(alias = "num_hidden_layers", alias = "num_layers")]
    pub n_layers: Option<usize>,
    #[serde(alias = "intermediate_size")]
    pub d_ff: Option<usize>,
    #[serde(alias = "vocab_size")]
    pub vocab_size: Option<usize>,
    #[serde(alias = "max_position_embeddings", alias = "max_seq_len")]
    pub max_seq_len: Option<usize>,
    #[serde(alias = "rms_norm_eps", alias = "layer_norm_epsilon")]
    pub norm_eps: Option<f64>,
}

impl HfConfig {
    /// Load from a `config.json` file.
    pub fn from_file(path: &Path) -> Result<Self, KoreError> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| KoreError::StorageError(format!("Failed to read config: {}", e)))?;
        let config: HfConfig = serde_json::from_str(&data)
            .map_err(|e| KoreError::StorageError(format!("Failed to parse config: {}", e)))?;
        Ok(config)
    }

    /// Convert to a `TransformerConfig`.
    pub fn to_transformer_config(&self) -> TransformerConfig {
        TransformerConfig {
            vocab_size: self.vocab_size.unwrap_or(32000),
            d_model: self.d_model.unwrap_or(4096),
            n_heads: self.n_heads.unwrap_or(32),
            n_layers: self.n_layers.unwrap_or(32),
            d_ff: self.d_ff.unwrap_or(11008),
            max_seq_len: self.max_seq_len.unwrap_or(2048),
            norm_eps: self.norm_eps.unwrap_or(1e-5) as f32,
        }
    }
}

// ============================================================================
// Weight name mapping
// ============================================================================

/// Maps HuggingFace weight names to our internal names.
///
/// Common patterns:
/// - `model.embed_tokens.weight` → embedding
/// - `model.layers.{i}.self_attn.q_proj.weight` → layer i Q projection
/// - `model.layers.{i}.self_attn.k_proj.weight` → layer i K projection
/// - `model.layers.{i}.self_attn.v_proj.weight` → layer i V projection
/// - `model.layers.{i}.self_attn.o_proj.weight` → layer i output projection
/// - `model.layers.{i}.mlp.gate_proj.weight` → layer i FFN W1 (gate)
/// - `model.layers.{i}.mlp.down_proj.weight` → layer i FFN W2 (down)
/// - `model.layers.{i}.mlp.up_proj.weight` → layer i FFN W3 (up)
/// - `model.layers.{i}.input_layernorm.weight` → layer i attn norm
/// - `model.layers.{i}.post_attention_layernorm.weight` → layer i ffn norm
/// - `model.norm.weight` → final norm
/// - `lm_head.weight` → output projection
#[derive(Debug)]
pub enum WeightTarget {
    EmbedTokens,
    LayerAttnNorm(usize),
    LayerQ(usize),
    LayerK(usize),
    LayerV(usize),
    LayerO(usize),
    LayerFfnNorm(usize),
    LayerGate(usize),
    LayerDown(usize),
    LayerUp(usize),
    FinalNorm,
    LmHead,
    Unknown(String),
}

/// Parse a HuggingFace weight name into a `WeightTarget`.
pub fn parse_weight_name(name: &str) -> WeightTarget {
    if name == "model.embed_tokens.weight" || name == "embed_tokens.weight" {
        return WeightTarget::EmbedTokens;
    }
    if name == "model.norm.weight" || name == "norm.weight" {
        return WeightTarget::FinalNorm;
    }
    if name == "lm_head.weight" {
        return WeightTarget::LmHead;
    }

    // Try to parse layer index
    if let Some(rest) = name.strip_prefix("model.layers.").or_else(|| name.strip_prefix("layers.")) {
        if let Some(dot_pos) = rest.find('.') {
            if let Ok(layer_idx) = rest[..dot_pos].parse::<usize>() {
                let suffix = &rest[dot_pos + 1..];
                return match suffix {
                    "self_attn.q_proj.weight" => WeightTarget::LayerQ(layer_idx),
                    "self_attn.k_proj.weight" => WeightTarget::LayerK(layer_idx),
                    "self_attn.v_proj.weight" => WeightTarget::LayerV(layer_idx),
                    "self_attn.o_proj.weight" => WeightTarget::LayerO(layer_idx),
                    "mlp.gate_proj.weight" => WeightTarget::LayerGate(layer_idx),
                    "mlp.down_proj.weight" => WeightTarget::LayerDown(layer_idx),
                    "mlp.up_proj.weight" => WeightTarget::LayerUp(layer_idx),
                    "input_layernorm.weight" => WeightTarget::LayerAttnNorm(layer_idx),
                    "post_attention_layernorm.weight" => WeightTarget::LayerFfnNorm(layer_idx),
                    _ => WeightTarget::Unknown(name.to_string()),
                };
            }
        }
    }

    WeightTarget::Unknown(name.to_string())
}

// ============================================================================
// Tensor conversion from safetensors
// ============================================================================

/// Convert a safetensors tensor view to a kore Tensor (f32).
fn safetensor_to_kore(view: &safetensors::tensor::TensorView<'_>) -> Result<Tensor, KoreError> {
    let shape: Vec<usize> = view.shape().to_vec();
    let dtype = view.dtype();
    let data = view.data();

    let f32_data: Vec<f32> = match dtype {
        safetensors::Dtype::F32 => {
            data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        }
        safetensors::Dtype::F16 => {
            data.chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect()
        }
        safetensors::Dtype::BF16 => {
            data.chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect()
        }
        _ => {
            return Err(KoreError::StorageError(
                format!("Unsupported safetensors dtype: {:?}", dtype),
            ));
        }
    };

    Ok(Tensor::from_f32(&f32_data, &shape))
}

// ============================================================================
// Model loading
// ============================================================================

/// Load a transformer model from a directory containing `config.json` and
/// `.safetensors` weight file(s).
pub fn load_model(model_dir: &Path) -> Result<Transformer, KoreError> {
    // Load config
    let config_path = model_dir.join("config.json");
    let hf_config = HfConfig::from_file(&config_path)?;
    let config = hf_config.to_transformer_config();

    eprintln!(
        "Loading model: vocab={}, d_model={}, n_heads={}, n_layers={}, d_ff={}",
        config.vocab_size, config.d_model, config.n_heads, config.n_layers, config.d_ff
    );

    let mut model = Transformer::new(config);

    // Find safetensors files
    let st_files = find_safetensor_files(model_dir)?;
    if st_files.is_empty() {
        return Err(KoreError::StorageError(
            "No .safetensors files found in model directory".into(),
        ));
    }

    eprintln!("Found {} safetensors file(s)", st_files.len());

    // Load weights from each file
    let mut loaded_count = 0usize;
    let mut skipped = Vec::new();

    for st_path in &st_files {
        let file_data = std::fs::read(st_path)
            .map_err(|e| KoreError::StorageError(format!("Failed to read {}: {}", st_path.display(), e)))?;

        let tensors = SafeTensors::deserialize(&file_data)
            .map_err(|e| KoreError::StorageError(format!("Failed to parse safetensors: {}", e)))?;

        for (name, view) in tensors.tensors() {
            let target = parse_weight_name(&name);
            let tensor = safetensor_to_kore(&view)?;

            match target {
                WeightTarget::EmbedTokens => {
                    model.embedding.weight = tensor;
                    loaded_count += 1;
                }
                WeightTarget::FinalNorm => {
                    model.final_norm.weight = tensor;
                    loaded_count += 1;
                }
                WeightTarget::LmHead => {
                    model.lm_head = tensor;
                    loaded_count += 1;
                }
                WeightTarget::LayerAttnNorm(i) if i < model.layers.len() => {
                    model.layers[i].attn_norm.weight = tensor;
                    loaded_count += 1;
                }
                WeightTarget::LayerQ(i) if i < model.layers.len() => {
                    model.layers[i].attn.wq = tensor;
                    loaded_count += 1;
                }
                WeightTarget::LayerK(i) if i < model.layers.len() => {
                    model.layers[i].attn.wk = tensor;
                    loaded_count += 1;
                }
                WeightTarget::LayerV(i) if i < model.layers.len() => {
                    model.layers[i].attn.wv = tensor;
                    loaded_count += 1;
                }
                WeightTarget::LayerO(i) if i < model.layers.len() => {
                    model.layers[i].attn.wo = tensor;
                    loaded_count += 1;
                }
                WeightTarget::LayerFfnNorm(i) if i < model.layers.len() => {
                    model.layers[i].ffn_norm.weight = tensor;
                    loaded_count += 1;
                }
                WeightTarget::LayerGate(i) if i < model.layers.len() => {
                    model.layers[i].ffn.w1 = tensor;
                    loaded_count += 1;
                }
                WeightTarget::LayerDown(i) if i < model.layers.len() => {
                    model.layers[i].ffn.w2 = tensor;
                    loaded_count += 1;
                }
                WeightTarget::LayerUp(i) if i < model.layers.len() => {
                    model.layers[i].ffn.w3 = tensor;
                    loaded_count += 1;
                }
                _ => {
                    skipped.push(name.to_string());
                }
            }
        }
    }

    eprintln!("Loaded {} tensors, skipped {} unknown", loaded_count, skipped.len());
    if !skipped.is_empty() && skipped.len() <= 10 {
        for s in &skipped {
            eprintln!("  skipped: {}", s);
        }
    }

    Ok(model)
}

/// Find all `.safetensors` files in a directory.
fn find_safetensor_files(dir: &Path) -> Result<Vec<std::path::PathBuf>, KoreError> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| KoreError::StorageError(format!("Cannot read dir {}: {}", dir.display(), e)))?;

    let mut files: Vec<std::path::PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
        .collect();

    files.sort();
    Ok(files)
}

/// List the tensor names and shapes in a safetensors file (for inspection).
pub fn inspect_safetensors(path: &Path) -> Result<HashMap<String, Vec<usize>>, KoreError> {
    let data = std::fs::read(path)
        .map_err(|e| KoreError::StorageError(format!("Failed to read: {}", e)))?;
    let tensors = SafeTensors::deserialize(&data)
        .map_err(|e| KoreError::StorageError(format!("Failed to parse: {}", e)))?;

    let mut map = HashMap::new();
    for (name, view) in tensors.tensors() {
        map.insert(name.to_string(), view.shape().to_vec());
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_weight_names() {
        assert!(matches!(
            parse_weight_name("model.embed_tokens.weight"),
            WeightTarget::EmbedTokens
        ));
        assert!(matches!(
            parse_weight_name("model.norm.weight"),
            WeightTarget::FinalNorm
        ));
        assert!(matches!(
            parse_weight_name("lm_head.weight"),
            WeightTarget::LmHead
        ));
        assert!(matches!(
            parse_weight_name("model.layers.0.self_attn.q_proj.weight"),
            WeightTarget::LayerQ(0)
        ));
        assert!(matches!(
            parse_weight_name("model.layers.15.mlp.gate_proj.weight"),
            WeightTarget::LayerGate(15)
        ));
        assert!(matches!(
            parse_weight_name("model.layers.3.post_attention_layernorm.weight"),
            WeightTarget::LayerFfnNorm(3)
        ));
        assert!(matches!(
            parse_weight_name("some.unknown.weight"),
            WeightTarget::Unknown(_)
        ));
    }

    #[test]
    fn test_hf_config_deserialize() {
        let json = r#"{
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-5
        }"#;
        let config: HfConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.d_model, Some(4096));
        assert_eq!(config.n_heads, Some(32));
        assert_eq!(config.n_layers, Some(32));
        assert_eq!(config.d_ff, Some(11008));
        assert_eq!(config.vocab_size, Some(32000));

        let tc = config.to_transformer_config();
        assert_eq!(tc.d_model, 4096);
        assert_eq!(tc.n_layers, 32);
    }

    #[test]
    fn test_safetensor_to_kore_f32() {
        // Create a minimal safetensors buffer with one f32 tensor
        use safetensors::tensor::serialize;
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let tensors = vec![(
            "test".to_string(),
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                vec![2, 3],
                &bytes,
            ).unwrap(),
        )];

        let serialized = serialize(tensors.iter().map(|(n, v)| (n.as_str(), v.clone())), &None).unwrap();
        let loaded = SafeTensors::deserialize(&serialized).unwrap();

        for (name, view) in loaded.tensors() {
            assert_eq!(name, "test");
            let tensor = safetensor_to_kore(&view).unwrap();
            assert_eq!(tensor.shape().dims(), &[2, 3]);
            let slice = tensor.as_f32_slice().unwrap();
            assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        }
    }
}
