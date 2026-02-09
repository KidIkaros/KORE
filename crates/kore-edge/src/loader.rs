//! Load HuggingFace safetensors models directly into .koref format.
//!
//! Provides `load_safetensors_dir` which reads `config.json` + `.safetensors`
//! files from a model directory and produces a `KorefModel` ready for inference.

#[cfg(feature = "loader")]
use std::collections::HashMap;
#[cfg(feature = "loader")]
use std::path::Path;

#[cfg(feature = "loader")]
use crate::format::{EdgeDType, FormatError, KorefBuilder, KorefModel};

/// Subset of HuggingFace `config.json` fields.
#[cfg(all(feature = "loader", feature = "serde"))]
#[derive(serde::Deserialize, Debug)]
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
    #[serde(alias = "num_key_value_heads")]
    pub n_kv_heads: Option<usize>,
    #[serde(alias = "rms_norm_eps", alias = "layer_norm_epsilon")]
    pub norm_eps: Option<f64>,
    #[serde(alias = "rope_theta")]
    pub rope_base: Option<f64>,
    #[serde(alias = "model_type")]
    pub model_type: Option<String>,
}

/// Load a HuggingFace model directory into a `KorefModel`.
///
/// The directory should contain:
/// - `config.json` — model configuration
/// - One or more `.safetensors` files — model weights
///
/// All weights are converted to f32 and packed into the `.koref` format.
#[cfg(all(feature = "loader", feature = "serde"))]
pub fn load_safetensors_dir(model_dir: &Path) -> Result<KorefModel, FormatError> {
    // Load config.json
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| FormatError::IoError(format!("Failed to read config.json: {}", e)))?;
    let config: HfConfig = serde_json::from_str(&config_str)
        .map_err(|e| FormatError::InvalidHeader(format!("Failed to parse config.json: {}", e)))?;

    let model_type = config.model_type.as_deref().unwrap_or("llama");
    let vocab_size = config.vocab_size.unwrap_or(32000);
    let d_model = config.d_model.unwrap_or(4096);
    let n_heads = config.n_heads.unwrap_or(32);
    let n_kv_heads = config.n_kv_heads.unwrap_or(n_heads);
    let n_layers = config.n_layers.unwrap_or(32);
    let d_ff = config.d_ff.unwrap_or(11008);
    let max_seq_len = config.max_seq_len.unwrap_or(2048);
    let norm_eps = config.norm_eps.unwrap_or(1e-5) as f32;
    let rope_base = config.rope_base.unwrap_or(10000.0) as f32;

    let mut builder = KorefBuilder::new(
        model_type, vocab_size, d_model, n_heads, n_kv_heads,
        n_layers, d_ff, max_seq_len, norm_eps, rope_base,
    );

    // Find .safetensors files
    let mut st_files: Vec<std::path::PathBuf> = std::fs::read_dir(model_dir)
        .map_err(|e| FormatError::IoError(format!("Cannot read dir: {}", e)))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
        .collect();
    st_files.sort();

    if st_files.is_empty() {
        return Err(FormatError::IoError(
            "No .safetensors files found in model directory".into(),
        ));
    }

    let mut tensor_count = 0usize;

    for st_path in &st_files {
        let file_data = std::fs::read(st_path)
            .map_err(|e| FormatError::IoError(format!("Failed to read {}: {}", st_path.display(), e)))?;

        let tensors = safetensors::SafeTensors::deserialize(&file_data)
            .map_err(|e| FormatError::InvalidHeader(format!("safetensors parse error: {}", e)))?;

        for (name, view) in tensors.tensors() {
            let shape: Vec<usize> = view.shape().to_vec();
            let f32_data = safetensor_view_to_f32(&view)?;
            builder.add_f32(&name, &shape, &f32_data);
            tensor_count += 1;
        }
    }

    if tensor_count == 0 {
        return Err(FormatError::IoError("No tensors found in safetensors files".into()));
    }

    Ok(builder.build())
}

/// Convert a safetensors tensor view to f32 data.
#[cfg(feature = "loader")]
fn safetensor_view_to_f32(view: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>, FormatError> {
    let dtype = view.dtype();
    let data = view.data();

    match dtype {
        safetensors::Dtype::F32 => {
            Ok(data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
        safetensors::Dtype::F16 => {
            Ok(data.chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect())
        }
        safetensors::Dtype::BF16 => {
            Ok(data.chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect())
        }
        _ => Err(FormatError::InvalidHeader(
            format!("Unsupported safetensors dtype: {:?}", dtype),
        )),
    }
}

/// List tensor names and shapes from a safetensors file.
#[cfg(feature = "loader")]
pub fn inspect_safetensors(path: &Path) -> Result<HashMap<String, Vec<usize>>, FormatError> {
    let data = std::fs::read(path)
        .map_err(|e| FormatError::IoError(format!("Failed to read: {}", e)))?;
    let tensors = safetensors::SafeTensors::deserialize(&data)
        .map_err(|e| FormatError::InvalidHeader(format!("Failed to parse: {}", e)))?;

    let mut map = HashMap::new();
    for (name, view) in tensors.tensors() {
        map.insert(name.to_string(), view.shape().to_vec());
    }
    Ok(map)
}

#[cfg(test)]
#[cfg(all(feature = "loader", feature = "serde"))]
mod tests {
    use super::*;

    #[test]
    fn test_hf_config_parse() {
        let json = r#"{
            "model_type": "llama",
            "hidden_size": 256,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "num_hidden_layers": 2,
            "intermediate_size": 512,
            "vocab_size": 1000,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0
        }"#;
        let config: HfConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.d_model, Some(256));
        assert_eq!(config.n_heads, Some(4));
        assert_eq!(config.n_kv_heads, Some(4));
        assert_eq!(config.n_layers, Some(2));
        assert_eq!(config.d_ff, Some(512));
        assert_eq!(config.vocab_size, Some(1000));
        assert_eq!(config.model_type, Some("llama".to_string()));
    }
}
