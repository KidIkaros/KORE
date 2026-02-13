//! Model sharder — splits multi-file safetensors into per-layer shard files.
//!
//! This is a one-time offline conversion step. Given a directory of HuggingFace
//! safetensors model files, this module produces one `.safetensors` file per
//! layer (plus embed, norm, and lm_head), suitable for layer-by-layer loading.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::config::LayeredConfig;

/// Shard a HuggingFace model directory into per-layer safetensors files.
///
/// Reads all `*.safetensors` files from `model_dir`, groups tensors by layer,
/// and writes one file per layer to `output_dir`.
///
/// # Returns
///
/// The number of shard files written.
pub fn shard_model(
    model_dir: &Path,
    output_dir: &Path,
    config: &LayeredConfig,
) -> Result<usize, String> {
    std::fs::create_dir_all(output_dir)
        .map_err(|e| format!("failed to create output dir: {e}"))?;

    // Collect all safetensors files
    let st_files = find_safetensors_files(model_dir)?;
    if st_files.is_empty() {
        return Err(format!("no .safetensors files found in {}", model_dir.display()));
    }

    tracing::info!("Found {} safetensors files in {}", st_files.len(), model_dir.display());

    // Group all tensors by layer
    let layer_names = config.layer_names();
    type TensorEntry = (String, Vec<u8>, safetensors::Dtype, Vec<usize>);
    let mut layer_tensors: HashMap<String, Vec<TensorEntry>> =
        HashMap::new();

    for file_path in &st_files {
        let file_bytes = std::fs::read(file_path)
            .map_err(|e| format!("failed to read {}: {e}", file_path.display()))?;

        let tensors = safetensors::SafeTensors::deserialize(&file_bytes)
            .map_err(|e| format!("failed to parse {}: {e}", file_path.display()))?;

        for (tensor_name, view) in tensors.tensors() {
            let layer_key = classify_tensor(&tensor_name, &layer_names, config);
            let entry = layer_tensors.entry(layer_key).or_default();
            entry.push((
                tensor_name.to_string(),
                view.data().to_vec(),
                view.dtype(),
                view.shape().to_vec(),
            ));
        }
    }

    // Write per-layer shard files
    let mut count = 0;
    for (layer_key, tensors) in &layer_tensors {
        let safe_name = layer_key.replace('.', "_");
        let out_path = output_dir.join(format!("{safe_name}.safetensors"));

        write_shard_file(&out_path, tensors)?;
        count += 1;

        tracing::debug!(
            "Wrote shard: {} ({} tensors, {:.1} MB)",
            out_path.display(),
            tensors.len(),
            tensors.iter().map(|(_, d, _, _)| d.len()).sum::<usize>() as f64 / 1e6,
        );
    }

    tracing::info!("Sharded model into {} per-layer files in {}", count, output_dir.display());
    Ok(count)
}

/// Classify a tensor name into its corresponding layer key.
///
/// Maps tensor names like `model.layers.5.self_attn.q_proj.weight` to
/// the layer key `model.layers.5`.
fn classify_tensor(
    tensor_name: &str,
    layer_names: &[String],
    config: &LayeredConfig,
) -> String {
    // Check embed
    if tensor_name.starts_with(&config.embed_name) {
        return config.embed_name.clone();
    }

    // Check transformer layers
    for i in 0..config.num_layers {
        let prefix = format!("{}.{}.", config.layer_prefix, i);
        if tensor_name.starts_with(&prefix) {
            return format!("{}.{}", config.layer_prefix, i);
        }
    }

    // Check norm
    if tensor_name.starts_with(&config.norm_name) {
        return config.norm_name.clone();
    }

    // Check lm_head
    if tensor_name.starts_with(&config.lm_head_name) {
        return config.lm_head_name.clone();
    }

    // Fallback: use the first two segments as the key
    let parts: Vec<&str> = tensor_name.splitn(3, '.').collect();
    if parts.len() >= 2 {
        tracing::warn!(
            "tensor '{}' did not match any known layer pattern, grouping by prefix '{}.{}'",
            tensor_name, parts[0], parts[1]
        );
        format!("{}.{}", parts[0], parts[1])
    } else {
        tracing::warn!(
            "tensor '{}' could not be classified, assigning to first layer",
            tensor_name
        );
        layer_names.first()
            .cloned()
            .unwrap_or_else(|| "unknown".to_string())
    }
}

/// Write a single shard file containing the given tensors.
fn write_shard_file(
    path: &Path,
    tensors: &[(String, Vec<u8>, safetensors::Dtype, Vec<usize>)],
) -> Result<(), String> {
    let tensor_views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
        .iter()
        .map(|(name, data, dtype, shape)| {
            let view = safetensors::tensor::TensorView::new(*dtype, shape.clone(), data)
                .map_err(|e| format!("invalid tensor {name}: {e}"))
                .expect("valid tensor view");
            (name.clone(), view)
        })
        .collect();

    let bytes = safetensors::tensor::serialize(tensor_views, &None)
        .map_err(|e| format!("failed to serialize shard: {e}"))?;

    std::fs::write(path, bytes)
        .map_err(|e| format!("failed to write {}: {e}", path.display()))?;

    Ok(())
}

/// Find all `.safetensors` files in a directory.
fn find_safetensors_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("cannot read directory {}: {e}", dir.display()))?;

    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .collect();

    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_tensor() {
        let config = LayeredConfig::llama(
            PathBuf::from("/tmp"), 32, 4096, 32000, 32, 32, 11008,
        );
        let layer_names = config.layer_names();

        assert_eq!(
            classify_tensor("model.embed_tokens.weight", &layer_names, &config),
            "model.embed_tokens"
        );
        assert_eq!(
            classify_tensor("model.layers.5.self_attn.q_proj.weight", &layer_names, &config),
            "model.layers.5"
        );
        assert_eq!(
            classify_tensor("model.layers.31.mlp.gate_proj.weight", &layer_names, &config),
            "model.layers.31"
        );
        assert_eq!(
            classify_tensor("model.norm.weight", &layer_names, &config),
            "model.norm"
        );
        assert_eq!(
            classify_tensor("lm_head.weight", &layer_names, &config),
            "lm_head"
        );
    }
}
