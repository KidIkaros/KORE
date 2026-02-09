//! Model save/load using the safetensors format.
//!
//! Provides generic `save_state_dict` and `load_state_dict` functions that work
//! with any type implementing the `Module` trait.

use std::collections::HashMap;
use std::path::Path;

use kore_core::{KoreError, Tensor};
use safetensors::SafeTensors;
use safetensors::tensor::{TensorView, serialize};

/// Save a state dictionary (name → Tensor) to a safetensors file.
pub fn save_state_dict(
    state_dict: &HashMap<String, Tensor>,
    path: &Path,
) -> Result<(), KoreError> {
    let mut views = Vec::new();
    let mut buffers: Vec<(String, Vec<u8>)> = Vec::new();

    // Convert each tensor to raw f32 bytes
    for (name, tensor) in state_dict {
        let data = tensor.contiguous();
        let slice = data.as_f32_slice().ok_or_else(|| {
            KoreError::UnsupportedDType(tensor.dtype())
        })?;
        let bytes: Vec<u8> = slice.iter().flat_map(|f| f.to_le_bytes()).collect();
        buffers.push((name.clone(), bytes));
    }

    // Build TensorViews referencing the buffers
    for (name, bytes) in &buffers {
        let tensor = state_dict.get(name).unwrap();
        let shape: Vec<usize> = tensor.shape().dims().to_vec();
        let view = TensorView::new(safetensors::Dtype::F32, shape, bytes)
            .map_err(|e| KoreError::StorageError(format!("safetensors view error: {}", e)))?;
        views.push((name.as_str(), view));
    }

    let serialized = serialize(views, &None)
        .map_err(|e| KoreError::StorageError(format!("safetensors serialize error: {}", e)))?;

    std::fs::write(path, &serialized)
        .map_err(|e| KoreError::StorageError(format!("Failed to write {}: {}", path.display(), e)))?;

    Ok(())
}

/// Load a state dictionary from a safetensors file.
/// Returns a HashMap of name → Tensor.
pub fn load_state_dict(path: &Path) -> Result<HashMap<String, Tensor>, KoreError> {
    let data = std::fs::read(path)
        .map_err(|e| KoreError::StorageError(format!("Failed to read {}: {}", path.display(), e)))?;

    let tensors = SafeTensors::deserialize(&data)
        .map_err(|e| KoreError::StorageError(format!("safetensors parse error: {}", e)))?;

    let mut result = HashMap::new();
    for (name, view) in tensors.tensors() {
        let tensor = safetensor_view_to_tensor(&view)?;
        result.insert(name.to_string(), tensor);
    }

    Ok(result)
}

/// Save a Module's parameters to a safetensors file.
pub fn save_module(module: &dyn crate::Module, path: &Path) -> Result<(), KoreError> {
    let state_dict = module.state_dict();
    save_state_dict(&state_dict, path)
}

/// Load parameters from a safetensors file and return them as a state dict.
/// The caller is responsible for assigning the tensors to the module's fields.
pub fn load_module_state(path: &Path) -> Result<HashMap<String, Tensor>, KoreError> {
    load_state_dict(path)
}

/// Convert a safetensors TensorView to a kore Tensor (f32).
fn safetensor_view_to_tensor(view: &safetensors::tensor::TensorView<'_>) -> Result<Tensor, KoreError> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Linear, Module};

    #[test]
    fn test_save_load_roundtrip() {
        let layer = Linear::new(4, 3, true);
        let sd = layer.state_dict();

        let dir = std::env::temp_dir();
        let path = dir.join("kore_test_save_load.safetensors");

        // Save
        save_state_dict(&sd, &path).unwrap();
        assert!(path.exists());

        // Load
        let loaded = load_state_dict(&path).unwrap();
        assert!(loaded.contains_key("weight"));
        assert!(loaded.contains_key("bias"));

        // Verify shapes match
        assert_eq!(
            loaded["weight"].shape().dims(),
            sd["weight"].shape().dims()
        );
        assert_eq!(
            loaded["bias"].shape().dims(),
            sd["bias"].shape().dims()
        );

        // Verify data matches
        let orig = sd["weight"].as_f32_slice().unwrap();
        let load = loaded["weight"].as_f32_slice().unwrap();
        assert_eq!(orig.len(), load.len());
        for (&a, &b) in orig.iter().zip(load.iter()) {
            assert!((a - b).abs() < 1e-7f32, "data mismatch: {} vs {}", a, b);
        }

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_module() {
        let layer = Linear::new(2, 3, false);
        let dir = std::env::temp_dir();
        let path = dir.join("kore_test_save_module.safetensors");

        save_module(&layer, &path).unwrap();
        assert!(path.exists());

        let loaded = load_module_state(&path).unwrap();
        assert!(loaded.contains_key("weight"));
        assert_eq!(loaded["weight"].shape().dims(), &[3, 2]);

        let _ = std::fs::remove_file(&path);
    }
}
