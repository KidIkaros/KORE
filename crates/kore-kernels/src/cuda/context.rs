//! CUDA device context management.
//!
//! Provides lazy-initialized singleton `CudaDevice` handles per GPU index.
//! Uses `cudarc` for safe CUDA driver API access.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaDevice, CudaFunction};
use parking_lot::Mutex;

/// Global registry of CUDA device handles (one per GPU index).
static DEVICES: OnceLock<Mutex<HashMap<usize, Arc<CudaDevice>>>> = OnceLock::new();

fn devices() -> &'static Mutex<HashMap<usize, Arc<CudaDevice>>> {
    DEVICES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Get or create a CUDA device handle for the given GPU index.
///
/// The device is lazily initialized on first access and cached for reuse.
pub fn get_device(device_idx: usize) -> Result<Arc<CudaDevice>, CudaError> {
    let mut map = devices().lock();
    if let Some(dev) = map.get(&device_idx) {
        return Ok(Arc::clone(dev));
    }
    let dev = CudaDevice::new(device_idx).map_err(|e| CudaError::DeviceInit(
        format!("device {}: {}", device_idx, e),
    ))?;
    map.insert(device_idx, Arc::clone(&dev));
    Ok(dev)
}

/// Get a compiled CUDA function from a previously loaded module.
pub fn get_func(
    device: &Arc<CudaDevice>,
    module_name: &str,
    func_name: &str,
) -> Result<CudaFunction, CudaError> {
    device
        .get_func(module_name, func_name)
        .ok_or_else(|| CudaError::FuncNotFound {
            module: module_name.to_string(),
            func: func_name.to_string(),
        })
}

/// Check if any CUDA device is available.
pub fn is_cuda_available() -> bool {
    CudaDevice::new(0).is_ok()
}

/// Number of available CUDA devices.
pub fn device_count() -> usize {
    (0..16).take_while(|&i| CudaDevice::new(i).is_ok()).count()
}

/// CUDA-specific errors.
#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("CUDA device init failed: {0}")]
    DeviceInit(String),

    #[error("PTX compilation failed for module '{module}': {msg}")]
    PtxCompile { module: String, msg: String },

    #[error("Failed to load module '{module}': {msg}")]
    ModuleLoad { module: String, msg: String },

    #[error("Function '{func}' not found in module '{module}'")]
    FuncNotFound { module: String, func: String },

    #[error("CUDA kernel launch failed: {0}")]
    LaunchError(String),

    #[error("CUDA memory error: {0}")]
    MemoryError(String),

    #[error("Device mismatch: tensors on different devices")]
    DeviceMismatch,
}
