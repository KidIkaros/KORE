//! CUDA kernel launcher with PTX compilation and caching.
//!
//! Compiles PTX source at runtime via NVRTC, caches compiled modules,
//! and provides ergonomic kernel launch helpers.

use std::collections::HashSet;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, LaunchConfig};
use parking_lot::Mutex;

use super::context::CudaError;

/// Registry of compiled PTX modules per device.
/// Key: (device_idx, module_name)
static LOADED: std::sync::OnceLock<Mutex<HashSet<(usize, String)>>> =
    std::sync::OnceLock::new();

fn loaded_set() -> &'static Mutex<HashSet<(usize, String)>> {
    LOADED.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Ensure a PTX module is compiled and loaded on the given device.
/// No-op if already loaded.
pub fn ensure_module(
    device: &Arc<CudaDevice>,
    device_idx: usize,
    module_name: &str,
    ptx_source: &str,
) -> Result<(), CudaError> {
    let key = (device_idx, module_name.to_string());
    {
        let set = loaded_set().lock();
        if set.contains(&key) {
            return Ok(());
        }
    }

    let ptx = cudarc::nvrtc::compile_ptx(ptx_source).map_err(|e| CudaError::PtxCompile {
        module: module_name.to_string(),
        msg: e.to_string(),
    })?;

    device
        .load_ptx(ptx, module_name, &[])
        .map_err(|e| CudaError::ModuleLoad {
            module: module_name.to_string(),
            msg: e.to_string(),
        })?;

    loaded_set().lock().insert(key);
    Ok(())
}

/// Get a kernel function handle, loading the module if needed.
pub fn get_or_load_func(
    device: &Arc<CudaDevice>,
    device_idx: usize,
    module_name: &str,
    func_name: &str,
    ptx_source: &str,
) -> Result<CudaFunction, CudaError> {
    ensure_module(device, device_idx, module_name, ptx_source)?;
    device
        .get_func(module_name, func_name)
        .ok_or_else(|| CudaError::FuncNotFound {
            module: module_name.to_string(),
            func: func_name.to_string(),
        })
}

/// Compute grid dimensions for a 1D kernel launch.
pub fn grid_1d(n: usize, block_size: usize) -> LaunchConfig {
    let grid = (n + block_size - 1) / block_size;
    LaunchConfig {
        grid_dim: (grid as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Compute grid dimensions for a 2D kernel launch.
pub fn grid_2d(rows: usize, cols: usize, block_x: usize, block_y: usize) -> LaunchConfig {
    let grid_x = (cols + block_x - 1) / block_x;
    let grid_y = (rows + block_y - 1) / block_y;
    LaunchConfig {
        grid_dim: (grid_x as u32, grid_y as u32, 1),
        block_dim: (block_x as u32, block_y as u32, 1),
        shared_mem_bytes: 0,
    }
}

/// Compute grid dimensions for a 2D kernel with shared memory.
pub fn grid_2d_shared(
    rows: usize,
    cols: usize,
    block_x: usize,
    block_y: usize,
    shared_bytes: u32,
) -> LaunchConfig {
    let grid_x = (cols + block_x - 1) / block_x;
    let grid_y = (rows + block_y - 1) / block_y;
    LaunchConfig {
        grid_dim: (grid_x as u32, grid_y as u32, 1),
        block_dim: (block_x as u32, block_y as u32, 1),
        shared_mem_bytes: shared_bytes,
    }
}
