//! CUDA dispatch for tensor operations.
//!
//! Provides GPU-accelerated implementations of element-wise, matmul, and
//! reduction operations. Kernels are compiled from embedded CUDA source
//! at runtime via NVRTC and cached per device.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, CudaView, CudaViewMut, LaunchAsync, LaunchConfig};

use crate::storage::Storage;
use crate::tensor::Tensor;
use crate::{DType, Device, KoreError, Result};

// ============================================================================
// PTX module loading
// ============================================================================

/// Function names exported by each CUDA module.
const ELEMENTWISE_FUNCS: &[&str] = &[
    "add_f32", "sub_f32", "mul_f32", "div_f32",
    "add_scalar_f32", "mul_scalar_f32",
    "neg_f32", "abs_f32", "sqrt_f32", "exp_f32", "log_f32",
    "relu_f32", "gelu_f32", "silu_f32", "sigmoid_f32", "tanh_f32",
    "clamp_f32", "pow_scalar_f32",
];

const MATMUL_FUNCS: &[&str] = &[
    "matmul_f32", "matmul_f32_tiled2x2", "matmul_bias_relu_f32",
];

const REDUCE_FUNCS: &[&str] = &[
    "reduce_sum_f32", "reduce_max_f32",
    "reduce_sum_rows_f32", "reduce_max_rows_f32",
];

/// Get a CUDA kernel function, compiling and loading the module if needed.
/// Uses cudarc's internal module map: checks if function is already available,
/// compiles PTX only on first use.
fn get_func(
    dev: &Arc<CudaDevice>,
    module_name: &str,
    func_name: &str,
    cu_source: &str,
    func_names: &[&'static str],
) -> Result<cudarc::driver::CudaFunction> {
    // Fast path: function already loaded
    if let Some(f) = dev.get_func(module_name, func_name) {
        return Ok(f);
    }
    // Slow path: compile PTX and load module
    let ptx = cudarc::nvrtc::compile_ptx(cu_source)
        .map_err(|e| KoreError::CudaError(format!("PTX compile '{}': {}", module_name, e)))?;
    dev.load_ptx(ptx, module_name, func_names)
        .map_err(|e| KoreError::CudaError(format!("load module '{}': {}", module_name, e)))?;
    dev.get_func(module_name, func_name)
        .ok_or_else(|| KoreError::CudaError(format!("func '{}' not found in '{}'", func_name, module_name)))
}

// ============================================================================
// Embedded CUDA kernel sources
// ============================================================================

const ELEMENTWISE_CU: &str = include_str!("cuda_kernels/elementwise.cu");
const MATMUL_CU: &str = include_str!("cuda_kernels/matmul.cu");
#[allow(dead_code)]
const REDUCE_CU: &str = include_str!("cuda_kernels/reduce.cu");

const BLOCK_SIZE: usize = 256;

// ============================================================================
// Helpers
// ============================================================================

/// Extract (CudaDevice, device_idx, CudaSlice<u8>) from a GPU tensor's storage.
fn gpu_parts(t: &Tensor) -> Result<(Arc<CudaDevice>, usize, &CudaSlice<u8>)> {
    let dev = t.storage_ref().cuda_device()
        .ok_or_else(|| KoreError::CudaError("tensor not on GPU".into()))?;
    let idx = match t.device() {
        Device::Cuda(i) => i,
        _ => return Err(KoreError::CudaError("tensor not on GPU".into())),
    };
    let slice = t.storage_ref().as_cuda_slice()
        .ok_or_else(|| KoreError::CudaError("tensor not on GPU".into()))?;
    Ok((dev, idx, slice))
}

/// Reinterpret a CudaSlice<u8> as a CudaView<f32> (immutable) for kernel input.
/// Safety: caller must ensure the slice contains valid f32 data and numel*4 <= slice.len().
unsafe fn as_f32_view(slice: &CudaSlice<u8>, numel: usize) -> CudaView<'_, f32> {
    slice.transmute(numel).expect("f32 transmute failed")
}

/// Reinterpret a CudaSlice<u8> as a CudaViewMut<f32> (mutable) for kernel output.
/// Safety: caller must ensure numel*4 <= slice.len().
unsafe fn as_f32_view_mut(slice: &mut CudaSlice<u8>, numel: usize) -> CudaViewMut<'_, f32> {
    slice.transmute_mut(numel).expect("f32 transmute_mut failed")
}

/// Allocate a zeroed u8 buffer sized for `numel` f32 elements, returning the buffer.
fn alloc_f32_as_u8(dev: &Arc<CudaDevice>, numel: usize) -> Result<CudaSlice<u8>> {
    dev.alloc_zeros::<u8>(numel * 4)
        .map_err(|e| KoreError::CudaError(format!("alloc: {}", e)))
}

/// Create a new GPU tensor from a CudaSlice<u8> that holds f32 data.
fn tensor_from_gpu(
    dev: Arc<CudaDevice>,
    dev_idx: usize,
    buffer: CudaSlice<u8>,
    shape: &[usize],
) -> Tensor {
    let numel: usize = shape.iter().product();
    let storage = Storage::from_cuda(dev, buffer, dev_idx, DType::F32, numel);
    Tensor::from_storage(storage, shape)
}

fn grid_1d(n: usize, block: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (((n + block - 1) / block) as u32, 1, 1),
        block_dim: (block as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

// ============================================================================
// Binary element-wise ops
// ============================================================================

pub(crate) fn cuda_binary_op(
    a: &Tensor,
    b: &Tensor,
    func_name: &str,
) -> Result<Tensor> {
    let (dev, idx, a_slice) = gpu_parts(a)?;
    let (_, _, b_slice) = gpu_parts(b)?;
    let n = a.numel();

    let f = get_func(&dev, "elementwise", func_name, ELEMENTWISE_CU, ELEMENTWISE_FUNCS)?;
    let mut out = alloc_f32_as_u8(&dev, n)?;
    let cfg = grid_1d(n, BLOCK_SIZE);
    unsafe {
        let a_f32 = as_f32_view(a_slice, n);
        let b_f32 = as_f32_view(b_slice, n);
        let mut out_f32 = as_f32_view_mut(&mut out, n);
        f.launch(cfg, (&a_f32, &b_f32, &mut out_f32, n as u32))
            .map_err(|e| KoreError::CudaError(format!("launch {}: {}", func_name, e)))?;
    }
    Ok(tensor_from_gpu(dev, idx, out, a.shape().dims()))
}

// ============================================================================
// Unary element-wise ops
// ============================================================================

pub(crate) fn cuda_unary_op(
    a: &Tensor,
    func_name: &str,
) -> Result<Tensor> {
    let (dev, idx, a_slice) = gpu_parts(a)?;
    let n = a.numel();

    let f = get_func(&dev, "elementwise", func_name, ELEMENTWISE_CU, ELEMENTWISE_FUNCS)?;
    let mut out = alloc_f32_as_u8(&dev, n)?;
    let cfg = grid_1d(n, BLOCK_SIZE);
    unsafe {
        let a_f32 = as_f32_view(a_slice, n);
        let mut out_f32 = as_f32_view_mut(&mut out, n);
        f.launch(cfg, (&a_f32, &mut out_f32, n as u32))
            .map_err(|e| KoreError::CudaError(format!("launch {}: {}", func_name, e)))?;
    }
    Ok(tensor_from_gpu(dev, idx, out, a.shape().dims()))
}

// ============================================================================
// Scalar ops
// ============================================================================

pub(crate) fn cuda_scalar_op(
    a: &Tensor,
    scalar: f32,
    func_name: &str,
) -> Result<Tensor> {
    let (dev, idx, a_slice) = gpu_parts(a)?;
    let n = a.numel();

    let f = get_func(&dev, "elementwise", func_name, ELEMENTWISE_CU, ELEMENTWISE_FUNCS)?;
    let mut out = alloc_f32_as_u8(&dev, n)?;
    let cfg = grid_1d(n, BLOCK_SIZE);
    unsafe {
        let a_f32 = as_f32_view(a_slice, n);
        let mut out_f32 = as_f32_view_mut(&mut out, n);
        f.launch(cfg, (&a_f32, scalar, &mut out_f32, n as u32))
            .map_err(|e| KoreError::CudaError(format!("launch {}: {}", func_name, e)))?;
    }
    Ok(tensor_from_gpu(dev, idx, out, a.shape().dims()))
}

pub(crate) fn cuda_clamp(
    a: &Tensor,
    lo: f32,
    hi: f32,
) -> Result<Tensor> {
    let (dev, idx, a_slice) = gpu_parts(a)?;
    let n = a.numel();

    let f = get_func(&dev, "elementwise", "clamp_f32", ELEMENTWISE_CU, ELEMENTWISE_FUNCS)?;
    let mut out = alloc_f32_as_u8(&dev, n)?;
    let cfg = grid_1d(n, BLOCK_SIZE);
    unsafe {
        let a_f32 = as_f32_view(a_slice, n);
        let mut out_f32 = as_f32_view_mut(&mut out, n);
        f.launch(cfg, (&a_f32, lo, hi, &mut out_f32, n as u32))
            .map_err(|e| KoreError::CudaError(format!("launch clamp: {}", e)))?;
    }
    Ok(tensor_from_gpu(dev, idx, out, a.shape().dims()))
}

// ============================================================================
// Matrix multiplication
// ============================================================================

pub(crate) fn cuda_matmul_2d(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (dev, idx, a_slice) = gpu_parts(a)?;
    let (_, _, b_slice) = gpu_parts(b)?;

    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    let m = a_dims[0];
    let k = a_dims[1];
    let n = b_dims[1];

    let mut out = alloc_f32_as_u8(&dev, m * n)?;

    // Choose kernel variant based on size
    let (func_name, cfg) = if m >= 64 && n >= 64 {
        let grid_x = ((n + 63) / 64) as u32;
        let grid_y = ((m + 63) / 64) as u32;
        ("matmul_f32_tiled2x2", LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (32, 32, 1),
            shared_mem_bytes: 0,
        })
    } else {
        let tile = 32;
        let grid_x = ((n + tile - 1) / tile) as u32;
        let grid_y = ((m + tile - 1) / tile) as u32;
        ("matmul_f32", LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (tile as u32, tile as u32, 1),
            shared_mem_bytes: 0,
        })
    };

    let f = get_func(&dev, "matmul", func_name, MATMUL_CU, MATMUL_FUNCS)?;
    unsafe {
        let a_f32 = as_f32_view(a_slice, m * k);
        let b_f32 = as_f32_view(b_slice, k * n);
        let mut out_f32 = as_f32_view_mut(&mut out, m * n);
        f.launch(cfg, (&a_f32, &b_f32, &mut out_f32, m as u32, n as u32, k as u32))
            .map_err(|e| KoreError::CudaError(format!("launch {}: {}", func_name, e)))?;
    }
    Ok(tensor_from_gpu(dev, idx, out, &[m, n]))
}

// ============================================================================
// Reductions
// ============================================================================

#[allow(dead_code)]
pub(crate) fn cuda_sum(a: &Tensor) -> Result<Tensor> {
    let (dev, idx, a_slice) = gpu_parts(a)?;
    let n = a.numel();

    let f = get_func(&dev, "reduce", "reduce_sum_f32", REDUCE_CU, REDUCE_FUNCS)?;
    let mut out = alloc_f32_as_u8(&dev, 1)?;
    let cfg = grid_1d(n, BLOCK_SIZE);
    unsafe {
        let a_f32 = as_f32_view(a_slice, n);
        let mut out_f32 = as_f32_view_mut(&mut out, 1);
        f.launch(cfg, (&a_f32, &mut out_f32, n as u32))
            .map_err(|e| KoreError::CudaError(format!("launch reduce_sum: {}", e)))?;
    }
    Ok(tensor_from_gpu(dev, idx, out, &[]))
}
