//! CUDA kernel dispatch functions for Kore tensor operations.
//!
//! Each function loads the relevant PTX module (compiled from .cu source at runtime),
//! allocates output GPU memory, launches the kernel, and returns the result buffer.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};

use super::context::CudaError;
use super::launch::{get_or_load_func, grid_1d};

// ============================================================================
// PTX sources (embedded at compile time)
// ============================================================================

const ELEMENTWISE_CU: &str = include_str!("kernels/elementwise.cu");
const MATMUL_CU: &str = include_str!("kernels/matmul.cu");
const REDUCE_CU: &str = include_str!("kernels/reduce.cu");
const SOFTMAX_CU: &str = include_str!("kernels/softmax.cu");
const RMS_NORM_CU: &str = include_str!("kernels/rms_norm.cu");
const ROPE_CU: &str = include_str!("kernels/rope.cu");

const BLOCK_SIZE: usize = 256;

// ============================================================================
// Element-wise binary ops
// ============================================================================

fn binary_op(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    n: usize,
    func_name: &str,
) -> Result<CudaSlice<u8>, CudaError> {
    let f = get_or_load_func(dev, dev_idx, "elementwise", func_name, ELEMENTWISE_CU)?;
    let out = dev
        .alloc_zeros::<u8>(n * 4)
        .map_err(|e| CudaError::MemoryError(e.to_string()))?;
    let cfg = grid_1d(n, BLOCK_SIZE);
    unsafe {
        f.launch(cfg, (a, b, &out, n as u32))
            .map_err(|e| CudaError::LaunchError(e.to_string()))?;
    }
    Ok(out)
}

pub fn cuda_add_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    n: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    binary_op(dev, dev_idx, a, b, n, "add_f32")
}

pub fn cuda_sub_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    n: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    binary_op(dev, dev_idx, a, b, n, "sub_f32")
}

pub fn cuda_mul_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    n: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    binary_op(dev, dev_idx, a, b, n, "mul_f32")
}

pub fn cuda_div_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    n: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    binary_op(dev, dev_idx, a, b, n, "div_f32")
}

// ============================================================================
// Element-wise unary ops
// ============================================================================

fn unary_op(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    n: usize,
    func_name: &str,
) -> Result<CudaSlice<u8>, CudaError> {
    let f = get_or_load_func(dev, dev_idx, "elementwise", func_name, ELEMENTWISE_CU)?;
    let out = dev
        .alloc_zeros::<u8>(n * 4)
        .map_err(|e| CudaError::MemoryError(e.to_string()))?;
    let cfg = grid_1d(n, BLOCK_SIZE);
    unsafe {
        f.launch(cfg, (a, &out, n as u32))
            .map_err(|e| CudaError::LaunchError(e.to_string()))?;
    }
    Ok(out)
}

pub fn cuda_neg_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    n: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    unary_op(dev, dev_idx, a, n, "neg_f32")
}

pub fn cuda_abs_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    n: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    unary_op(dev, dev_idx, a, n, "abs_f32")
}

pub fn cuda_sqrt_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    n: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    unary_op(dev, dev_idx, a, n, "sqrt_f32")
}

pub fn cuda_exp_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    n: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    unary_op(dev, dev_idx, a, n, "exp_f32")
}

pub fn cuda_log_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    n: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    unary_op(dev, dev_idx, a, n, "log_f32")
}

pub fn cuda_relu_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    n: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    unary_op(dev, dev_idx, a, n, "relu_f32")
}

pub fn cuda_gelu_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    n: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    unary_op(dev, dev_idx, a, n, "gelu_f32")
}

pub fn cuda_silu_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    n: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    unary_op(dev, dev_idx, a, n, "silu_f32")
}

pub fn cuda_sigmoid_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    n: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    unary_op(dev, dev_idx, a, n, "sigmoid_f32")
}

pub fn cuda_tanh_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    n: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    unary_op(dev, dev_idx, a, n, "tanh_f32")
}

// ============================================================================
// Scalar ops
// ============================================================================

pub fn cuda_add_scalar_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    scalar: f32,
    n: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    let f = get_or_load_func(dev, dev_idx, "elementwise", "add_scalar_f32", ELEMENTWISE_CU)?;
    let out = dev
        .alloc_zeros::<u8>(n * 4)
        .map_err(|e| CudaError::MemoryError(e.to_string()))?;
    let cfg = grid_1d(n, BLOCK_SIZE);
    unsafe {
        f.launch(cfg, (a, scalar, &out, n as u32))
            .map_err(|e| CudaError::LaunchError(e.to_string()))?;
    }
    Ok(out)
}

pub fn cuda_mul_scalar_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    scalar: f32,
    n: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    let f = get_or_load_func(dev, dev_idx, "elementwise", "mul_scalar_f32", ELEMENTWISE_CU)?;
    let out = dev
        .alloc_zeros::<u8>(n * 4)
        .map_err(|e| CudaError::MemoryError(e.to_string()))?;
    let cfg = grid_1d(n, BLOCK_SIZE);
    unsafe {
        f.launch(cfg, (a, scalar, &out, n as u32))
            .map_err(|e| CudaError::LaunchError(e.to_string()))?;
    }
    Ok(out)
}

pub fn cuda_pow_scalar_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    exponent: f32,
    n: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    let f = get_or_load_func(dev, dev_idx, "elementwise", "pow_scalar_f32", ELEMENTWISE_CU)?;
    let out = dev
        .alloc_zeros::<u8>(n * 4)
        .map_err(|e| CudaError::MemoryError(e.to_string()))?;
    let cfg = grid_1d(n, BLOCK_SIZE);
    unsafe {
        f.launch(cfg, (a, exponent, &out, n as u32))
            .map_err(|e| CudaError::LaunchError(e.to_string()))?;
    }
    Ok(out)
}

pub fn cuda_clamp_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    lo: f32,
    hi: f32,
    n: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    let f = get_or_load_func(dev, dev_idx, "elementwise", "clamp_f32", ELEMENTWISE_CU)?;
    let out = dev
        .alloc_zeros::<u8>(n * 4)
        .map_err(|e| CudaError::MemoryError(e.to_string()))?;
    let cfg = grid_1d(n, BLOCK_SIZE);
    unsafe {
        f.launch(cfg, (a, lo, hi, &out, n as u32))
            .map_err(|e| CudaError::LaunchError(e.to_string()))?;
    }
    Ok(out)
}

// ============================================================================
// Matrix multiplication
// ============================================================================

/// CUDA tiled GEMM: C[M,N] = A[M,K] @ B[K,N]
pub fn cuda_matmul_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    let out = dev
        .alloc_zeros::<u8>(m * n * 4)
        .map_err(|e| CudaError::MemoryError(e.to_string()))?;

    // Choose kernel based on matrix size
    let (func_name, cfg) = if m >= 64 && n >= 64 {
        // Use 2x2 thread-tile kernel for larger matrices
        let block_x = 32u32;
        let block_y = 32u32;
        let grid_x = ((n + 63) / 64) as u32;
        let grid_y = ((m + 63) / 64) as u32;
        (
            "matmul_f32_tiled2x2",
            LaunchConfig {
                grid_dim: (grid_x, grid_y, 1),
                block_dim: (block_x, block_y, 1),
                shared_mem_bytes: 0,
            },
        )
    } else {
        // Use basic tiled kernel for smaller matrices
        let tile = 32;
        let grid_x = ((n + tile - 1) / tile) as u32;
        let grid_y = ((m + tile - 1) / tile) as u32;
        (
            "matmul_f32",
            LaunchConfig {
                grid_dim: (grid_x, grid_y, 1),
                block_dim: (tile as u32, tile as u32, 1),
                shared_mem_bytes: 0,
            },
        )
    };

    let f = get_or_load_func(dev, dev_idx, "matmul", func_name, MATMUL_CU)?;
    unsafe {
        f.launch(cfg, (a, b, &out, m as u32, n as u32, k as u32))
            .map_err(|e| CudaError::LaunchError(e.to_string()))?;
    }
    Ok(out)
}

// ============================================================================
// Reductions
// ============================================================================

/// Sum all elements. Returns a single-element GPU buffer.
pub fn cuda_reduce_sum_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    n: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    let f = get_or_load_func(dev, dev_idx, "reduce", "reduce_sum_f32", REDUCE_CU)?;
    let out = dev
        .alloc_zeros::<u8>(4)
        .map_err(|e| CudaError::MemoryError(e.to_string()))?;
    let cfg = grid_1d(n, BLOCK_SIZE);
    unsafe {
        f.launch(cfg, (a, &out, n as u32))
            .map_err(|e| CudaError::LaunchError(e.to_string()))?;
    }
    Ok(out)
}

/// Max of all elements. Returns a single-element GPU buffer.
pub fn cuda_reduce_max_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    n: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    let f = get_or_load_func(dev, dev_idx, "reduce", "reduce_max_f32", REDUCE_CU)?;
    // Initialize to -inf
    let neg_inf_bytes = f32::NEG_INFINITY.to_ne_bytes();
    let out = dev
        .htod_copy(neg_inf_bytes.to_vec())
        .map_err(|e| CudaError::MemoryError(e.to_string()))?;
    let cfg = grid_1d(n, BLOCK_SIZE);
    unsafe {
        f.launch(cfg, (a, &out, n as u32))
            .map_err(|e| CudaError::LaunchError(e.to_string()))?;
    }
    Ok(out)
}

/// Row-wise sum: output[row] = sum(input[row, :]). Returns [rows] GPU buffer.
pub fn cuda_reduce_sum_rows_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    rows: usize,
    cols: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    let f = get_or_load_func(dev, dev_idx, "reduce", "reduce_sum_rows_f32", REDUCE_CU)?;
    let out = dev
        .alloc_zeros::<u8>(rows * 4)
        .map_err(|e| CudaError::MemoryError(e.to_string()))?;
    let block = if cols <= 256 { 256 } else { 512 };
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        f.launch(cfg, (a, &out, rows as u32, cols as u32))
            .map_err(|e| CudaError::LaunchError(e.to_string()))?;
    }
    Ok(out)
}

/// Row-wise max: output[row] = max(input[row, :]). Returns [rows] GPU buffer.
pub fn cuda_reduce_max_rows_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    rows: usize,
    cols: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    let f = get_or_load_func(dev, dev_idx, "reduce", "reduce_max_rows_f32", REDUCE_CU)?;
    let out = dev
        .alloc_zeros::<u8>(rows * 4)
        .map_err(|e| CudaError::MemoryError(e.to_string()))?;
    let block = if cols <= 256 { 256 } else { 512 };
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        f.launch(cfg, (a, &out, rows as u32, cols as u32))
            .map_err(|e| CudaError::LaunchError(e.to_string()))?;
    }
    Ok(out)
}

// ============================================================================
// Softmax
// ============================================================================

/// Row-wise softmax: output[row,:] = softmax(input[row,:])
pub fn cuda_softmax_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a: &CudaSlice<u8>,
    rows: usize,
    cols: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    let f = get_or_load_func(dev, dev_idx, "softmax", "softmax_f32", SOFTMAX_CU)?;
    let out = dev
        .alloc_zeros::<u8>(rows * cols * 4)
        .map_err(|e| CudaError::MemoryError(e.to_string()))?;
    // One block per row, 256 threads per block
    let block = if cols <= 256 { 256 } else { 512 };
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        f.launch(cfg, (a, &out, rows as u32, cols as u32))
            .map_err(|e| CudaError::LaunchError(e.to_string()))?;
    }
    Ok(out)
}

// ============================================================================
// RMS Norm
// ============================================================================

/// RMS normalization: output[row,:] = (input[row,:] / rms) * weight[:]
pub fn cuda_rms_norm_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    input: &CudaSlice<u8>,
    weight: &CudaSlice<u8>,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<CudaSlice<u8>, CudaError> {
    let f = get_or_load_func(dev, dev_idx, "rms_norm", "rms_norm_f32", RMS_NORM_CU)?;
    let out = dev
        .alloc_zeros::<u8>(rows * cols * 4)
        .map_err(|e| CudaError::MemoryError(e.to_string()))?;
    let block = if cols <= 256 { 256 } else { 512 };
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        f.launch(cfg, (input, weight, &out, rows as u32, cols as u32, eps))
            .map_err(|e| CudaError::LaunchError(e.to_string()))?;
    }
    Ok(out)
}

// ============================================================================
// Rotary Position Embeddings (RoPE)
// ============================================================================

/// Precompute RoPE frequency table on GPU.
/// Returns freqs buffer of shape [seq_len, half_dim].
pub fn cuda_rope_freqs_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    seq_len: usize,
    half_dim: usize,
    theta: f32,
) -> Result<CudaSlice<u8>, CudaError> {
    let f = get_or_load_func(dev, dev_idx, "rope", "rope_freqs_f32", ROPE_CU)?;
    let total = seq_len * half_dim;
    let out = dev
        .alloc_zeros::<u8>(total * 4)
        .map_err(|e| CudaError::MemoryError(e.to_string()))?;
    let cfg = grid_1d(total, BLOCK_SIZE);
    unsafe {
        f.launch(cfg, (&out, seq_len as u32, half_dim as u32, theta))
            .map_err(|e| CudaError::LaunchError(e.to_string()))?;
    }
    Ok(out)
}

/// Apply RoPE in-place on GPU.
/// data: [seq_len, num_heads, head_dim] (contiguous, f32)
/// freqs: [seq_len, head_dim/2] (precomputed)
pub fn cuda_rope_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    data: &CudaSlice<u8>,
    freqs: &CudaSlice<u8>,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<(), CudaError> {
    let f = get_or_load_func(dev, dev_idx, "rope", "rope_f32", ROPE_CU)?;
    let half_dim = head_dim / 2;
    let total = seq_len * num_heads * half_dim;
    let cfg = grid_1d(total, BLOCK_SIZE);
    unsafe {
        f.launch(cfg, (data, freqs, seq_len as u32, num_heads as u32, head_dim as u32))
            .map_err(|e| CudaError::LaunchError(e.to_string()))?;
    }
    Ok(())
}
