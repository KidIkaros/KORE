//! CUDA kernel dispatch functions for Kore tensor operations.
//!
//! Each function loads the relevant PTX module (compiled from .cu source at runtime),
//! allocates output GPU memory, launches the kernel, and returns the result buffer.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use parking_lot::Mutex;

use super::context::CudaError;
use super::launch::{get_or_load_func, grid_1d};

// ============================================================================
// Cached dummy buffer for optional kernel parameters
// ============================================================================

/// Per-device cached 4-byte dummy buffer. Used as a valid device pointer for
/// optional kernel parameters that are gated by a `has_*` u32 flag — the
/// pointer is never dereferenced when the flag is 0. Cached to avoid
/// allocating a new buffer on every kernel dispatch call.
static DUMMY_BUFS: OnceLock<Mutex<HashMap<usize, Arc<CudaSlice<u8>>>>> = OnceLock::new();

fn get_dummy_buf(dev: &Arc<CudaDevice>, dev_idx: usize) -> Result<Arc<CudaSlice<u8>>, CudaError> {
    let map_mu = DUMMY_BUFS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = map_mu.lock();
    if let Some(buf) = map.get(&dev_idx) {
        return Ok(Arc::clone(buf));
    }
    let buf = Arc::new(
        dev.alloc_zeros::<u8>(4)
            .map_err(|e| CudaError::MemoryError(e.to_string()))?,
    );
    map.insert(dev_idx, Arc::clone(&buf));
    Ok(buf)
}

// ============================================================================
// PTX sources (embedded at compile time)
// ============================================================================

const ELEMENTWISE_CU: &str = include_str!("kernels/elementwise.cu");
const MATMUL_CU: &str = include_str!("kernels/matmul.cu");
const REDUCE_CU: &str = include_str!("kernels/reduce.cu");
const SOFTMAX_CU: &str = include_str!("kernels/softmax.cu");
const RMS_NORM_CU: &str = include_str!("kernels/rms_norm.cu");
const ROPE_CU: &str = include_str!("kernels/rope.cu");
const FUSED_NORM_PROJ_CU: &str = include_str!("kernels/fused_norm_proj.cu");
const DEQUANT_MATMUL_CU: &str = include_str!("kernels/dequant_matmul.cu");
const MAMBA_SCAN_CU: &str = include_str!("kernels/mamba_scan.cu");

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

// ============================================================================
// Fused RMSNorm + Linear Projection
// ============================================================================

/// Fused RMSNorm + Linear: y[row,:] = RMSNorm(x[row,:]) @ W^T + bias
///
/// Eliminates intermediate VRAM write of normalized activations.
/// Uses tiled approach: one block per (row, output_tile).
///
/// - `input`: [rows, hidden] f32
/// - `gamma`: [hidden] f32 RMSNorm scale
/// - `weight`: [out_dim, hidden] f32 projection weight
/// - `bias`: optional [out_dim] f32 bias (pass None for no bias)
pub fn cuda_fused_rms_norm_proj_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    input: &CudaSlice<u8>,
    gamma: &CudaSlice<u8>,
    weight: &CudaSlice<u8>,
    bias: Option<&CudaSlice<u8>>,
    rows: usize,
    hidden: usize,
    out_dim: usize,
    eps: f32,
) -> Result<CudaSlice<u8>, CudaError> {
    let out = dev
        .alloc_zeros::<u8>(rows * out_dim * 4)
        .map_err(|e| CudaError::MemoryError(e.to_string()))?;

    let dummy = get_dummy_buf(dev, dev_idx)?;
    let has_bias: u32 = bias.is_some() as u32;
    let bias_ptr = bias.unwrap_or(&dummy);

    // Choose kernel variant based on hidden size.
    // smem variant caches x_hat in shared memory, avoiding redundant recomputation
    // per output column. The 8192 threshold corresponds to 32KB of shared memory
    // (8192 * 4 bytes) which fits comfortably on SM 7.0+. For hidden > 8192,
    // the tiled variant recomputes x_hat per tile — still faster than two
    // separate kernel launches due to eliminated VRAM round-trip.
    if hidden <= 8192 {
        let f = get_or_load_func(
            dev, dev_idx, "fused_norm_proj", "fused_rms_norm_proj_smem_f32",
            FUSED_NORM_PROJ_CU,
        )?;
        let block = BLOCK_SIZE;
        let shared_bytes = (hidden * 4) as u32;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block as u32, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        unsafe {
            f.launch(cfg, (input, gamma, weight, has_bias, bias_ptr, &out,
                rows as u32, hidden as u32, out_dim as u32, eps))
                .map_err(|e| CudaError::LaunchError(e.to_string()))?;
        }
    } else {
        let f = get_or_load_func(
            dev, dev_idx, "fused_norm_proj", "fused_rms_norm_proj_f32",
            FUSED_NORM_PROJ_CU,
        )?;
        let grid_y = (out_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, grid_y as u32, 1),
            block_dim: (BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            f.launch(cfg, (input, gamma, weight, has_bias, bias_ptr, &out,
                rows as u32, hidden as u32, out_dim as u32, eps))
                .map_err(|e| CudaError::LaunchError(e.to_string()))?;
        }
    }

    Ok(out)
}

/// Fused RMSNorm + SiLU gate: y = RMSNorm(x) * SiLU(z)
///
/// Common in Mamba input paths where norm output is gated.
///
/// - `input`: [rows, cols] f32
/// - `gamma`: [cols] f32 RMSNorm scale
/// - `gate`: [rows, cols] f32 gate tensor (z)
pub fn cuda_fused_rms_norm_silu_gate_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    input: &CudaSlice<u8>,
    gamma: &CudaSlice<u8>,
    gate: &CudaSlice<u8>,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<CudaSlice<u8>, CudaError> {
    let f = get_or_load_func(
        dev, dev_idx, "fused_norm_proj", "fused_rms_norm_silu_gate_f32",
        FUSED_NORM_PROJ_CU,
    )?;
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
        f.launch(cfg, (input, gamma, gate, &out, rows as u32, cols as u32, eps))
            .map_err(|e| CudaError::LaunchError(e.to_string()))?;
    }
    Ok(out)
}

// ============================================================================
// Dequantized Quaternary MatMul
// ============================================================================

/// On-the-fly dequantized quaternary GEMM: C[M,N] = dequant(A_packed) @ B
///
/// Unpacks 2-bit quaternary weights ({-3,-1,+1,+3}) from packed bytes in
/// registers — never materializes full f32 weight matrix in VRAM.
/// 16x memory bandwidth reduction vs f32 weights.
///
/// - `a_packed`: [M, K_packed] packed uint8 (4 quats per byte, kore-btes format)
/// - `a_scales`: [M] per-row f32 scale factors
/// - `b`: [K, N] dense f32 activations
/// - `m`, `n`, `k`: logical dimensions
pub fn cuda_dequant_quat_matmul_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a_packed: &CudaSlice<u8>,
    a_scales: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    let k_packed = (k + 3) / 4;
    let out = dev
        .alloc_zeros::<u8>(m * n * 4)
        .map_err(|e| CudaError::MemoryError(e.to_string()))?;

    let (func_name, cfg) = if m >= 64 && n >= 64 {
        let grid_x = ((n + 63) / 64) as u32;
        let grid_y = ((m + 63) / 64) as u32;
        (
            "dequant_quat_matmul_tiled2x2_f32",
            LaunchConfig {
                grid_dim: (grid_x, grid_y, 1),
                block_dim: (32, 32, 1),
                shared_mem_bytes: 0,
            },
        )
    } else {
        let tile = 32;
        let grid_x = ((n + tile - 1) / tile) as u32;
        let grid_y = ((m + tile - 1) / tile) as u32;
        (
            "dequant_quat_matmul_f32",
            LaunchConfig {
                grid_dim: (grid_x, grid_y, 1),
                block_dim: (tile as u32, tile as u32, 1),
                shared_mem_bytes: 0,
            },
        )
    };

    let f = get_or_load_func(dev, dev_idx, "dequant_matmul", func_name, DEQUANT_MATMUL_CU)?;
    unsafe {
        f.launch(cfg, (a_packed, a_scales, b, &out,
            m as u32, n as u32, k as u32, k_packed as u32))
            .map_err(|e| CudaError::LaunchError(e.to_string()))?;
    }
    Ok(out)
}

/// Fused dequant matmul + bias + ReLU: C = max(0, dequant(A) @ B + bias)
pub fn cuda_dequant_quat_matmul_bias_relu_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    a_packed: &CudaSlice<u8>,
    a_scales: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    bias: &CudaSlice<u8>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaSlice<u8>, CudaError> {
    let k_packed = (k + 3) / 4;
    let out = dev
        .alloc_zeros::<u8>(m * n * 4)
        .map_err(|e| CudaError::MemoryError(e.to_string()))?;

    let tile = 32;
    let grid_x = ((n + tile - 1) / tile) as u32;
    let grid_y = ((m + tile - 1) / tile) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (tile as u32, tile as u32, 1),
        shared_mem_bytes: 0,
    };

    let f = get_or_load_func(
        dev, dev_idx, "dequant_matmul", "dequant_quat_matmul_bias_relu_f32",
        DEQUANT_MATMUL_CU,
    )?;
    unsafe {
        f.launch(cfg, (a_packed, a_scales, b, bias, &out,
            m as u32, n as u32, k as u32, k_packed as u32))
            .map_err(|e| CudaError::LaunchError(e.to_string()))?;
    }
    Ok(out)
}

// ============================================================================
// Mamba-3 Flash Scan (Chunked Parallel SSM)
// ============================================================================

/// Chunked size for the flash scan (must match SCAN_CHUNK in mamba_scan.cu).
const SCAN_CHUNK: usize = 128;

/// Maximum supported d_state (must match MAX_DSTATE in mamba_scan.cu).
const MAX_DSTATE: usize = 64;

/// Mamba-3 chunked parallel scan on GPU.
///
/// Splits the sequence into chunks of 128 timesteps, processes each chunk
/// in registers, then propagates boundary states across chunks.
///
/// Thread model: each thread owns specific headdim positions and processes
/// ALL d_state elements for those positions. No atomics, no inter-thread races.
///
/// - `x`: [batch, seq_len, nheads, headdim] f32
/// - `dt`: [batch, seq_len, nheads] f32
/// - `a_real`: [nheads] f32
/// - `b`: [batch, seq_len, ngroups, d_state] f32
/// - `c`: [batch, seq_len, ngroups, d_state] f32
/// - `dt_bias`: optional [nheads] f32
/// - `z`: optional [batch, seq_len, nheads, headdim] f32 gate
/// - `d_skip`: optional [nheads] f32 skip connection
///
/// Returns `(output, chunk_last_h, chunk_last_bx)`.
#[allow(clippy::too_many_arguments)]
pub fn cuda_mamba3_scan_f32(
    dev: &Arc<CudaDevice>,
    dev_idx: usize,
    x: &CudaSlice<u8>,
    dt: &CudaSlice<u8>,
    a_real: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    c: &CudaSlice<u8>,
    dt_bias: Option<&CudaSlice<u8>>,
    z: Option<&CudaSlice<u8>>,
    d_skip: Option<&CudaSlice<u8>>,
    batch: usize,
    seq_len: usize,
    nheads: usize,
    headdim: usize,
    ngroups: usize,
    d_state: usize,
    alpha: f32,
    dt_softplus: bool,
) -> Result<(CudaSlice<u8>, CudaSlice<u8>, CudaSlice<u8>), CudaError> {
    assert!(d_state <= MAX_DSTATE,
        "cuda_mamba3_scan_f32: d_state={} exceeds MAX_DSTATE={}", d_state, MAX_DSTATE);

    let num_chunks = (seq_len + SCAN_CHUNK - 1) / SCAN_CHUNK;
    let state_size = d_state * headdim;

    // Allocate outputs
    let output = dev
        .alloc_zeros::<u8>(batch * seq_len * nheads * headdim * 4)
        .map_err(|e| CudaError::MemoryError(e.to_string()))?;
    let chunk_last_h = dev
        .alloc_zeros::<u8>(batch * nheads * num_chunks * state_size * 4)
        .map_err(|e| CudaError::MemoryError(e.to_string()))?;
    let chunk_last_bx = dev
        .alloc_zeros::<u8>(batch * nheads * num_chunks * state_size * 4)
        .map_err(|e| CudaError::MemoryError(e.to_string()))?;

    let has_dt_bias: u32 = dt_bias.is_some() as u32;
    let has_z: u32 = z.is_some() as u32;
    let has_d_skip: u32 = d_skip.is_some() as u32;

    let dummy = get_dummy_buf(dev, dev_idx)?;
    let dt_bias_ptr = dt_bias.unwrap_or(&dummy);
    let z_ptr = z.unwrap_or(&dummy);
    let d_skip_ptr = d_skip.unwrap_or(&dummy);

    // Block size: one thread per headdim position (capped at 256).
    // Smaller block if headdim is small to avoid idle threads.
    let block_size = headdim.min(256) as u32;

    // Phase 1: Intra-chunk scan
    let f1 = get_or_load_func(
        dev, dev_idx, "mamba_scan", "mamba3_scan_chunk_f32", MAMBA_SCAN_CU,
    )?;
    let total_blocks = batch * nheads * num_chunks;
    let cfg1 = LaunchConfig {
        grid_dim: (total_blocks as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        f1.launch(cfg1, (
            x, dt, a_real, b, c,
            has_dt_bias, dt_bias_ptr,
            has_z, z_ptr,
            has_d_skip, d_skip_ptr,
            &output, &chunk_last_h, &chunk_last_bx,
            batch as u32, seq_len as u32, nheads as u32, headdim as u32,
            ngroups as u32, d_state as u32, num_chunks as u32,
            alpha, dt_softplus as u32,
        )).map_err(|e| CudaError::LaunchError(e.to_string()))?;
    }

    // Phase 2: Inter-chunk prefix scan (only if multiple chunks)
    if num_chunks > 1 {
        let f2 = get_or_load_func(
            dev, dev_idx, "mamba_scan", "mamba3_scan_prefix_f32", MAMBA_SCAN_CU,
        )?;
        let prefix_blocks = batch * nheads;
        let cfg2 = LaunchConfig {
            grid_dim: (prefix_blocks as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            f2.launch(cfg2, (
                &chunk_last_h, &chunk_last_bx, a_real, dt,
                has_dt_bias, dt_bias_ptr,
                batch as u32, seq_len as u32, nheads as u32,
                state_size as u32, num_chunks as u32,
                alpha, dt_softplus as u32,
            )).map_err(|e| CudaError::LaunchError(e.to_string()))?;
        }
    }

    Ok((output, chunk_last_h, chunk_last_bx))
}
