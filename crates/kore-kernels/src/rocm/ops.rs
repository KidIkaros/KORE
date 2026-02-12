//! ROCm kernel dispatch functions for Kore tensor operations.
//!
//! Mirrors `cuda/ops.rs` — shares the same kernel source files (`.cu`),
//! compiled via hiprtc instead of NVRTC. Uses `HipBuffer` instead of `CudaSlice<u8>`.

use std::ffi::c_void;

use super::context::RocmError;
use super::launch::{get_or_load_func, launch_kernel, HipLaunchConfig};
use super::memory::HipBuffer;

// ============================================================================
// Kernel sources (shared with CUDA — same .cu files)
// ============================================================================

const MAMBA_SCAN_CU: &str = include_str!("../cuda/kernels/mamba_scan.cu");
const MAMBA_SCAN_BWD_CU: &str = include_str!("../cuda/kernels/mamba_scan_backward.cu");

const SCAN_CHUNK: usize = 128;
const MAX_DSTATE: usize = 64;

// ============================================================================
// Dummy buffer for optional kernel parameters
// ============================================================================

/// Per-device cached 4-byte dummy buffer. Used as a valid device pointer for
/// optional kernel parameters gated by `has_*` u32 flags.
fn get_dummy_buf(device_idx: usize) -> Result<HipBuffer, RocmError> {
    HipBuffer::zeros(device_idx, 4)
}

// ============================================================================
// Mamba-3 Flash Scan (forward)
// ============================================================================

/// Output from the ROCm forward scan.
pub struct RocmScanForwardOut {
    pub output: HipBuffer,
    pub chunk_last_h: HipBuffer,
    pub chunk_last_bx: HipBuffer,
    pub h_all: Option<HipBuffer>,
    pub bx_all: Option<HipBuffer>,
}

/// Mamba-3 chunked parallel scan on ROCm GPU.
///
/// Mirrors `cuda_mamba3_scan_f32` exactly — same kernel source, same thread model.
#[allow(clippy::too_many_arguments)]
pub fn rocm_mamba3_scan_f32(
    device_idx: usize,
    x: &HipBuffer,
    dt: &HipBuffer,
    a_real: &HipBuffer,
    a_imag: &HipBuffer,
    b: &HipBuffer,
    c: &HipBuffer,
    dt_bias: Option<&HipBuffer>,
    z: Option<&HipBuffer>,
    d_skip: Option<&HipBuffer>,
    batch: usize,
    seq_len: usize,
    nheads: usize,
    headdim: usize,
    ngroups: usize,
    d_state: usize,
    alpha: f32,
    dt_softplus: bool,
    use_rope: bool,
    save_states: bool,
) -> Result<RocmScanForwardOut, RocmError> {
    assert!(d_state <= MAX_DSTATE,
        "rocm_mamba3_scan_f32: d_state={} exceeds MAX_DSTATE={}", d_state, MAX_DSTATE);

    let num_chunks = (seq_len + SCAN_CHUNK - 1) / SCAN_CHUNK;
    let state_size = d_state * headdim;
    let frame = nheads * d_state * headdim;
    let nf = seq_len + 1;

    // Allocate outputs
    let output = HipBuffer::zeros(device_idx, batch * seq_len * nheads * headdim * 4)?;
    let chunk_last_h = HipBuffer::zeros(device_idx, batch * nheads * num_chunks * state_size * 4)?;
    let chunk_last_bx = HipBuffer::zeros(device_idx, batch * nheads * num_chunks * state_size * 4)?;

    let h_all_buf = if save_states {
        Some(HipBuffer::zeros(device_idx, batch * nf * frame * 4)?)
    } else {
        None
    };
    let bx_all_buf = if save_states {
        Some(HipBuffer::zeros(device_idx, batch * nf * frame * 4)?)
    } else {
        None
    };

    let dummy = get_dummy_buf(device_idx)?;

    let has_dt_bias: u32 = dt_bias.is_some() as u32;
    let has_z: u32 = z.is_some() as u32;
    let has_d_skip: u32 = d_skip.is_some() as u32;
    let save_flag: u32 = save_states as u32;

    let dt_bias_ptr = dt_bias.unwrap_or(&dummy).as_device_ptr();
    let z_ptr = z.unwrap_or(&dummy).as_device_ptr();
    let d_skip_ptr = d_skip.unwrap_or(&dummy).as_device_ptr();
    let h_all_ptr = h_all_buf.as_ref().unwrap_or(&dummy).as_device_ptr();
    let bx_all_ptr = bx_all_buf.as_ref().unwrap_or(&dummy).as_device_ptr();

    let block_size = headdim.min(256) as u32;

    // Phase 1: Intra-chunk scan
    let f1 = get_or_load_func(
        device_idx, "mamba_scan", "mamba3_scan_chunk_f32", MAMBA_SCAN_CU,
    )?;
    let total_blocks = (batch * nheads * num_chunks) as u32;

    // Build kernel parameter pointers
    let mut x_ptr = x.as_device_ptr();
    let mut dt_p = dt.as_device_ptr();
    let mut ar_p = a_real.as_device_ptr();
    let mut ai_p = a_imag.as_device_ptr();
    let mut b_p = b.as_device_ptr();
    let mut c_p = c.as_device_ptr();
    let mut hdb = has_dt_bias;
    let mut dtb_p = dt_bias_ptr;
    let mut hz = has_z;
    let mut z_p = z_ptr;
    let mut hds = has_d_skip;
    let mut ds_p = d_skip_ptr;
    let mut out_p = output.as_device_ptr();
    let mut clh_p = chunk_last_h.as_device_ptr();
    let mut clb_p = chunk_last_bx.as_device_ptr();
    let mut sf = save_flag;
    let mut ha_p = h_all_ptr;
    let mut ba_p = bx_all_ptr;
    let mut bat = batch as u32;
    let mut sl = seq_len as u32;
    let mut nh = nheads as u32;
    let mut hd = headdim as u32;
    let mut ng = ngroups as u32;
    let mut ds = d_state as u32;
    let mut nc = num_chunks as u32;
    let mut al = alpha;
    let mut dsp = dt_softplus as u32;
    let mut ur = use_rope as u32;

    let mut params: Vec<*mut c_void> = vec![
        &mut x_ptr as *mut _ as *mut c_void,
        &mut dt_p as *mut _ as *mut c_void,
        &mut ar_p as *mut _ as *mut c_void,
        &mut ai_p as *mut _ as *mut c_void,
        &mut b_p as *mut _ as *mut c_void,
        &mut c_p as *mut _ as *mut c_void,
        &mut hdb as *mut _ as *mut c_void,
        &mut dtb_p as *mut _ as *mut c_void,
        &mut hz as *mut _ as *mut c_void,
        &mut z_p as *mut _ as *mut c_void,
        &mut hds as *mut _ as *mut c_void,
        &mut ds_p as *mut _ as *mut c_void,
        &mut out_p as *mut _ as *mut c_void,
        &mut clh_p as *mut _ as *mut c_void,
        &mut clb_p as *mut _ as *mut c_void,
        &mut sf as *mut _ as *mut c_void,
        &mut ha_p as *mut _ as *mut c_void,
        &mut ba_p as *mut _ as *mut c_void,
        &mut bat as *mut _ as *mut c_void,
        &mut sl as *mut _ as *mut c_void,
        &mut nh as *mut _ as *mut c_void,
        &mut hd as *mut _ as *mut c_void,
        &mut ng as *mut _ as *mut c_void,
        &mut ds as *mut _ as *mut c_void,
        &mut nc as *mut _ as *mut c_void,
        &mut al as *mut _ as *mut c_void,
        &mut dsp as *mut _ as *mut c_void,
        &mut ur as *mut _ as *mut c_void,
    ];

    let cfg1 = HipLaunchConfig {
        grid_dim: (total_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { launch_kernel(device_idx, f1, &cfg1, &mut params)?; }

    // Phase 2: Inter-chunk prefix scan (only if multiple chunks)
    if num_chunks > 1 {
        let f2 = get_or_load_func(
            device_idx, "mamba_scan", "mamba3_scan_prefix_f32", MAMBA_SCAN_CU,
        )?;
        let prefix_blocks = (batch * nheads) as u32;
        let mut ss = state_size as u32;

        let mut params2: Vec<*mut c_void> = vec![
            &mut clh_p as *mut _ as *mut c_void,
            &mut clb_p as *mut _ as *mut c_void,
            &mut ar_p as *mut _ as *mut c_void,
            &mut dt_p as *mut _ as *mut c_void,
            &mut hdb as *mut _ as *mut c_void,
            &mut dtb_p as *mut _ as *mut c_void,
            &mut bat as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut ss as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut al as *mut _ as *mut c_void,
            &mut dsp as *mut _ as *mut c_void,
        ];

        let cfg2 = HipLaunchConfig {
            grid_dim: (prefix_blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { launch_kernel(device_idx, f2, &cfg2, &mut params2)?; }
    }

    Ok(RocmScanForwardOut {
        output,
        chunk_last_h,
        chunk_last_bx,
        h_all: h_all_buf,
        bx_all: bx_all_buf,
    })
}

// ============================================================================
// Mamba-3 scan backward pass
// ============================================================================

/// Backward pass for the Mamba-3 chunked scan on ROCm GPU.
///
/// Mirrors `cuda_mamba3_scan_backward_f32` exactly — same kernel source.
/// Returns `(dx, d_dt, d_b, d_c, dz)`.
#[allow(clippy::too_many_arguments)]
pub fn rocm_mamba3_scan_backward_f32(
    device_idx: usize,
    grad_output: &HipBuffer,
    x: &HipBuffer,
    dt: &HipBuffer,
    a_real: &HipBuffer,
    a_imag: &HipBuffer,
    b: &HipBuffer,
    c: &HipBuffer,
    h_all: &HipBuffer,
    bx_all: &HipBuffer,
    dt_bias: Option<&HipBuffer>,
    z: Option<&HipBuffer>,
    d_skip: Option<&HipBuffer>,
    batch: usize,
    seq_len: usize,
    nheads: usize,
    headdim: usize,
    ngroups: usize,
    d_state: usize,
    alpha: f32,
    dt_softplus: bool,
    use_rope: bool,
) -> Result<(HipBuffer, HipBuffer, HipBuffer, HipBuffer, HipBuffer), RocmError> {
    assert!(d_state <= MAX_DSTATE,
        "rocm_mamba3_scan_backward_f32: d_state={} exceeds MAX_DSTATE={}", d_state, MAX_DSTATE);

    // Allocate gradient output buffers (zeroed — atomicAdd accumulates into these)
    let dx = HipBuffer::zeros(device_idx, batch * seq_len * nheads * headdim * 4)?;
    let d_dt = HipBuffer::zeros(device_idx, batch * seq_len * nheads * 4)?;
    let d_b = HipBuffer::zeros(device_idx, batch * seq_len * ngroups * d_state * 4)?;
    let d_c = HipBuffer::zeros(device_idx, batch * seq_len * ngroups * d_state * 4)?;
    let dz = HipBuffer::zeros(device_idx, batch * seq_len * nheads * headdim * 4)?;

    let dummy = get_dummy_buf(device_idx)?;

    let has_dt_bias: u32 = dt_bias.is_some() as u32;
    let has_z: u32 = z.is_some() as u32;
    let has_d_skip: u32 = d_skip.is_some() as u32;

    let dt_bias_ptr = dt_bias.unwrap_or(&dummy).as_device_ptr();
    let z_ptr = z.unwrap_or(&dummy).as_device_ptr();
    let d_skip_ptr = d_skip.unwrap_or(&dummy).as_device_ptr();

    let block_size = headdim.min(256) as u32;
    let total_blocks = (batch * nheads) as u32;

    let f = get_or_load_func(
        device_idx, "mamba_scan_backward", "mamba3_scan_backward_f32", MAMBA_SCAN_BWD_CU,
    )?;

    // Build kernel parameter pointers
    let mut go_p = grad_output.as_device_ptr();
    let mut x_p = x.as_device_ptr();
    let mut dt_p = dt.as_device_ptr();
    let mut ar_p = a_real.as_device_ptr();
    let mut ai_p = a_imag.as_device_ptr();
    let mut b_p = b.as_device_ptr();
    let mut c_p = c.as_device_ptr();
    let mut ha_p = h_all.as_device_ptr();
    let mut ba_p = bx_all.as_device_ptr();
    let mut hdb = has_dt_bias;
    let mut dtb_p = dt_bias_ptr;
    let mut hz = has_z;
    let mut zp = z_ptr;
    let mut hds = has_d_skip;
    let mut dsp_p = d_skip_ptr;
    let mut dx_p = dx.as_device_ptr();
    let mut ddt_p = d_dt.as_device_ptr();
    let mut db_p = d_b.as_device_ptr();
    let mut dc_p = d_c.as_device_ptr();
    let mut dz_p = dz.as_device_ptr();
    let mut bat = batch as u32;
    let mut sl = seq_len as u32;
    let mut nh = nheads as u32;
    let mut hd = headdim as u32;
    let mut ng = ngroups as u32;
    let mut ds = d_state as u32;
    let mut al = alpha;
    let mut dsp = dt_softplus as u32;
    let mut ur = use_rope as u32;

    let mut params: Vec<*mut c_void> = vec![
        &mut go_p as *mut _ as *mut c_void,
        &mut x_p as *mut _ as *mut c_void,
        &mut dt_p as *mut _ as *mut c_void,
        &mut ar_p as *mut _ as *mut c_void,
        &mut ai_p as *mut _ as *mut c_void,
        &mut b_p as *mut _ as *mut c_void,
        &mut c_p as *mut _ as *mut c_void,
        &mut ha_p as *mut _ as *mut c_void,
        &mut ba_p as *mut _ as *mut c_void,
        &mut hdb as *mut _ as *mut c_void,
        &mut dtb_p as *mut _ as *mut c_void,
        &mut hz as *mut _ as *mut c_void,
        &mut zp as *mut _ as *mut c_void,
        &mut hds as *mut _ as *mut c_void,
        &mut dsp_p as *mut _ as *mut c_void,
        &mut dx_p as *mut _ as *mut c_void,
        &mut ddt_p as *mut _ as *mut c_void,
        &mut db_p as *mut _ as *mut c_void,
        &mut dc_p as *mut _ as *mut c_void,
        &mut dz_p as *mut _ as *mut c_void,
        &mut bat as *mut _ as *mut c_void,
        &mut sl as *mut _ as *mut c_void,
        &mut nh as *mut _ as *mut c_void,
        &mut hd as *mut _ as *mut c_void,
        &mut ng as *mut _ as *mut c_void,
        &mut ds as *mut _ as *mut c_void,
        &mut al as *mut _ as *mut c_void,
        &mut dsp as *mut _ as *mut c_void,
        &mut ur as *mut _ as *mut c_void,
    ];

    let cfg = HipLaunchConfig {
        grid_dim: (total_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { launch_kernel(device_idx, f, &cfg, &mut params)?; }

    Ok((dx, d_dt, d_b, d_c, dz))
}
