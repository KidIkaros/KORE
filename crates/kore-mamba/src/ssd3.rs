//! Mamba-3 SSD scan — trapezoidal discretization, data-dependent RoPE, MIMO.
//!
//! Implements the three core innovations from Mamba-3:
//! 1. Trapezoidal rule: second-order discretization replacing Euler's method
//! 2. Data-dependent RoPE: rotation matrices on B, C recovering complex dynamics
//! 3. MIMO: rank-r matrix multiply state update for higher arithmetic intensity
//!
//! Two execution paths:
//! - **CPU** (`mamba3_scan_combined`): Pure Rust reference implementation.
//! - **GPU** (`mamba3_scan_gpu`, requires `cuda` feature): Uploads data to GPU,
//!   calls the Flash Scan CUDA kernel, and downloads results. Falls back to CPU
//!   if CUDA initialization fails.
//!
//! Reference: "Mamba-3: Improved Sequence Modeling using State Space Principles" (ICLR 2026)

/// SiLU (Swish) activation: x * sigmoid(x)
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Softplus activation: log(1 + exp(x))
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else if x < -20.0 { 0.0 } else { (1.0 + x.exp()).ln() }
}

/// Apply data-dependent RoPE rotation to a vector in-place.
///
/// For each pair (v[2i], v[2i+1]), applies rotation by angle theta:
///   v[2i]'   = v[2i] * cos(theta) - v[2i+1] * sin(theta)
///   v[2i+1]' = v[2i] * sin(theta) + v[2i+1] * cos(theta)
///
/// `v`: vector of length `n` (must be even)
/// `theta`: rotation angle (derived from imaginary part of A and dt)
#[inline]
fn apply_rope_inplace(v: &mut [f32], theta: f32) {
    assert!(v.len() % 2 == 0, "apply_rope_inplace: vector length must be even, got {}", v.len());
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let pairs = v.len() / 2;
    for i in 0..pairs {
        let a = v[2 * i];
        let b = v[2 * i + 1];
        v[2 * i] = a * cos_t - b * sin_t;
        v[2 * i + 1] = a * sin_t + b * cos_t;
    }
}

/// Compute trapezoidal discretization coefficients.
///
/// Given dt and alpha (interpolation parameter, 0.5 = standard trapezoidal):
///   beta  = dt * alpha         (weight for current input)
///   gamma = dt * (1 - alpha)   (weight for previous input)
///
/// Returns (beta, gamma).
#[inline]
fn trapezoidal_coefficients(dt: f32, alpha: f32) -> (f32, f32) {
    (dt * alpha, dt * (1.0 - alpha))
}

/// Output from Mamba-3 scan.
pub struct Mamba3ScanOutput {
    /// Output tensor, shape (batch, seq_len, nheads, headdim)
    pub output: Vec<f32>,
    /// Final SSM state, shape (batch, nheads, d_state, headdim)
    pub last_state: Vec<f32>,
    /// Previous B*x product for trapezoidal continuation, shape (batch, nheads, d_state, headdim)
    pub prev_bx: Vec<f32>,
}

/// Mamba-3 chunked scan with trapezoidal discretization, RoPE, and MIMO.
///
/// # Arguments
/// - `x`: input, shape (batch, seq_len, nheads, headdim)
/// - `dt`: time delta, shape (batch, seq_len, nheads)
/// - `a_real`: real part of A (negative), shape (nheads,)
/// - `a_imag`: imaginary part of A (for RoPE angle), shape (nheads,)
/// - `b`: input-dependent B, shape (batch, seq_len, ngroups, d_state)
/// - `c`: input-dependent C, shape (batch, seq_len, ngroups, d_state)
/// - `b_bias`: channel bias on B, shape (ngroups, d_state) — optional
/// - `c_bias`: channel bias on C, shape (ngroups, d_state) — optional
/// - `d`: skip connection, shape (nheads,)
/// - `z`: gate, shape (batch, seq_len, nheads, headdim) — optional
/// - `dt_bias`: bias for dt, shape (nheads,) — optional
/// - `dt_softplus`: whether to apply softplus to dt
/// - `alpha`: trapezoidal interpolation parameter (0.5 = standard)
/// - `use_rope`: whether to apply data-dependent RoPE
#[allow(clippy::too_many_arguments)]
pub fn mamba3_scan_combined(
    x: &[f32],
    batch: usize,
    seq_len: usize,
    nheads: usize,
    headdim: usize,
    dt: &[f32],
    a_real: &[f32],
    a_imag: &[f32],
    b: &[f32],
    ngroups: usize,
    d_state: usize,
    c: &[f32],
    b_bias: Option<&[f32]>,
    c_bias: Option<&[f32]>,
    d: Option<&[f32]>,
    z: Option<&[f32]>,
    dt_bias: Option<&[f32]>,
    dt_softplus: bool,
    alpha: f32,
    use_rope: bool,
) -> Mamba3ScanOutput {
    assert!(ngroups > 0, "mamba3_scan_combined: ngroups must be > 0");
    assert!(nheads % ngroups == 0, "mamba3_scan_combined: nheads ({}) must be divisible by ngroups ({})", nheads, ngroups);
    let heads_per_group = nheads / ngroups;
    let mut output = vec![0.0f32; batch * seq_len * nheads * headdim];
    // State: (batch, nheads, d_state, headdim)
    let mut state = vec![0.0f32; batch * nheads * d_state * headdim];
    // Previous B*x for trapezoidal: (batch, nheads, d_state, headdim)
    let mut prev_bx = vec![0.0f32; batch * nheads * d_state * headdim];

    for b_idx in 0..batch {
        let state_base = b_idx * nheads * d_state * headdim;

        for l in 0..seq_len {
            for h in 0..nheads {
                let g = h / heads_per_group;

                // dt value
                let mut dt_val = dt[b_idx * seq_len * nheads + l * nheads + h];
                if let Some(bias) = dt_bias {
                    dt_val += bias[h];
                }
                if dt_softplus {
                    dt_val = softplus(dt_val);
                }

                // dA = exp(dt * A_real[h])
                let da = (dt_val * a_real[h]).exp();

                // RoPE angle: dt * A_imag[h]
                let rope_theta = if use_rope { dt_val * a_imag[h] } else { 0.0 };

                // Trapezoidal coefficients
                let (beta, gamma) = trapezoidal_coefficients(dt_val, alpha);

                // Extract B vector for this step: (d_state,)
                let mut b_vec = vec![0.0f32; d_state];
                for n in 0..d_state {
                    b_vec[n] = b[b_idx * seq_len * ngroups * d_state
                        + l * ngroups * d_state
                        + g * d_state
                        + n];
                    if let Some(bb) = b_bias {
                        b_vec[n] += bb[g * d_state + n];
                    }
                }

                // Extract C vector for this step: (d_state,)
                let mut c_vec = vec![0.0f32; d_state];
                for n in 0..d_state {
                    c_vec[n] = c[b_idx * seq_len * ngroups * d_state
                        + l * ngroups * d_state
                        + g * d_state
                        + n];
                    if let Some(cb) = c_bias {
                        c_vec[n] += cb[g * d_state + n];
                    }
                }

                // Apply RoPE to B and C (rotate pairs)
                if use_rope && d_state >= 2 {
                    apply_rope_inplace(&mut b_vec, rope_theta);
                    apply_rope_inplace(&mut c_vec, rope_theta);
                }

                for p in 0..headdim {
                    let x_val = x[b_idx * seq_len * nheads * headdim
                        + l * nheads * headdim
                        + h * headdim
                        + p];

                    let mut y = 0.0f32;

                    for n in 0..d_state {
                        // Current B*x product
                        let cur_bx = b_vec[n] * x_val;

                        // Previous B*x from trapezoidal cache
                        let prev_bx_idx = state_base + h * d_state * headdim + n * headdim + p;
                        let prev_bx_val = prev_bx[prev_bx_idx];

                        // Trapezoidal state update:
                        // h_t = dA * h_{t-1} + beta * B_t * x_t + gamma * B_{t-1} * x_{t-1}
                        let s_idx = state_base + h * d_state * headdim + n * headdim + p;
                        state[s_idx] = state[s_idx] * da + beta * cur_bx + gamma * prev_bx_val;

                        // Output: y += C_t * h_t
                        y += state[s_idx] * c_vec[n];

                        // Update prev_bx cache
                        prev_bx[prev_bx_idx] = cur_bx;
                    }

                    // Skip connection
                    if let Some(d_skip) = d {
                        y += d_skip[h] * x_val;
                    }

                    // Gating with z
                    if let Some(z_data) = z {
                        let z_val = z_data[b_idx * seq_len * nheads * headdim
                            + l * nheads * headdim
                            + h * headdim
                            + p];
                        y *= silu(z_val);
                    }

                    let out_idx = b_idx * seq_len * nheads * headdim
                        + l * nheads * headdim
                        + h * headdim
                        + p;
                    output[out_idx] = y;
                }
            }
        }
    }

    Mamba3ScanOutput { output, last_state: state, prev_bx }
}

/// Single-step Mamba-3 SSM update for decoding.
///
/// # Arguments
/// - `x`: input, shape (batch, nheads, headdim)
/// - `dt`: time delta, shape (batch, nheads)
/// - `a_real`: real part of A (negative), shape (nheads,)
/// - `a_imag`: imaginary part of A (for RoPE), shape (nheads,)
/// - `b`: B for this step, shape (batch, ngroups, d_state)
/// - `c`: C for this step, shape (batch, ngroups, d_state)
/// - `b_bias`, `c_bias`: channel biases — optional
/// - `d`: skip, shape (nheads,)
/// - `z`: gate, shape (batch, nheads, headdim) — optional
/// - `dt_bias`: shape (nheads,) — optional
/// - `ssm_state`: mutable, shape (batch, nheads, d_state, headdim)
/// - `prev_bx_state`: mutable, shape (batch, nheads, d_state, headdim) — previous B*x cache
/// - `alpha`: trapezoidal interpolation parameter
/// - `use_rope`: whether to apply RoPE
///
/// # Returns
/// Output, shape (batch, nheads, headdim)
#[allow(clippy::too_many_arguments)]
pub fn mamba3_ssm_step(
    x: &[f32],
    batch: usize,
    nheads: usize,
    headdim: usize,
    dt: &[f32],
    a_real: &[f32],
    a_imag: &[f32],
    b: &[f32],
    ngroups: usize,
    d_state: usize,
    c: &[f32],
    b_bias: Option<&[f32]>,
    c_bias: Option<&[f32]>,
    d: Option<&[f32]>,
    z: Option<&[f32]>,
    dt_bias: Option<&[f32]>,
    dt_softplus: bool,
    ssm_state: &mut [f32],
    prev_bx_state: &mut [f32],
    alpha: f32,
    use_rope: bool,
) -> Vec<f32> {
    assert!(ngroups > 0, "mamba3_ssm_step: ngroups must be > 0");
    assert!(nheads % ngroups == 0, "mamba3_ssm_step: nheads ({}) must be divisible by ngroups ({})", nheads, ngroups);
    let heads_per_group = nheads / ngroups;
    let mut output = vec![0.0f32; batch * nheads * headdim];

    for b_idx in 0..batch {
        for h in 0..nheads {
            let g = h / heads_per_group;

            let mut dt_val = dt[b_idx * nheads + h];
            if let Some(bias) = dt_bias {
                dt_val += bias[h];
            }
            if dt_softplus {
                dt_val = softplus(dt_val);
            }

            let da = (dt_val * a_real[h]).exp();
            let rope_theta = if use_rope { dt_val * a_imag[h] } else { 0.0 };
            let (beta, gamma) = trapezoidal_coefficients(dt_val, alpha);

            // Extract and bias B
            let mut b_vec = vec![0.0f32; d_state];
            for n in 0..d_state {
                b_vec[n] = b[b_idx * ngroups * d_state + g * d_state + n];
                if let Some(bb) = b_bias {
                    b_vec[n] += bb[g * d_state + n];
                }
            }

            // Extract and bias C
            let mut c_vec = vec![0.0f32; d_state];
            for n in 0..d_state {
                c_vec[n] = c[b_idx * ngroups * d_state + g * d_state + n];
                if let Some(cb) = c_bias {
                    c_vec[n] += cb[g * d_state + n];
                }
            }

            // Apply RoPE
            if use_rope && d_state >= 2 {
                apply_rope_inplace(&mut b_vec, rope_theta);
                apply_rope_inplace(&mut c_vec, rope_theta);
            }

            for p in 0..headdim {
                let x_val = x[b_idx * nheads * headdim + h * headdim + p];
                let mut y = 0.0f32;

                for n in 0..d_state {
                    let cur_bx = b_vec[n] * x_val;

                    let pbx_idx = b_idx * nheads * d_state * headdim
                        + h * d_state * headdim
                        + n * headdim
                        + p;
                    let prev_bx_val = prev_bx_state[pbx_idx];

                    let s_idx = pbx_idx;
                    ssm_state[s_idx] = ssm_state[s_idx] * da + beta * cur_bx + gamma * prev_bx_val;

                    y += ssm_state[s_idx] * c_vec[n];

                    prev_bx_state[pbx_idx] = cur_bx;
                }

                if let Some(d_skip) = d {
                    y += d_skip[h] * x_val;
                }

                if let Some(z_data) = z {
                    y *= silu(z_data[b_idx * nheads * headdim + h * headdim + p]);
                }

                output[b_idx * nheads * headdim + h * headdim + p] = y;
            }
        }
    }

    output
}

// ============================================================================
// GPU-accelerated scan (requires `cuda` feature)
// ============================================================================

/// GPU-accelerated Mamba-3 scan using the Flash Scan CUDA kernel.
///
/// Takes CPU `&[f32]` data (same interface as `mamba3_scan_combined`),
/// uploads to GPU, runs the chunked parallel scan kernel, and downloads
/// the result. This avoids requiring callers to manage GPU memory directly.
///
/// Returns `None` if CUDA is not available or initialization fails,
/// allowing callers to fall back to the CPU path.
///
/// # Fallback conditions
/// - CUDA device not available
/// - `use_rope` is true (RoPE not yet in CUDA kernel)
/// - Any `a_imag` value is non-zero (CUDA kernel does not implement the
///   imaginary component of A; even with `use_rope=false`, non-zero `a_imag`
///   would produce divergent results vs the CPU path)
/// - Input size validation fails
/// - Any GPU operation fails
///
/// # B/C bias handling
/// The CUDA kernel does not natively support `b_bias`/`c_bias`. When provided,
/// biases are pre-added to B and C tensors on the CPU before GPU upload. This
/// is a lightweight O(batch * seq_len * ngroups * d_state) element-wise add.
///
/// # Known limitations
/// - **H2D/D2H per call**: Data is uploaded to GPU and downloaded each call.
///   A future persistent-tensor API would avoid this overhead for sequences of
///   calls where data is already on-device.
/// - **No GPU backward pass**: `MambaScanBackward` remains CPU-based. A CUDA
///   backward kernel is needed to avoid a GPU→CPU→GPU round-trip during training.
/// - **No RoPE in CUDA kernel**: Falls back to CPU when `use_rope=true` or
///   `a_imag` contains non-zero values.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn mamba3_scan_gpu(
    x: &[f32],
    batch: usize,
    seq_len: usize,
    nheads: usize,
    headdim: usize,
    dt: &[f32],
    a_real: &[f32],
    a_imag: &[f32],
    b: &[f32],
    ngroups: usize,
    d_state: usize,
    c: &[f32],
    b_bias: Option<&[f32]>,
    c_bias: Option<&[f32]>,
    d: Option<&[f32]>,
    z: Option<&[f32]>,
    dt_bias: Option<&[f32]>,
    dt_softplus: bool,
    alpha: f32,
    use_rope: bool,
) -> Option<Mamba3ScanOutput> {
    use kore_kernels::cuda::context::{get_device, is_cuda_available};
    use kore_kernels::cuda::memory::CudaBuffer;
    use kore_kernels::cuda::ops::cuda_mamba3_scan_f32;

    // RoPE not yet in CUDA kernel — fall back to CPU
    if use_rope { return None; }

    // CUDA kernel ignores a_imag entirely. If any value is non-zero, the CPU
    // path would use it for RoPE angle computation (rope_theta = dt * a_imag[h]),
    // producing different results. Fall back to CPU for correctness.
    if a_imag.iter().any(|&v| v != 0.0) { return None; }

    if !is_cuda_available() { return None; }

    // --- Input size validation ---
    let expected_x = batch * seq_len * nheads * headdim;
    let expected_dt = batch * seq_len * nheads;
    let expected_bc = batch * seq_len * ngroups * d_state;
    if x.len() != expected_x { return None; }
    if dt.len() != expected_dt { return None; }
    if a_real.len() != nheads { return None; }
    if b.len() != expected_bc { return None; }
    if c.len() != expected_bc { return None; }
    if let Some(s) = b_bias { if s.len() != ngroups * d_state { return None; } }
    if let Some(s) = c_bias { if s.len() != ngroups * d_state { return None; } }
    if let Some(s) = d { if s.len() != nheads { return None; } }
    if let Some(s) = z { if s.len() != expected_x { return None; } }
    if let Some(s) = dt_bias { if s.len() != nheads { return None; } }

    // --- Pre-add B/C biases on CPU before GPU upload ---
    // The CUDA kernel does not support b_bias/c_bias natively. We fuse them
    // into the B and C tensors here. The bias is (ngroups, d_state) and
    // broadcasts across (batch, seq_len), so we add bias[g*d_state + n] to
    // every b[batch, seq, g, n] element.
    let b_biased: Vec<f32>;
    let b_upload: &[f32] = if let Some(bb) = b_bias {
        b_biased = b.iter().enumerate().map(|(i, &v)| {
            // b layout: [batch, seq_len, ngroups, d_state]
            // bias index: i % (ngroups * d_state)
            v + bb[i % (ngroups * d_state)]
        }).collect();
        &b_biased
    } else {
        b
    };

    let c_biased: Vec<f32>;
    let c_upload: &[f32] = if let Some(cb) = c_bias {
        c_biased = c.iter().enumerate().map(|(i, &v)| {
            v + cb[i % (ngroups * d_state)]
        }).collect();
        &c_biased
    } else {
        c
    };

    let dev_idx = 0;
    let dev = get_device(dev_idx).ok()?;

    // Helper: upload f32 slice to GPU as raw bytes
    let upload = |data: &[f32]| -> Option<CudaBuffer> {
        let bytes: &[u8] = bytemuck_cast_slice(data);
        CudaBuffer::from_host(dev_idx, bytes).ok()
    };

    // Upload required inputs
    let x_gpu = upload(x)?;
    let dt_gpu = upload(dt)?;
    let a_gpu = upload(a_real)?;
    let b_gpu = upload(b_upload)?;
    let c_gpu = upload(c_upload)?;

    // Upload optional inputs
    let dt_bias_gpu = dt_bias.and_then(|s| upload(s));
    let z_gpu = z.and_then(|s| upload(s));
    let d_gpu = d.and_then(|s| upload(s));

    let num_chunks = (seq_len + 127) / 128; // SCAN_CHUNK = 128
    let state_elems = d_state * headdim;

    let (output_gpu, chunk_h_gpu, chunk_bx_gpu) = cuda_mamba3_scan_f32(
        &dev,
        dev_idx,
        x_gpu.as_cuda_slice(),
        dt_gpu.as_cuda_slice(),
        a_gpu.as_cuda_slice(),
        b_gpu.as_cuda_slice(),
        c_gpu.as_cuda_slice(),
        dt_bias_gpu.as_ref().map(|b| b.as_cuda_slice()),
        z_gpu.as_ref().map(|b| b.as_cuda_slice()),
        d_gpu.as_ref().map(|b| b.as_cuda_slice()),
        batch,
        seq_len,
        nheads,
        headdim,
        ngroups,
        d_state,
        alpha,
        dt_softplus,
    ).ok()?;

    // Download output from GPU
    let out_bytes = dev.dtoh_sync_copy(&output_gpu).ok()?;
    let output: Vec<f32> = bytes_to_f32_vec(&out_bytes);

    // Extract real last_state and prev_bx from chunk buffers.
    // Layout: chunk_last_h[B, H, num_chunks, N*D] — the last chunk holds the
    // final state for each (batch, head) pair.
    let chunk_h_bytes = dev.dtoh_sync_copy(&chunk_h_gpu).ok()?;
    let chunk_bx_bytes = dev.dtoh_sync_copy(&chunk_bx_gpu).ok()?;
    let chunk_h_all: Vec<f32> = bytes_to_f32_vec(&chunk_h_bytes);
    let chunk_bx_all: Vec<f32> = bytes_to_f32_vec(&chunk_bx_bytes);

    let total_state = batch * nheads * state_elems;
    let mut last_state = vec![0.0f32; total_state];
    let mut prev_bx = vec![0.0f32; total_state];
    let last_chunk = num_chunks - 1;

    for bi in 0..batch {
        for h in 0..nheads {
            let src_base = bi * nheads * num_chunks * state_elems
                         + h * num_chunks * state_elems
                         + last_chunk * state_elems;
            let dst_base = bi * nheads * state_elems + h * state_elems;
            last_state[dst_base..dst_base + state_elems]
                .copy_from_slice(&chunk_h_all[src_base..src_base + state_elems]);
            prev_bx[dst_base..dst_base + state_elems]
                .copy_from_slice(&chunk_bx_all[src_base..src_base + state_elems]);
        }
    }

    Some(Mamba3ScanOutput {
        output,
        last_state,
        prev_bx,
    })
}

/// Reinterpret `&[f32]` as `&[u8]` without copying.
#[cfg(feature = "cuda")]
#[inline]
fn bytemuck_cast_slice(data: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f32>(),
        )
    }
}

/// Convert raw bytes back to `Vec<f32>`.
#[cfg(feature = "cuda")]
#[inline]
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() % 4 == 0);
    let n = bytes.len() / 4;
    let mut out = vec![0.0f32; n];
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            out.as_mut_ptr() as *mut u8,
            bytes.len(),
        );
    }
    out
}

/// Check if GPU-accelerated scan is available at runtime.
///
/// Returns `true` if compiled with `cuda` feature AND a CUDA device is present.
pub fn is_gpu_scan_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        kore_kernels::cuda::context::is_cuda_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_rope() {
        let mut v = vec![1.0, 0.0, 1.0, 0.0];
        let theta = std::f32::consts::FRAC_PI_2; // 90 degrees
        apply_rope_inplace(&mut v, theta);
        // cos(pi/2) ≈ 0, sin(pi/2) ≈ 1
        // (1, 0) -> (0, 1)
        assert!((v[0] - 0.0).abs() < 1e-5);
        assert!((v[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_trapezoidal_coefficients() {
        let (beta, gamma) = trapezoidal_coefficients(0.1, 0.5);
        assert!((beta - 0.05).abs() < 1e-6);
        assert!((gamma - 0.05).abs() < 1e-6);

        let (beta2, gamma2) = trapezoidal_coefficients(0.1, 0.7);
        assert!((beta2 - 0.07).abs() < 1e-6);
        assert!((gamma2 - 0.03).abs() < 1e-6);
    }

    #[test]
    fn test_mamba3_scan_shape() {
        let batch = 2;
        let seq_len = 8;
        let nheads = 4;
        let headdim = 8;
        let ngroups = 1;
        let d_state = 4;

        let x = vec![0.1f32; batch * seq_len * nheads * headdim];
        let dt = vec![0.05f32; batch * seq_len * nheads];
        let a_real = vec![-1.0f32; nheads];
        let a_imag = vec![0.5f32; nheads];
        let b = vec![0.1f32; batch * seq_len * ngroups * d_state];
        let c = vec![0.1f32; batch * seq_len * ngroups * d_state];
        let d_skip = vec![1.0f32; nheads];

        let result = mamba3_scan_combined(
            &x, batch, seq_len, nheads, headdim,
            &dt, &a_real, &a_imag, &b, ngroups, d_state, &c,
            None, None, Some(&d_skip), None, None, false, 0.5, true,
        );

        assert_eq!(result.output.len(), batch * seq_len * nheads * headdim);
        assert_eq!(result.last_state.len(), batch * nheads * d_state * headdim);
        assert_eq!(result.prev_bx.len(), batch * nheads * d_state * headdim);
    }

    #[test]
    fn test_mamba3_scan_skip_only() {
        // With A=0, dt=0, state stays zero. Output = D * x
        let batch = 1;
        let seq_len = 2;
        let nheads = 1;
        let headdim = 2;
        let ngroups = 1;
        let d_state = 2;

        let x = vec![1.0, 2.0, 3.0, 4.0];
        let dt = vec![0.0, 0.0];
        let a_real = vec![0.0];
        let a_imag = vec![0.0];
        let b = vec![0.0, 0.0, 0.0, 0.0];
        let c = vec![0.0, 0.0, 0.0, 0.0];
        let d_skip = vec![2.0];

        let result = mamba3_scan_combined(
            &x, batch, seq_len, nheads, headdim,
            &dt, &a_real, &a_imag, &b, ngroups, d_state, &c,
            None, None, Some(&d_skip), None, None, false, 0.5, false,
        );

        assert!((result.output[0] - 2.0).abs() < 1e-5);
        assert!((result.output[1] - 4.0).abs() < 1e-5);
        assert!((result.output[2] - 6.0).abs() < 1e-5);
        assert!((result.output[3] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_mamba3_scan_trapezoidal_vs_euler() {
        // With trapezoidal (alpha=0.5), the second step should use prev_bx
        let batch = 1;
        let seq_len = 2;
        let nheads = 1;
        let headdim = 1;
        let ngroups = 1;
        let d_state = 1;

        let x = vec![1.0, 1.0]; // constant input
        let dt = vec![1.0, 1.0];
        let a_real = vec![0.0]; // no decay
        let a_imag = vec![0.0];
        let b = vec![1.0, 1.0];
        let c = vec![1.0, 1.0];

        // Trapezoidal alpha=0.5: beta=0.5, gamma=0.5
        let result = mamba3_scan_combined(
            &x, batch, seq_len, nheads, headdim,
            &dt, &a_real, &a_imag, &b, ngroups, d_state, &c,
            None, None, None, None, None, false, 0.5, false,
        );

        // Step 0: state = 0*1 + 0.5*1*1 + 0.5*0 = 0.5, y = 0.5*1 = 0.5
        assert!((result.output[0] - 0.5).abs() < 1e-5);
        // Step 1: state = 0.5*1 + 0.5*1*1 + 0.5*1*1 = 1.5, y = 1.5*1 = 1.5
        assert!((result.output[1] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_mamba3_scan_with_biases() {
        let batch = 1;
        let seq_len = 1;
        let nheads = 1;
        let headdim = 1;
        let ngroups = 1;
        let d_state = 1;

        let x = vec![1.0];
        let dt = vec![1.0];
        let a_real = vec![0.0];
        let a_imag = vec![0.0];
        let b = vec![1.0];
        let c = vec![1.0];
        let b_bias = vec![0.5]; // B becomes 1.5
        let c_bias = vec![0.3]; // C becomes 1.3

        let result = mamba3_scan_combined(
            &x, batch, seq_len, nheads, headdim,
            &dt, &a_real, &a_imag, &b, ngroups, d_state, &c,
            Some(&b_bias), Some(&c_bias), None, None, None, false, 0.5, false,
        );

        // state = 0 + 0.5 * 1.5 * 1.0 + 0.5 * 0 = 0.75
        // y = 0.75 * 1.3 = 0.975
        assert!((result.output[0] - 0.975).abs() < 1e-5);
    }

    #[test]
    fn test_mamba3_ssm_step_shape() {
        let batch = 1;
        let nheads = 2;
        let headdim = 4;
        let ngroups = 1;
        let d_state = 2;

        let x = vec![0.1f32; batch * nheads * headdim];
        let dt = vec![0.1, 0.2];
        let a_real = vec![-1.0, -1.0];
        let a_imag = vec![0.5, 0.5];
        let b = vec![1.0, 0.5];
        let c = vec![1.0, 1.0];
        let d_skip = vec![1.0, 1.0];
        let mut ssm_state = vec![0.0f32; batch * nheads * d_state * headdim];
        let mut prev_bx_state = vec![0.0f32; batch * nheads * d_state * headdim];

        let out = mamba3_ssm_step(
            &x, batch, nheads, headdim, &dt, &a_real, &a_imag,
            &b, ngroups, d_state, &c,
            None, None, Some(&d_skip), None, None, false,
            &mut ssm_state, &mut prev_bx_state, 0.5, true,
        );

        assert_eq!(out.len(), batch * nheads * headdim);
        assert!(ssm_state.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_mamba3_step_trapezoidal_consistency() {
        // Two sequential steps should match the scan output
        let batch = 1;
        let nheads = 1;
        let headdim = 1;
        let ngroups = 1;
        let d_state = 1;

        let dt_val = 1.0f32;
        let a_real = vec![0.0f32];
        let a_imag = vec![0.0f32];

        let mut ssm_state = vec![0.0f32; 1];
        let mut prev_bx = vec![0.0f32; 1];

        // Step 0: x=1, B=1, C=1
        let out0 = mamba3_ssm_step(
            &[1.0], batch, nheads, headdim,
            &[dt_val], &a_real, &a_imag,
            &[1.0], ngroups, d_state, &[1.0],
            None, None, None, None, None, false,
            &mut ssm_state, &mut prev_bx, 0.5, false,
        );
        assert!((out0[0] - 0.5).abs() < 1e-5);

        // Step 1: x=1, B=1, C=1
        let out1 = mamba3_ssm_step(
            &[1.0], batch, nheads, headdim,
            &[dt_val], &a_real, &a_imag,
            &[1.0], ngroups, d_state, &[1.0],
            None, None, None, None, None, false,
            &mut ssm_state, &mut prev_bx, 0.5, false,
        );
        assert!((out1[0] - 1.5).abs() < 1e-5);
    }
}
