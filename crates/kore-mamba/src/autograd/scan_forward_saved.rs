//! Forward pass variant that saves per-timestep states for backward.

use std::sync::Arc;

use kore_core::autograd::GradNode;
use kore_core::Tensor;

use super::MambaScanBackward;

/// GPU-resident saved context for backward pass.
/// Holds all forward inputs and per-timestep states on GPU to avoid
/// D2H+H2D round-trips between forward and backward.
#[cfg(feature = "cuda")]
pub struct GpuSavedContext {
    pub dev: Arc<cudarc::driver::CudaDevice>,
    pub dev_idx: usize,
    pub x: cudarc::driver::CudaSlice<u8>,
    pub dt: cudarc::driver::CudaSlice<u8>,
    pub a_real: cudarc::driver::CudaSlice<u8>,
    pub a_imag: cudarc::driver::CudaSlice<u8>,
    pub b: cudarc::driver::CudaSlice<u8>,
    pub c: cudarc::driver::CudaSlice<u8>,
    pub dt_bias: Option<cudarc::driver::CudaSlice<u8>>,
    pub z: Option<cudarc::driver::CudaSlice<u8>>,
    pub d_skip: Option<cudarc::driver::CudaSlice<u8>>,
    pub h_all: cudarc::driver::CudaSlice<u8>,
    pub bx_all: cudarc::driver::CudaSlice<u8>,
}

/// All data saved during the forward pass, needed by the backward.
pub struct MambaScanSaved {
    // Inputs (cloned)
    pub x: Vec<f32>,
    pub dt: Vec<f32>,
    pub a_real: Vec<f32>,
    pub a_imag: Vec<f32>,
    pub b: Vec<f32>,
    pub c: Vec<f32>,
    pub b_bias: Option<Vec<f32>>,
    pub c_bias: Option<Vec<f32>>,
    pub d: Option<Vec<f32>>,
    pub z: Option<Vec<f32>>,
    pub dt_bias: Option<Vec<f32>>,

    // Shape metadata
    pub batch: usize,
    pub seq_len: usize,
    pub nheads: usize,
    pub headdim: usize,
    pub ngroups: usize,
    pub d_state: usize,
    pub dt_softplus: bool,
    pub alpha: f32,
    pub use_rope: bool,

    // Per-timestep hidden states: [batch * (seq_len+1) * nheads * d_state * headdim]
    // Index [b][l] at offset: (b * (seq_len+1) + l) * nheads * d_state * headdim
    // l=0 is the initial zero state, l=t+1 is the state after timestep t.
    pub h_all: Vec<f32>,
    // Per-timestep prev_bx cache: same layout as h_all
    pub bx_all: Vec<f32>,

    // GPU-resident saved context (set when forward ran on GPU with save_states=true).
    // When present, backward uses these directly instead of re-uploading CPU data.
    #[cfg(feature = "cuda")]
    pub gpu_ctx: Option<GpuSavedContext>,
}

/// SiLU activation
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Softplus activation
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else if x < -20.0 { 0.0 } else { (1.0 + x.exp()).ln() }
}

/// Apply RoPE rotation in-place
#[inline]
fn apply_rope_inplace(v: &mut [f32], theta: f32) {
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    for i in 0..v.len() / 2 {
        let a = v[2 * i];
        let b = v[2 * i + 1];
        v[2 * i] = a * cos_t - b * sin_t;
        v[2 * i + 1] = a * sin_t + b * cos_t;
    }
}

/// Forward scan that saves all intermediate states for backward.
///
/// Returns `(output_tensor_with_grad_node, scan_output)` where the tensor
/// has a `GradNode` attached if any input tracks gradients.
///
/// The `input_nodes` should contain GradNodes for [x, dt, B, C, z?] in order.
#[allow(clippy::too_many_arguments)]
pub fn mamba3_scan_with_grad(
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
    input_nodes: Vec<Arc<GradNode>>,
) -> (Tensor, crate::ssd3::Mamba3ScanOutput) {
    assert!(ngroups > 0);
    assert!(nheads % ngroups == 0);
    let heads_per_group = nheads / ngroups;

    let out_size = batch * seq_len * nheads * headdim;
    let state_size = batch * nheads * d_state * headdim;
    let frame_size = nheads * d_state * headdim;
    let num_frames = seq_len + 1; // frame 0 = initial zeros

    let mut output = vec![0.0f32; out_size];
    let mut state = vec![0.0f32; state_size];
    let mut prev_bx = vec![0.0f32; state_size];

    // Allocate per-timestep saves
    let mut h_all = vec![0.0f32; batch * num_frames * frame_size];
    let mut bx_all = vec![0.0f32; batch * num_frames * frame_size];
    // Frame 0 is already zeros (initial state)

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

                let da = (dt_val * a_real[h]).exp();
                let rope_theta = if use_rope { dt_val * a_imag[h] } else { 0.0 };
                let (beta, gamma) = (dt_val * alpha, dt_val * (1.0 - alpha));

                // Extract B vector
                let mut b_vec = vec![0.0f32; d_state];
                for n in 0..d_state {
                    b_vec[n] = b[b_idx * seq_len * ngroups * d_state
                        + l * ngroups * d_state + g * d_state + n];
                    if let Some(bb) = b_bias {
                        b_vec[n] += bb[g * d_state + n];
                    }
                }

                // Extract C vector
                let mut c_vec = vec![0.0f32; d_state];
                for n in 0..d_state {
                    c_vec[n] = c[b_idx * seq_len * ngroups * d_state
                        + l * ngroups * d_state + g * d_state + n];
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
                    let x_val = x[b_idx * seq_len * nheads * headdim
                        + l * nheads * headdim + h * headdim + p];

                    let mut y = 0.0f32;

                    for n in 0..d_state {
                        let cur_bx = b_vec[n] * x_val;
                        let s_idx = state_base + h * d_state * headdim + n * headdim + p;
                        let prev_bx_val = prev_bx[s_idx];

                        state[s_idx] = state[s_idx] * da + beta * cur_bx + gamma * prev_bx_val;
                        y += state[s_idx] * c_vec[n];
                        prev_bx[s_idx] = cur_bx;
                    }

                    if let Some(d_skip) = d {
                        y += d_skip[h] * x_val;
                    }

                    if let Some(z_data) = z {
                        let z_val = z_data[b_idx * seq_len * nheads * headdim
                            + l * nheads * headdim + h * headdim + p];
                        y *= silu(z_val);
                    }

                    output[b_idx * seq_len * nheads * headdim
                        + l * nheads * headdim + h * headdim + p] = y;
                }
            }

            // Save state snapshot after this timestep (frame l+1)
            let save_offset = (b_idx * num_frames + (l + 1)) * frame_size;
            h_all[save_offset..save_offset + frame_size]
                .copy_from_slice(&state[state_base..state_base + frame_size]);
            bx_all[save_offset..save_offset + frame_size]
                .copy_from_slice(&prev_bx[state_base..state_base + frame_size]);
        }
    }

    let scan_output = crate::ssd3::Mamba3ScanOutput {
        output: output.clone(),
        last_state: state,
        prev_bx,
    };

    // Build saved context for backward
    let saved = MambaScanSaved {
        x: x.to_vec(),
        dt: dt.to_vec(),
        a_real: a_real.to_vec(),
        a_imag: a_imag.to_vec(),
        b: b.to_vec(),
        c: c.to_vec(),
        b_bias: b_bias.map(|s| s.to_vec()),
        c_bias: c_bias.map(|s| s.to_vec()),
        d: d.map(|s| s.to_vec()),
        z: z.map(|s| s.to_vec()),
        dt_bias: dt_bias.map(|s| s.to_vec()),
        batch,
        seq_len,
        nheads,
        headdim,
        ngroups,
        d_state,
        dt_softplus,
        alpha,
        use_rope,
        h_all,
        bx_all,
        #[cfg(feature = "cuda")]
        gpu_ctx: None,
    };

    // Build output tensor with grad node
    let out_shape = [batch, seq_len, nheads * headdim];
    let out_tensor = Tensor::from_f32(&output, &out_shape);

    if !input_nodes.is_empty() {
        let grad_fn = Box::new(MambaScanBackward { saved });
        let node = GradNode::with_grad_fn(grad_fn, input_nodes);
        (out_tensor.with_grad_node(node), scan_output)
    } else {
        (out_tensor, scan_output)
    }
}
