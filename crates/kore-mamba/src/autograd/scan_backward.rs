//! Backward pass for `mamba3_scan_combined`.

use kore_core::autograd::GradFn;
use kore_core::Tensor;

use super::scan_forward_saved::MambaScanSaved;

pub struct MambaScanBackward {
    pub saved: MambaScanSaved,
}

#[inline]
fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

#[inline]
fn silu_grad(x: f32) -> f32 {
    let sig = 1.0 / (1.0 + (-x).exp());
    sig * (1.0 + x * (1.0 - sig))
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else if x < -20.0 { 0.0 } else { (1.0 + x.exp()).ln() }
}

#[inline]
fn softplus_grad(x: f32) -> f32 {
    if x > 20.0 { 1.0 } else if x < -20.0 { 0.0 } else { 1.0 / (1.0 + (-x).exp()) }
}

#[inline]
fn apply_rope_inplace(v: &mut [f32], theta: f32) {
    let (cos_t, sin_t) = (theta.cos(), theta.sin());
    for i in 0..v.len() / 2 {
        let (a, b) = (v[2 * i], v[2 * i + 1]);
        v[2 * i] = a * cos_t - b * sin_t;
        v[2 * i + 1] = a * sin_t + b * cos_t;
    }
}

/// Helper to reinterpret `&[f32]` as `&[u8]` without copying.
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

/// Helper to reinterpret `Vec<u8>` as `Vec<f32>`.
#[cfg(feature = "cuda")]
#[inline]
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() % 4 == 0);
    let n = bytes.len() / 4;
    let mut out = vec![0.0f32; n];
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
    }
    out
}

impl MambaScanBackward {
    /// Try to run backward on GPU. Returns None if CUDA unavailable or GPU context missing.
    #[cfg(feature = "cuda")]
    fn try_gpu_backward(&self, go_data: &[f32]) -> Option<Vec<Option<Tensor>>> {
        use kore_kernels::cuda::ops::cuda_mamba3_scan_backward_f32;

        let s = &self.saved;
        let (batch, seq_len, nheads, headdim) = (s.batch, s.seq_len, s.nheads, s.headdim);
        let (ngroups, d_state) = (s.ngroups, s.d_state);
        let has_z = s.z.is_some();

        let gpu_ctx = s.gpu_ctx.as_ref()?;
        let dev = &gpu_ctx.dev;
        let dev_idx = gpu_ctx.dev_idx;

        // Upload grad_output to GPU
        let go_gpu = dev.htod_copy(bytemuck_cast_slice(go_data).to_vec()).ok()?;

        let (dx_gpu, d_dt_gpu, d_b_gpu, d_c_gpu, dz_gpu) = cuda_mamba3_scan_backward_f32(
            dev,
            dev_idx,
            &go_gpu,
            &gpu_ctx.x,
            &gpu_ctx.dt,
            &gpu_ctx.a_real,
            &gpu_ctx.a_imag,
            &gpu_ctx.b,
            &gpu_ctx.c,
            &gpu_ctx.h_all,
            &gpu_ctx.bx_all,
            gpu_ctx.dt_bias.as_ref(),
            gpu_ctx.z.as_ref(),
            gpu_ctx.d_skip.as_ref(),
            batch,
            seq_len,
            nheads,
            headdim,
            ngroups,
            d_state,
            s.alpha,
            s.dt_softplus,
            s.use_rope,
        ).ok()?;

        // Download gradients from GPU
        let dx: Vec<f32> = bytes_to_f32_vec(&dev.dtoh_sync_copy(&dx_gpu).ok()?);
        let d_dt: Vec<f32> = bytes_to_f32_vec(&dev.dtoh_sync_copy(&d_dt_gpu).ok()?);
        let d_b: Vec<f32> = bytes_to_f32_vec(&dev.dtoh_sync_copy(&d_b_gpu).ok()?);
        let d_c: Vec<f32> = bytes_to_f32_vec(&dev.dtoh_sync_copy(&d_c_gpu).ok()?);

        let dx_t = Tensor::from_f32(&dx, &[batch, seq_len, nheads * headdim]);
        let dt_t = Tensor::from_f32(&d_dt, &[batch, seq_len, nheads]);
        let b_t = Tensor::from_f32(&d_b, &[batch, seq_len, ngroups * d_state]);
        let c_t = Tensor::from_f32(&d_c, &[batch, seq_len, ngroups * d_state]);

        if has_z {
            let dz: Vec<f32> = bytes_to_f32_vec(&dev.dtoh_sync_copy(&dz_gpu).ok()?);
            let z_t = Tensor::from_f32(&dz, &[batch, seq_len, nheads * headdim]);
            Some(vec![Some(dx_t), Some(dt_t), Some(b_t), Some(c_t), Some(z_t)])
        } else {
            Some(vec![Some(dx_t), Some(dt_t), Some(b_t), Some(c_t)])
        }
    }

    /// CPU backward pass (existing implementation).
    fn cpu_backward(&self, go_data: &[f32]) -> Vec<Option<Tensor>> {
        let s = &self.saved;
        let (batch, seq_len, nheads, headdim) = (s.batch, s.seq_len, s.nheads, s.headdim);
        let (ngroups, d_state) = (s.ngroups, s.d_state);
        let hpg = nheads / ngroups;
        let frame = nheads * d_state * headdim;
        let nf = seq_len + 1;
        let has_z = s.z.is_some();

        let mut dx = vec![0.0f32; batch * seq_len * nheads * headdim];
        let mut d_dt = vec![0.0f32; batch * seq_len * nheads];
        let mut d_b = vec![0.0f32; batch * seq_len * ngroups * d_state];
        let mut d_c = vec![0.0f32; batch * seq_len * ngroups * d_state];
        let mut dz = if has_z { vec![0.0f32; batch * seq_len * nheads * headdim] } else { vec![] };

        let ss = nheads * d_state * headdim;

        for bi in 0..batch {
            let mut dh = vec![0.0f32; ss];
            let mut d_cur_bx = vec![0.0f32; ss];

            for l in (0..seq_len).rev() {
                for h in 0..nheads {
                    let g = h / hpg;
                    let raw = s.dt[bi * seq_len * nheads + l * nheads + h];
                    let biased = raw + s.dt_bias.as_ref().map_or(0.0, |b| b[h]);
                    let dtv = if s.dt_softplus { softplus(biased) } else { biased };
                    let da = (dtv * s.a_real[h]).exp();
                    let rope_t = if s.use_rope { dtv * s.a_imag[h] } else { 0.0 };
                    let (beta, gamma) = (dtv * s.alpha, dtv * (1.0 - s.alpha));

                    let mut bv = vec![0.0f32; d_state];
                    let mut cv = vec![0.0f32; d_state];
                    for n in 0..d_state {
                        let idx = bi * seq_len * ngroups * d_state + l * ngroups * d_state + g * d_state + n;
                        bv[n] = s.b[idx] + s.b_bias.as_ref().map_or(0.0, |bb| bb[g * d_state + n]);
                        cv[n] = s.c[idx] + s.c_bias.as_ref().map_or(0.0, |cb| cb[g * d_state + n]);
                    }
                    if s.use_rope && d_state >= 2 {
                        apply_rope_inplace(&mut bv, rope_t);
                        apply_rope_inplace(&mut cv, rope_t);
                    }

                    let mut d_dtv = 0.0f32;

                    for p in 0..headdim {
                        let oi = bi * seq_len * nheads * headdim + l * nheads * headdim + h * headdim + p;
                        let xi = oi;
                        let xv = s.x[xi];

                        // Ungate
                        let dy_pre = if has_z {
                            let zv = s.z.as_ref().unwrap()[oi];
                            let sz = silu(zv);
                            let mut yp = 0.0f32;
                            for n in 0..d_state {
                                let si = h * d_state * headdim + n * headdim + p;
                                yp += s.h_all[(bi * nf + l + 1) * frame + si] * cv[n];
                            }
                            if let Some(ref ds) = s.d { yp += ds[h] * xv; }
                            dz[oi] += go_data[oi] * yp * silu_grad(zv);
                            go_data[oi] * sz
                        } else {
                            go_data[oi]
                        };

                        // D skip grad
                        if let Some(ref ds) = s.d {
                            dx[xi] += dy_pre * ds[h];
                        }

                        for n in 0..d_state {
                            let si = h * d_state * headdim + n * headdim + p;
                            let ht = s.h_all[(bi * nf + l + 1) * frame + si];
                            let hp = s.h_all[(bi * nf + l) * frame + si];
                            let pbx = s.bx_all[(bi * nf + l) * frame + si];
                            let cur_bx = bv[n] * xv;

                            let ci = bi * seq_len * ngroups * d_state + l * ngroups * d_state + g * d_state + n;
                            d_c[ci] += dy_pre * ht;

                            dh[si] += dy_pre * cv[n];

                            let d_cbx = d_cur_bx[si];
                            let bi_idx = bi * seq_len * ngroups * d_state + l * ngroups * d_state + g * d_state + n;
                            dx[xi] += d_cbx * bv[n];
                            d_b[bi_idx] += d_cbx * xv;

                            let dh_t = dh[si];

                            let dh_prop = dh_t * da;

                            d_dtv += dh_t * hp * da * s.a_real[h];

                            dx[xi] += dh_t * beta * bv[n];
                            d_b[bi_idx] += dh_t * beta * xv;
                            d_dtv += dh_t * cur_bx * s.alpha;

                            d_cur_bx[si] = dh_t * gamma;
                            d_dtv += dh_t * pbx * (1.0 - s.alpha);

                            dh[si] = dh_prop;
                        }
                    }

                    let dtg = if s.dt_softplus { d_dtv * softplus_grad(biased) } else { d_dtv };
                    d_dt[bi * seq_len * nheads + l * nheads + h] += dtg;
                }
            }
        }

        let dx_t = Tensor::from_f32(&dx, &[batch, seq_len, nheads * headdim]);
        let dt_t = Tensor::from_f32(&d_dt, &[batch, seq_len, nheads]);
        let b_t = Tensor::from_f32(&d_b, &[batch, seq_len, ngroups * d_state]);
        let c_t = Tensor::from_f32(&d_c, &[batch, seq_len, ngroups * d_state]);

        if has_z {
            let z_t = Tensor::from_f32(&dz, &[batch, seq_len, nheads * headdim]);
            vec![Some(dx_t), Some(dt_t), Some(b_t), Some(c_t), Some(z_t)]
        } else {
            vec![Some(dx_t), Some(dt_t), Some(b_t), Some(c_t)]
        }
    }
}

impl GradFn for MambaScanBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let go = grad_output.contiguous();
        let go_data = go.as_f32_slice().unwrap();

        // Try GPU backward first (zero-copy from GPU-resident saved states)
        #[cfg(feature = "cuda")]
        {
            if let Some(result) = self.try_gpu_backward(go_data) {
                return result;
            }
        }

        // Fallback to CPU backward
        self.cpu_backward(go_data)
    }

    fn name(&self) -> &str { "MambaScanBackward" }
}
