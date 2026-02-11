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

impl GradFn for MambaScanBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let s = &self.saved;
        let go = grad_output.contiguous();
        let go_data = go.as_f32_slice().unwrap();

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
            // dh: gradient w.r.t. h[l+1], propagated backward through time
            let mut dh = vec![0.0f32; ss];
            // d_cur_bx: gradient w.r.t. cur_bx = B_l*x_l that was saved as
            // prev_bx for timestep l+1. Set when processing l+1, consumed at l.
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

                            // dC: d(loss)/d(C[n]) += dy_pre * h_t[n,p]
                            let ci = bi * seq_len * ngroups * d_state + l * ngroups * d_state + g * d_state + n;
                            d_c[ci] += dy_pre * ht;

                            // dh from output: y_pre += h_t[n,p] * C[n]
                            dh[si] += dy_pre * cv[n];

                            // --- Consume d_cur_bx from timestep l+1 ---
                            // d_cur_bx[si] is the gradient of cur_bx = B_l[n]*x_l[p]
                            // computed at this timestep l, used as prev_bx at l+1.
                            let d_cbx = d_cur_bx[si];
                            let bi_idx = bi * seq_len * ngroups * d_state + l * ngroups * d_state + g * d_state + n;
                            dx[xi] += d_cbx * bv[n];
                            d_b[bi_idx] += d_cbx * xv;

                            // --- Recurrence backward ---
                            let dh_t = dh[si];

                            // d(h_{t-1}) += dh_t * da
                            let dh_prop = dh_t * da;

                            // d(da) = dh_t * h_{t-1}; da = exp(dtv * a_real)
                            // d(dtv) += dh_t * h_{t-1} * da * a_real
                            d_dtv += dh_t * hp * da * s.a_real[h];

                            // d(beta * B[n] * x[p])
                            dx[xi] += dh_t * beta * bv[n];
                            d_b[bi_idx] += dh_t * beta * xv;
                            // d(beta) = dh_t * cur_bx; beta = dtv * alpha
                            d_dtv += dh_t * cur_bx * s.alpha;

                            // d(gamma * prev_bx[n,p])
                            // This gradient flows to the cur_bx at timestep l-1
                            d_cur_bx[si] = dh_t * gamma;
                            // d(gamma) = dh_t * prev_bx; gamma = dtv * (1-alpha)
                            d_dtv += dh_t * pbx * (1.0 - s.alpha);

                            // Set dh for next reverse iteration (timestep l-1)
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

    fn name(&self) -> &str { "MambaScanBackward" }
}
