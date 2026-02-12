// Backward pass for Mamba-3 chunked SSM scan.
//
// Mirrors the CPU backward in scan_backward.rs exactly:
//   - Iterates backward through timesteps (t = seq_len-1 .. 0)
//   - Propagates dh (gradient w.r.t. hidden state) backward through time
//   - Propagates d_cur_bx (trapezoidal prev_bx gradient) backward
//   - Computes dx, d_dt, d_b, d_c, dz
//
// Thread model (same as forward kernel):
//   Grid:  (batch * nheads, 1, 1) — one block per (batch, head)
//   Block: (min(headdim, 256), 1, 1) — threads map to headdim positions
//   Each thread loops: for p = tid; p < headdim; p += blockDim.x
//   Each thread maintains dh[MAX_DSTATE] and d_cbx[MAX_DSTATE] in registers
//
// Requires saved per-timestep states from forward pass:
//   h_all:  [B, (L+1), H, N, D] — h_all[b][0] = zeros, h_all[b][l+1] = state after step l
//   bx_all: [B, (L+1), H, N, D] — same layout for prev_bx cache
//
// d_b and d_c use atomicAdd because multiple heads in the same group
// write to the same (batch, timestep, group, state) location.
// dx and d_dt are unique per (batch, timestep, head, headdim) — no atomics needed.

extern "C" {

#define MAX_DSTATE 64

__device__ float softplus_bwd(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return logf(1.0f + expf(x));
}

__device__ float softplus_grad_bwd(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

__device__ float silu_bwd(float x) {
    return x / (1.0f + expf(-x));
}

__device__ float silu_grad_bwd(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return sig * (1.0f + x * (1.0f - sig));
}

__device__ void apply_rope_bwd(float* v, unsigned int len, float cos_t, float sin_t) {
    unsigned int pairs = len / 2;
    for (unsigned int i = 0; i < pairs; i++) {
        float a = v[2 * i];
        float b = v[2 * i + 1];
        v[2 * i]     = a * cos_t - b * sin_t;
        v[2 * i + 1] = a * sin_t + b * cos_t;
    }
}

__global__ void mamba3_scan_backward_f32(
    // Gradient input
    const float* __restrict__ grad_output,   // [B, L, H, D]
    // Saved forward inputs
    const float* __restrict__ x,             // [B, L, H, D]
    const float* __restrict__ dt,            // [B, L, H]
    const float* __restrict__ a_real,        // [H]
    const float* __restrict__ a_imag,        // [H]
    const float* __restrict__ b_in,          // [B, L, G, N]
    const float* __restrict__ c_in,          // [B, L, G, N]
    // Saved per-timestep states
    const float* __restrict__ h_all,         // [B, (L+1), H, N, D]
    const float* __restrict__ bx_all,        // [B, (L+1), H, N, D]
    // Optional inputs (same pattern as forward)
    unsigned int has_dt_bias,
    const float* __restrict__ dt_bias,       // [H]
    unsigned int has_z,
    const float* __restrict__ z,             // [B, L, H, D]
    unsigned int has_d_skip,
    const float* __restrict__ d_skip,        // [H]
    // Gradient outputs
    float* __restrict__ dx,                  // [B, L, H, D]
    float* __restrict__ d_dt_out,            // [B, L, H]
    float* __restrict__ d_b_out,             // [B, L, G, N]
    float* __restrict__ d_c_out,             // [B, L, G, N]
    float* __restrict__ dz_out,              // [B, L, H, D] (only written if has_z)
    // Dimensions
    unsigned int batch,
    unsigned int seq_len,
    unsigned int nheads,
    unsigned int headdim,
    unsigned int ngroups,
    unsigned int d_state,
    float alpha,
    unsigned int dt_softplus,
    unsigned int use_rope
) {
    unsigned int block_id = blockIdx.x;
    unsigned int head = block_id % nheads;
    unsigned int batch_idx = block_id / nheads;
    if (batch_idx >= batch) return;

    unsigned int tid = threadIdx.x;
    unsigned int group = head / (nheads / ngroups);

    // Frame size for h_all/bx_all indexing: H * N * D
    unsigned int frame = nheads * d_state * headdim;
    unsigned int nf = seq_len + 1;  // number of frames (0 = initial zeros)

    float a_real_h = a_real[head];
    float a_imag_h = a_imag[head];

    // Each thread processes headdim positions p = tid, tid+blockDim.x, ...
    for (unsigned int p = tid; p < headdim; p += blockDim.x) {
        // Per-thread register state for backward propagation
        float dh[MAX_DSTATE];
        float d_cbx[MAX_DSTATE];
        for (unsigned int n = 0; n < d_state; n++) {
            dh[n] = 0.0f;
            d_cbx[n] = 0.0f;
        }

        // Backward time loop: t = seq_len-1 .. 0
        for (int t = (int)seq_len - 1; t >= 0; t--) {
            // --- Compute dt value (same as forward) ---
            float raw_dt = dt[batch_idx * seq_len * nheads + t * nheads + head];
            float biased_dt = raw_dt;
            if (has_dt_bias) biased_dt += dt_bias[head];
            float dtv = dt_softplus ? softplus_bwd(biased_dt) : biased_dt;

            float da = expf(dtv * a_real_h);
            float rope_t = (use_rope) ? dtv * a_imag_h : 0.0f;
            float beta = dtv * alpha;
            float gam = dtv * (1.0f - alpha);

            // --- Load B and C vectors into registers, apply bias + RoPE ---
            float bv[MAX_DSTATE];
            float cv[MAX_DSTATE];
            for (unsigned int n = 0; n < d_state; n++) {
                unsigned int bc_idx = batch_idx * seq_len * ngroups * d_state
                                    + t * ngroups * d_state + group * d_state + n;
                bv[n] = b_in[bc_idx];
                cv[n] = c_in[bc_idx];
            }
            if (use_rope && d_state >= 2) {
                float cos_t = cosf(rope_t);
                float sin_t = sinf(rope_t);
                apply_rope_bwd(bv, d_state, cos_t, sin_t);
                apply_rope_bwd(cv, d_state, cos_t, sin_t);
            }

            // --- Load x and grad_output for this position ---
            unsigned int oi = batch_idx * seq_len * nheads * headdim
                            + t * nheads * headdim + head * headdim + p;
            float xv = x[oi];
            float go_val = grad_output[oi];

            // --- Ungate (SiLU gating backward) ---
            float dy_pre;
            if (has_z) {
                float zv = z[oi];
                float sz = silu_bwd(zv);
                // Recompute y_pre for dz: y_pre = sum(h_t[n]*C[n]) + D*x
                float yp = 0.0f;
                for (unsigned int n = 0; n < d_state; n++) {
                    unsigned int si = head * d_state * headdim + n * headdim + p;
                    yp += h_all[(batch_idx * nf + t + 1) * frame + si] * cv[n];
                }
                if (has_d_skip) yp += d_skip[head] * xv;
                dz_out[oi] += go_val * yp * silu_grad_bwd(zv);
                dy_pre = go_val * sz;
            } else {
                dy_pre = go_val;
            }

            // --- D skip gradient ---
            float dx_acc = 0.0f;
            if (has_d_skip) {
                dx_acc += dy_pre * d_skip[head];
            }

            // --- Main gradient loop over d_state ---
            float d_dtv = 0.0f;

            for (unsigned int n = 0; n < d_state; n++) {
                unsigned int si = head * d_state * headdim + n * headdim + p;

                // Load saved states
                float ht  = h_all[(batch_idx * nf + t + 1) * frame + si];
                float hp  = h_all[(batch_idx * nf + t    ) * frame + si];
                float pbx = bx_all[(batch_idx * nf + t   ) * frame + si];
                float cur_bx = bv[n] * xv;

                // dC: d(loss)/d(C[n]) += dy_pre * h_t[n,p]
                unsigned int ci = batch_idx * seq_len * ngroups * d_state
                                + t * ngroups * d_state + group * d_state + n;
                atomicAdd(&d_c_out[ci], dy_pre * ht);

                // dh from output: y += h_t[n,p] * C[n]
                dh[n] += dy_pre * cv[n];

                // Consume d_cur_bx from future timestep (trapezoidal)
                float d_cbx_val = d_cbx[n];
                unsigned int bi_idx = ci;  // same index as d_b
                dx_acc += d_cbx_val * bv[n];
                atomicAdd(&d_b_out[bi_idx], d_cbx_val * xv);

                // Recurrence backward
                float dh_t = dh[n];

                // d(h_{t-1}) += dh_t * da
                float dh_prop = dh_t * da;

                // d(da) = dh_t * h_{t-1}; da = exp(dtv * a_real)
                // d(dtv) += dh_t * h_{t-1} * da * a_real
                d_dtv += dh_t * hp * da * a_real_h;

                // d(beta * B[n] * x[p])
                dx_acc += dh_t * beta * bv[n];
                atomicAdd(&d_b_out[bi_idx], dh_t * beta * xv);
                // d(beta) = dh_t * cur_bx; beta = dtv * alpha
                d_dtv += dh_t * cur_bx * alpha;

                // d(gamma * prev_bx[n,p])
                d_cbx[n] = dh_t * gam;
                // d(gamma) = dh_t * prev_bx; gamma = dtv * (1-alpha)
                d_dtv += dh_t * pbx * (1.0f - alpha);

                // Set dh for next reverse iteration (timestep t-1)
                dh[n] = dh_prop;
            }

            // Write dx (unique per (batch, t, head, p) — no atomic needed)
            dx[oi] = dx_acc;

            // d_dt with softplus chain rule
            float dtg = dt_softplus ? d_dtv * softplus_grad_bwd(biased_dt) : d_dtv;
            // d_dt is unique per (batch, t, head) but multiple p threads contribute
            // to the same location, so we need atomicAdd here
            atomicAdd(&d_dt_out[batch_idx * seq_len * nheads + t * nheads + head], dtg);
        }
    }
}

} // extern "C"
