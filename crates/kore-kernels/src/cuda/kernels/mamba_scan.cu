// Flash Scan: Chunked parallel Mamba-3 SSM scan on GPU.
//
// Thread ownership: each thread owns headdim positions p = tid, tid+blockDim, ...
// For each p, the thread maintains ALL d_state elements of h[n,p] and prev_bx[n,p]
// in registers, computes full y = sum(C[n]*h[n,p]), and writes ONCE per timestep.
// No atomicAdd, no inter-thread races.
//
// Trapezoidal: h_t = da * h_{t-1} + beta * B_t*x_t + gamma * prev_bx
// Output: y_t = sum_n(C_t[n] * h_t[n,p])

extern "C" {

#define SCAN_CHUNK 128
#define MAX_DSTATE 64  // max supported d_state; assert in Rust dispatch

__device__ float softplus_scan(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return logf(1.0f + expf(x));
}

__device__ float silu_scan(float x) {
    return x / (1.0f + expf(-x));
}

// Phase 1: Intra-chunk sequential scan.
// Grid: (batch * nheads * num_chunks, 1, 1)
// Block: (min(headdim, 256), 1, 1)
__global__ void mamba3_scan_chunk_f32(
    const float* __restrict__ x,          // [B, T, H, D]
    const float* __restrict__ dt,         // [B, T, H]
    const float* __restrict__ a_real,     // [H]
    const float* __restrict__ b_in,       // [B, T, G, N]
    const float* __restrict__ c_in,       // [B, T, G, N]
    unsigned int has_dt_bias,
    const float* __restrict__ dt_bias,    // [H]
    unsigned int has_z,
    const float* __restrict__ z,          // [B, T, H, D]
    unsigned int has_d_skip,
    const float* __restrict__ d_skip,     // [H]
    float* __restrict__ output,           // [B, T, H, D]
    float* __restrict__ chunk_last_h,     // [B, H, num_chunks, N, D]
    float* __restrict__ chunk_last_bx,    // [B, H, num_chunks, N, D]
    unsigned int batch,
    unsigned int seq_len,
    unsigned int nheads,
    unsigned int headdim,
    unsigned int ngroups,
    unsigned int d_state,
    unsigned int num_chunks,
    float alpha,
    unsigned int dt_softplus
) {
    unsigned int block_id = blockIdx.x;
    unsigned int chunk_idx = block_id % num_chunks;
    unsigned int temp = block_id / num_chunks;
    unsigned int head = temp % nheads;
    unsigned int batch_idx = temp / nheads;
    if (batch_idx >= batch) return;

    unsigned int tid = threadIdx.x;
    unsigned int group = head / (nheads / ngroups);
    unsigned int chunk_start = chunk_idx * SCAN_CHUNK;
    unsigned int chunk_end = chunk_start + SCAN_CHUNK;
    if (chunk_end > seq_len) chunk_end = seq_len;
    unsigned int state_size = d_state * headdim;

    // Each thread processes headdim positions p = tid, tid+blockDim.x, ...
    // For each p, maintain d_state floats of state — no cross-thread sharing needed.
    for (unsigned int p = tid; p < headdim; p += blockDim.x) {
        float h[MAX_DSTATE];
        float pbx[MAX_DSTATE];
        for (unsigned int n = 0; n < d_state; n++) {
            h[n] = 0.0f;
            pbx[n] = 0.0f;
        }

        for (unsigned int t = chunk_start; t < chunk_end; t++) {
            float dt_val = dt[batch_idx * seq_len * nheads + t * nheads + head];
            if (has_dt_bias) dt_val += dt_bias[head];
            if (dt_softplus) dt_val = softplus_scan(dt_val);

            float da = expf(dt_val * a_real[head]);
            float beta = dt_val * alpha;
            float gam = dt_val * (1.0f - alpha);

            float x_val = x[batch_idx * seq_len * nheads * headdim
                           + t * nheads * headdim + head * headdim + p];

            float y = 0.0f;
            for (unsigned int n = 0; n < d_state; n++) {
                unsigned int bc_idx = batch_idx * seq_len * ngroups * d_state
                                    + t * ngroups * d_state + group * d_state + n;
                float b_val = b_in[bc_idx];
                float c_val = c_in[bc_idx];

                float cur_bx = b_val * x_val;
                h[n] = da * h[n] + beta * cur_bx + gam * pbx[n];
                pbx[n] = cur_bx;
                y += c_val * h[n];
            }

            // Skip connection: y += D[head] * x
            if (has_d_skip) y += d_skip[head] * x_val;

            // SiLU gating: y *= silu(z)
            if (has_z) {
                float z_val = z[batch_idx * seq_len * nheads * headdim
                               + t * nheads * headdim + head * headdim + p];
                y *= silu_scan(z_val);
            }

            // Single write per (t, head, p) — no atomics, no races
            output[batch_idx * seq_len * nheads * headdim
                  + t * nheads * headdim + head * headdim + p] = y;
        }

        // Store boundary states for inter-chunk prefix
        for (unsigned int n = 0; n < d_state; n++) {
            unsigned int idx = batch_idx * nheads * num_chunks * state_size
                             + head * num_chunks * state_size
                             + chunk_idx * state_size + n * headdim + p;
            chunk_last_h[idx] = h[n];
            chunk_last_bx[idx] = pbx[n];
        }
    }
}

// Phase 2: Inter-chunk prefix (boundary state propagation).
// Grid: (batch * nheads, 1, 1), Block: (256, 1, 1)
__global__ void mamba3_scan_prefix_f32(
    float* __restrict__ chunk_last_h,   // [B, H, num_chunks, N*D]
    float* __restrict__ chunk_last_bx,  // [B, H, num_chunks, N*D]
    const float* __restrict__ a_real,   // [H]
    const float* __restrict__ dt,       // [B, T, H]
    unsigned int has_dt_bias,
    const float* __restrict__ dt_bias,  // [H]
    unsigned int batch,
    unsigned int seq_len,
    unsigned int nheads,
    unsigned int state_size,
    unsigned int num_chunks,
    float alpha,
    unsigned int dt_softplus
) {
    unsigned int block_id = blockIdx.x;
    unsigned int head = block_id % nheads;
    unsigned int batch_idx = block_id / nheads;
    if (batch_idx >= batch) return;

    unsigned int tid = threadIdx.x;

    for (unsigned int c = 1; c < num_chunks; c++) {
        unsigned int t_boundary = c * SCAN_CHUNK;
        if (t_boundary >= seq_len) break;

        float dt_val = dt[batch_idx * seq_len * nheads + t_boundary * nheads + head];
        if (has_dt_bias) dt_val += dt_bias[head];
        if (dt_softplus) dt_val = softplus_scan(dt_val);
        float da = expf(dt_val * a_real[head]);

        unsigned int prev_base = batch_idx * nheads * num_chunks * state_size
                               + head * num_chunks * state_size
                               + (c - 1) * state_size;
        unsigned int curr_base = batch_idx * nheads * num_chunks * state_size
                               + head * num_chunks * state_size
                               + c * state_size;

        for (unsigned int si = tid; si < state_size; si += blockDim.x) {
            chunk_last_h[curr_base + si] = da * chunk_last_h[prev_base + si]
                                         + chunk_last_h[curr_base + si];
        }
        __syncthreads();
    }
}

} // extern "C"
