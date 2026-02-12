// Fused RMSNorm + Linear Projection kernel for Kore.
//
// Standard path: RMSNorm(x) → write VRAM → read VRAM → MatMul(W)
// Fused path:    Load x → RMSNorm in registers → MatMul(W) in SRAM → write result
//
// Eliminates one full activation read+write (2 * batch * hidden_size * sizeof(float)).
// One block per output row. Tiled along output dim for large projections.

extern "C" {

// ============================================================================
// Warp-level reductions
// ============================================================================

__device__ float warp_reduce_sum_fnp(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// Fused RMSNorm + Linear: y[row, :] = (x[row, :] / rms * gamma) @ W^T
//
// input:  [rows, hidden]  — activations
// gamma:  [hidden]        — RMSNorm scale
// weight: [out_dim, hidden] — projection weight (stored row-major, output-dim-major)
// output: [rows, out_dim] — projected output
// bias:   [out_dim]       — optional bias (nullptr if unused)
//
// Grid:  (rows, ceil(out_dim / TILE_OUT), 1)
// Block: (BLOCK_SIZE, 1, 1) where BLOCK_SIZE = 256
//
// Strategy:
//   Phase 1: All threads in the block cooperatively compute inv_rms for this row.
//   Phase 2: Each block-column tile computes a slice of the output projection.
//            For each output element, threads accumulate dot(x_hat, W[j,:])
//            over the hidden dimension in a strided loop.
// ============================================================================

#define FNP_BLOCK 256

__global__ void fused_rms_norm_proj_f32(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    unsigned int rows,
    unsigned int hidden,
    unsigned int out_dim,
    float eps
) {
    __shared__ float s_reduce[32]; // warp reduction scratch

    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    unsigned int tid = threadIdx.x;
    const float* x_row = input + row * hidden;

    // ---- Phase 1: compute inv_rms for this row ----
    float ss = 0.0f;
    for (unsigned int i = tid; i < hidden; i += FNP_BLOCK) {
        float v = x_row[i];
        ss += v * v;
    }
    ss = warp_reduce_sum_fnp(ss);

    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;
    if (lane == 0) s_reduce[warp_id] = ss;
    __syncthreads();

    unsigned int num_warps = FNP_BLOCK / 32;
    if (warp_id == 0) {
        ss = (lane < num_warps) ? s_reduce[lane] : 0.0f;
        ss = warp_reduce_sum_fnp(ss);
    }
    if (tid == 0) s_reduce[0] = ss;
    __syncthreads();
    ss = s_reduce[0];

    float inv_rms = rsqrtf(ss / (float)hidden + eps);

    // ---- Phase 2: fused norm + projection ----
    // Each block handles a tile of output columns.
    // blockIdx.y selects which tile of out_dim we compute.
    unsigned int out_tile_start = blockIdx.y * FNP_BLOCK;

    // Each thread computes one output element within the tile.
    unsigned int j = out_tile_start + tid;
    if (j >= out_dim) return;

    const float* w_row = weight + j * hidden; // W[j, :]

    float acc = 0.0f;
    for (unsigned int i = 0; i < hidden; i++) {
        float x_hat = x_row[i] * inv_rms * gamma[i];
        acc += x_hat * w_row[i];
    }

    if (bias != nullptr) {
        acc += bias[j];
    }

    output[row * out_dim + j] = acc;
}

// ============================================================================
// Variant with shared memory caching of x_hat for better reuse.
// Used when out_dim >> hidden (common for MLP up-projections).
//
// Grid:  (rows, 1, 1)
// Block: (BLOCK_SIZE, 1, 1)
//
// Each thread loops over output columns, reusing x_hat from shared memory.
// Requires hidden <= 8192 (32KB shared memory at 4 bytes/float).
// ============================================================================

__global__ void fused_rms_norm_proj_smem_f32(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    unsigned int rows,
    unsigned int hidden,
    unsigned int out_dim,
    float eps
) {
    extern __shared__ float smem[]; // dynamic shared memory: [hidden] floats for x_hat

    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;
    const float* x_row = input + row * hidden;

    // ---- Phase 1: compute inv_rms ----
    // Use first 32 floats of smem as reduction scratch
    float* s_reduce = smem;

    float ss = 0.0f;
    for (unsigned int i = tid; i < hidden; i += block_size) {
        float v = x_row[i];
        ss += v * v;
    }
    ss = warp_reduce_sum_fnp(ss);

    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;
    if (lane == 0) s_reduce[warp_id] = ss;
    __syncthreads();

    unsigned int num_warps = block_size / 32;
    if (warp_id == 0) {
        ss = (lane < num_warps) ? s_reduce[lane] : 0.0f;
        ss = warp_reduce_sum_fnp(ss);
    }
    if (tid == 0) s_reduce[0] = ss;
    __syncthreads();
    ss = s_reduce[0];
    float inv_rms = rsqrtf(ss / (float)hidden + eps);

    // ---- Phase 2: compute x_hat and store in shared memory ----
    float* x_hat = smem; // reuse smem (reduction scratch only needed 32 floats, x_hat needs hidden)
    for (unsigned int i = tid; i < hidden; i += block_size) {
        x_hat[i] = x_row[i] * inv_rms * gamma[i];
    }
    __syncthreads();

    // ---- Phase 3: project using cached x_hat ----
    float* out_row = output + row * out_dim;
    for (unsigned int j = tid; j < out_dim; j += block_size) {
        const float* w_row = weight + j * hidden;
        float acc = 0.0f;
        for (unsigned int i = 0; i < hidden; i++) {
            acc += x_hat[i] * w_row[i];
        }
        if (bias != nullptr) {
            acc += bias[j];
        }
        out_row[j] = acc;
    }
}

// ============================================================================
// Standalone fused RMSNorm (improved version with SiLU gate option)
// For use in Mamba input projection: y = RMSNorm(x) * SiLU(z)
// ============================================================================

__global__ void fused_rms_norm_silu_gate_f32(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ gate,
    float* __restrict__ output,
    unsigned int rows,
    unsigned int cols,
    float eps
) {
    __shared__ float s_reduce[32];

    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    unsigned int tid = threadIdx.x;
    const float* x_row = input + row * cols;
    const float* z_row = gate + row * cols;
    float* out_row = output + row * cols;

    // Compute inv_rms
    float ss = 0.0f;
    for (unsigned int c = tid; c < cols; c += blockDim.x) {
        float v = x_row[c];
        ss += v * v;
    }
    ss = warp_reduce_sum_fnp(ss);

    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;
    if (lane == 0) s_reduce[warp_id] = ss;
    __syncthreads();

    unsigned int num_warps = blockDim.x / 32;
    if (warp_id == 0) {
        ss = (lane < num_warps) ? s_reduce[lane] : 0.0f;
        ss = warp_reduce_sum_fnp(ss);
    }
    if (tid == 0) s_reduce[0] = ss;
    __syncthreads();
    ss = s_reduce[0];
    float inv_rms = rsqrtf(ss / (float)cols + eps);

    // Fused normalize + SiLU gate
    for (unsigned int c = tid; c < cols; c += blockDim.x) {
        float x_hat = x_row[c] * inv_rms * gamma[c];
        float z = z_row[c];
        float silu_z = z / (1.0f + expf(-z));
        out_row[c] = x_hat * silu_z;
    }
}

} // extern "C"
