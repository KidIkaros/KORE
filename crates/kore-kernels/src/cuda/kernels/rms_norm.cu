// Fused RMS Normalization kernel for Kore.
// output[i] = (input[i] / sqrt(mean(input^2) + eps)) * weight[i]
// One block per row, warp-shuffle for sum-of-squares reduction.

extern "C" {

__device__ float warp_reduce_sum_rn(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// RMS Norm: output[row,:] = (input[row,:] / rms) * weight[:]
// input: [rows, cols], weight: [cols], output: [rows, cols]
__global__ void rms_norm_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    unsigned int rows,
    unsigned int cols,
    float eps
) {
    __shared__ float shared[32];

    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    unsigned int tid = threadIdx.x;
    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    // Compute sum of squares
    float ss = 0.0f;
    for (unsigned int c = tid; c < cols; c += blockDim.x) {
        float val = row_in[c];
        ss += val * val;
    }
    ss = warp_reduce_sum_rn(ss);

    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;
    if (lane == 0) shared[warp_id] = ss;
    __syncthreads();

    unsigned int num_warps = blockDim.x / 32;
    if (warp_id == 0) {
        ss = (lane < num_warps) ? shared[lane] : 0.0f;
        ss = warp_reduce_sum_rn(ss);
    }
    if (tid == 0) shared[0] = ss;
    __syncthreads();
    ss = shared[0];

    // Compute 1/rms
    float inv_rms = rsqrtf(ss / (float)cols + eps);

    // Normalize and scale
    for (unsigned int c = tid; c < cols; c += blockDim.x) {
        row_out[c] = row_in[c] * inv_rms * weight[c];
    }
}

// In-place RMS Norm (for inference — avoids extra allocation)
__global__ void rms_norm_inplace_f32(
    float* __restrict__ data,
    const float* __restrict__ weight,
    unsigned int rows,
    unsigned int cols,
    float eps
) {
    __shared__ float shared[32];

    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    unsigned int tid = threadIdx.x;
    float* row_data = data + row * cols;

    float ss = 0.0f;
    for (unsigned int c = tid; c < cols; c += blockDim.x) {
        float val = row_data[c];
        ss += val * val;
    }
    ss = warp_reduce_sum_rn(ss);

    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;
    if (lane == 0) shared[warp_id] = ss;
    __syncthreads();

    unsigned int num_warps = blockDim.x / 32;
    if (warp_id == 0) {
        ss = (lane < num_warps) ? shared[lane] : 0.0f;
        ss = warp_reduce_sum_rn(ss);
    }
    if (tid == 0) shared[0] = ss;
    __syncthreads();
    ss = shared[0];

    float inv_rms = rsqrtf(ss / (float)cols + eps);

    for (unsigned int c = tid; c < cols; c += blockDim.x) {
        row_data[c] = row_data[c] * inv_rms * weight[c];
    }
}

} // extern "C"
