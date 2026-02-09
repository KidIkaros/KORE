// Fused numerically-stable softmax kernel for Kore.
// Single kernel: max-subtract, exp, sum, normalize.
// One block per row, warp-shuffle for reductions.
// Handles rows up to 32K elements.

extern "C" {

__device__ float warp_reduce_sum_sm(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float warp_reduce_max_sm(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Fused softmax: output[row, :] = softmax(input[row, :])
// One block per row. blockDim.x should be 256 or 512.
__global__ void softmax_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned int rows,
    unsigned int cols
) {
    __shared__ float shared[32]; // one per warp

    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    unsigned int tid = threadIdx.x;
    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    // Pass 1: find max for numerical stability
    float max_val = -INFINITY;
    for (unsigned int c = tid; c < cols; c += blockDim.x) {
        max_val = fmaxf(max_val, row_in[c]);
    }
    max_val = warp_reduce_max_sm(max_val);

    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;
    if (lane == 0) shared[warp_id] = max_val;
    __syncthreads();

    unsigned int num_warps = blockDim.x / 32;
    if (warp_id == 0) {
        max_val = (lane < num_warps) ? shared[lane] : -INFINITY;
        max_val = warp_reduce_max_sm(max_val);
    }
    // Broadcast max to all threads
    if (tid == 0) shared[0] = max_val;
    __syncthreads();
    max_val = shared[0];

    // Pass 2: compute exp(x - max) and sum
    float sum = 0.0f;
    for (unsigned int c = tid; c < cols; c += blockDim.x) {
        float val = expf(row_in[c] - max_val);
        row_out[c] = val; // store intermediate
        sum += val;
    }
    sum = warp_reduce_sum_sm(sum);

    if (lane == 0) shared[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < num_warps) ? shared[lane] : 0.0f;
        sum = warp_reduce_sum_sm(sum);
    }
    if (tid == 0) shared[0] = sum;
    __syncthreads();
    sum = shared[0];

    // Pass 3: normalize
    float inv_sum = 1.0f / sum;
    for (unsigned int c = tid; c < cols; c += blockDim.x) {
        row_out[c] *= inv_sum;
    }
}

// Causal masked softmax: applies -inf mask for positions > current
// input: [rows, cols], mask applied per row where col > row_offset + row_within_batch
__global__ void softmax_causal_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned int rows,
    unsigned int cols,
    unsigned int seq_offset
) {
    __shared__ float shared[32];

    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    unsigned int tid = threadIdx.x;
    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;
    unsigned int causal_limit = seq_offset + row + 1;

    // Pass 1: max with causal mask
    float max_val = -INFINITY;
    for (unsigned int c = tid; c < cols; c += blockDim.x) {
        float val = (c < causal_limit) ? row_in[c] : -INFINITY;
        max_val = fmaxf(max_val, val);
    }
    max_val = warp_reduce_max_sm(max_val);

    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;
    if (lane == 0) shared[warp_id] = max_val;
    __syncthreads();

    unsigned int num_warps = blockDim.x / 32;
    if (warp_id == 0) {
        max_val = (lane < num_warps) ? shared[lane] : -INFINITY;
        max_val = warp_reduce_max_sm(max_val);
    }
    if (tid == 0) shared[0] = max_val;
    __syncthreads();
    max_val = shared[0];

    // Pass 2: exp + sum
    float sum = 0.0f;
    for (unsigned int c = tid; c < cols; c += blockDim.x) {
        float val = (c < causal_limit) ? expf(row_in[c] - max_val) : 0.0f;
        row_out[c] = val;
        sum += val;
    }
    sum = warp_reduce_sum_sm(sum);

    if (lane == 0) shared[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < num_warps) ? shared[lane] : 0.0f;
        sum = warp_reduce_sum_sm(sum);
    }
    if (tid == 0) shared[0] = sum;
    __syncthreads();
    sum = shared[0];

    // Pass 3: normalize
    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (unsigned int c = tid; c < cols; c += blockDim.x) {
        row_out[c] *= inv_sum;
    }
}

} // extern "C"
