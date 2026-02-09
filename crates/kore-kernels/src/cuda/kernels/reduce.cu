// Reduction kernels for Kore: sum, mean, max, min
// Uses warp-shuffle reductions for maximum throughput.
// Two-pass: per-block partial reduce → final reduce for large tensors.

extern "C" {

// Warp-level reduction using shuffle instructions
__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float warp_reduce_min(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block-level sum reduction
__global__ void reduce_sum_f32(const float* input, float* output, unsigned int n) {
    __shared__ float shared[32]; // one per warp

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Grid-stride accumulation
    float sum = 0.0f;
    for (unsigned int i = idx; i < n; i += stride) {
        sum += input[i];
    }

    // Warp reduce
    sum = warp_reduce_sum(sum);

    // Write warp results to shared memory
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;
    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    // Final reduce across warps (first warp only)
    unsigned int num_warps = blockDim.x / 32;
    if (warp_id == 0) {
        sum = (lane < num_warps) ? shared[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            atomicAdd(output, sum);
        }
    }
}

// Block-level max reduction
__global__ void reduce_max_f32(const float* input, float* output, unsigned int n) {
    __shared__ float shared[32];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    float val = -INFINITY;
    for (unsigned int i = idx; i < n; i += stride) {
        val = fmaxf(val, input[i]);
    }

    val = warp_reduce_max(val);

    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    unsigned int num_warps = blockDim.x / 32;
    if (warp_id == 0) {
        val = (lane < num_warps) ? shared[lane] : -INFINITY;
        val = warp_reduce_max(val);
        if (lane == 0) {
            // Atomic max for float — use atomicCAS trick
            int* addr = (int*)output;
            int old = *addr, assumed;
            do {
                assumed = old;
                old = atomicCAS(addr, assumed,
                    __float_as_int(fmaxf(val, __int_as_float(assumed))));
            } while (assumed != old);
        }
    }
}

// Row-wise sum: output[row] = sum(input[row, :])
// input: [rows, cols], output: [rows]
__global__ void reduce_sum_rows_f32(
    const float* input, float* output,
    unsigned int rows, unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    __shared__ float shared[32];
    unsigned int tid = threadIdx.x;

    float sum = 0.0f;
    for (unsigned int c = tid; c < cols; c += blockDim.x) {
        sum += input[row * cols + c];
    }

    sum = warp_reduce_sum(sum);

    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;
    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    unsigned int num_warps = blockDim.x / 32;
    if (warp_id == 0) {
        sum = (lane < num_warps) ? shared[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            output[row] = sum;
        }
    }
}

// Row-wise max: output[row] = max(input[row, :])
__global__ void reduce_max_rows_f32(
    const float* input, float* output,
    unsigned int rows, unsigned int cols
) {
    unsigned int row = blockIdx.x;
    if (row >= rows) return;

    __shared__ float shared[32];
    unsigned int tid = threadIdx.x;

    float val = -INFINITY;
    for (unsigned int c = tid; c < cols; c += blockDim.x) {
        val = fmaxf(val, input[row * cols + c]);
    }

    val = warp_reduce_max(val);

    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    unsigned int num_warps = blockDim.x / 32;
    if (warp_id == 0) {
        val = (lane < num_warps) ? shared[lane] : -INFINITY;
        val = warp_reduce_max(val);
        if (lane == 0) {
            output[row] = val;
        }
    }
}

} // extern "C"
