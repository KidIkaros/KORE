// On-the-fly dequantization matmul for Kore.
//
// Accepts packed 2-bit quaternary weights ({-3,-1,+1,+3}) and f32 activations.
// Unpacks weights to f32 in registers before multiplying — never materializes
// the full f32 weight matrix in VRAM.
//
// Memory bandwidth savings: 16x vs f32 weights, 4x vs int8.
// Compatible with kore-btes quaternary packing (4 values per byte, 2 bits each).

extern "C" {

// Quaternary decode LUT: 2-bit index → float value
// Matches kore-btes Quat encoding: 0→-3, 1→-1, 2→+1, 3→+3
__device__ __constant__ float QUAT_LUT[4] = {-3.0f, -1.0f, 1.0f, 3.0f};

// ============================================================================
// Tiled GEMM with on-the-fly quaternary dequantization
//
// C[M, N] = dequant(A_packed[M, K_packed]) @ B[K, N]
//
// A_packed: [M, K_packed] packed uint8, 4 quat values per byte
// A_scales: [M] per-row scale factors
// B:        [K, N] dense f32 activations
// C:        [M, N] f32 output
//
// Grid:  (ceil(N/TILE_N), ceil(M/TILE_M), 1)
// Block: (TILE_N, TILE_M, 1) = (32, 32) = 1024 threads
// ============================================================================

#define DQ_TILE_M 32
#define DQ_TILE_N 32
#define DQ_TILE_K 32

__global__ void dequant_quat_matmul_f32(
    const unsigned char* __restrict__ A_packed, // [M, K_packed]
    const float* __restrict__ A_scales,          // [M]
    const float* __restrict__ B,                 // [K, N]
    float* __restrict__ C,                       // [M, N]
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int K_packed                        // = ceil(K/4)
) {
    __shared__ float As[DQ_TILE_M][DQ_TILE_K]; // dequantized A tile
    __shared__ float Bs[DQ_TILE_K][DQ_TILE_N]; // B tile

    unsigned int row = blockIdx.y * DQ_TILE_M + threadIdx.y;
    unsigned int col = blockIdx.x * DQ_TILE_N + threadIdx.x;

    float acc = 0.0f;

    // Loop over K-dimension tiles
    for (unsigned int t = 0; t < (K + DQ_TILE_K - 1) / DQ_TILE_K; t++) {
        // --- Load and dequantize A tile ---
        // Each thread loads one element of As[threadIdx.y][threadIdx.x]
        unsigned int a_k = t * DQ_TILE_K + threadIdx.x;
        if (row < M && a_k < K) {
            // Find which packed byte and which 2-bit slot
            unsigned int byte_idx = a_k / 4;
            unsigned int bit_slot = a_k % 4;
            unsigned char packed = A_packed[row * K_packed + byte_idx];
            unsigned int qidx = (packed >> (bit_slot * 2)) & 0x3;
            As[threadIdx.y][threadIdx.x] = QUAT_LUT[qidx];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // --- Load B tile (standard coalesced) ---
        unsigned int b_k = t * DQ_TILE_K + threadIdx.y;
        if (b_k < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_k * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // --- Tile GEMM ---
        #pragma unroll
        for (unsigned int k = 0; k < DQ_TILE_K; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Apply per-row scale and write output
    if (row < M && col < N) {
        C[row * N + col] = acc * A_scales[row];
    }
}

// ============================================================================
// Tiled 2x2 variant for larger matrices (higher arithmetic intensity)
//
// Each thread computes a 2x2 block of C.
// Block: (32, 32) = 1024 threads, tile: A[64][16], B[16][64]
// ============================================================================

#define DQ2_TILE_M 64
#define DQ2_TILE_N 64
#define DQ2_TILE_K 16
#define DQ2_THREAD_M 2
#define DQ2_THREAD_N 2
#define DQ2_BLOCK_X 32
#define DQ2_BLOCK_Y 32

__global__ void dequant_quat_matmul_tiled2x2_f32(
    const unsigned char* __restrict__ A_packed,
    const float* __restrict__ A_scales,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int K_packed
) {
    __shared__ float As[DQ2_TILE_M][DQ2_TILE_K];
    __shared__ float Bs[DQ2_TILE_K][DQ2_TILE_N];

    unsigned int row0 = blockIdx.y * DQ2_TILE_M + threadIdx.y * DQ2_THREAD_M;
    unsigned int col0 = blockIdx.x * DQ2_TILE_N + threadIdx.x * DQ2_THREAD_N;

    unsigned int tid = threadIdx.y * DQ2_BLOCK_X + threadIdx.x; // 0..1023

    float acc[DQ2_THREAD_M][DQ2_THREAD_N] = {{0.0f}};

    for (unsigned int t = 0; t < (K + DQ2_TILE_K - 1) / DQ2_TILE_K; t++) {
        // Collaborative load of As[64][16]: 1024 elements, 1 per thread
        {
            unsigned int a_smem_row = tid / DQ2_TILE_K;        // 0..63
            unsigned int a_smem_col = tid % DQ2_TILE_K;        // 0..15
            unsigned int a_global_row = blockIdx.y * DQ2_TILE_M + a_smem_row;
            unsigned int a_global_col = t * DQ2_TILE_K + a_smem_col;

            if (a_global_row < M && a_global_col < K) {
                unsigned int byte_idx = a_global_col / 4;
                unsigned int bit_slot = a_global_col % 4;
                unsigned char packed = A_packed[a_global_row * K_packed + byte_idx];
                unsigned int qidx = (packed >> (bit_slot * 2)) & 0x3;
                As[a_smem_row][a_smem_col] = QUAT_LUT[qidx];
            } else {
                As[a_smem_row][a_smem_col] = 0.0f;
            }
        }

        // Collaborative load of Bs[16][64]: 1024 elements, 1 per thread
        {
            unsigned int b_smem_row = tid / DQ2_TILE_N;        // 0..15
            unsigned int b_smem_col = tid % DQ2_TILE_N;        // 0..63
            unsigned int b_global_row = t * DQ2_TILE_K + b_smem_row;
            unsigned int b_global_col = blockIdx.x * DQ2_TILE_N + b_smem_col;
            if (b_global_row < K && b_global_col < N) {
                Bs[b_smem_row][b_smem_col] = B[b_global_row * N + b_global_col];
            } else {
                Bs[b_smem_row][b_smem_col] = 0.0f;
            }
        }

        __syncthreads();

        // Compute 2x2 output per thread
        #pragma unroll
        for (unsigned int k = 0; k < DQ2_TILE_K; k++) {
            float a_vals[DQ2_THREAD_M];
            float b_vals[DQ2_THREAD_N];

            #pragma unroll
            for (int m = 0; m < DQ2_THREAD_M; m++) {
                a_vals[m] = As[threadIdx.y * DQ2_THREAD_M + m][k];
            }
            #pragma unroll
            for (int n = 0; n < DQ2_THREAD_N; n++) {
                b_vals[n] = Bs[k][threadIdx.x * DQ2_THREAD_N + n];
            }

            #pragma unroll
            for (int m = 0; m < DQ2_THREAD_M; m++) {
                #pragma unroll
                for (int n = 0; n < DQ2_THREAD_N; n++) {
                    acc[m][n] += a_vals[m] * b_vals[n];
                }
            }
        }

        __syncthreads();
    }

    // Write 2x2 result with per-row scale
    #pragma unroll
    for (int m = 0; m < DQ2_THREAD_M; m++) {
        #pragma unroll
        for (int n = 0; n < DQ2_THREAD_N; n++) {
            unsigned int out_row = row0 + m;
            unsigned int out_col = col0 + n;
            if (out_row < M && out_col < N) {
                C[out_row * N + out_col] = acc[m][n] * A_scales[out_row];
            }
        }
    }
}

// ============================================================================
// Fused dequant matmul + bias + ReLU
// ============================================================================

__global__ void dequant_quat_matmul_bias_relu_f32(
    const unsigned char* __restrict__ A_packed,
    const float* __restrict__ A_scales,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int K_packed
) {
    __shared__ float As[DQ_TILE_M][DQ_TILE_K];
    __shared__ float Bs[DQ_TILE_K][DQ_TILE_N];

    unsigned int row = blockIdx.y * DQ_TILE_M + threadIdx.y;
    unsigned int col = blockIdx.x * DQ_TILE_N + threadIdx.x;

    float acc = 0.0f;

    for (unsigned int t = 0; t < (K + DQ_TILE_K - 1) / DQ_TILE_K; t++) {
        unsigned int a_k = t * DQ_TILE_K + threadIdx.x;
        if (row < M && a_k < K) {
            unsigned int byte_idx = a_k / 4;
            unsigned int bit_slot = a_k % 4;
            unsigned char packed = A_packed[row * K_packed + byte_idx];
            unsigned int qidx = (packed >> (bit_slot * 2)) & 0x3;
            As[threadIdx.y][threadIdx.x] = QUAT_LUT[qidx];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        unsigned int b_k = t * DQ_TILE_K + threadIdx.y;
        if (b_k < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_k * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (unsigned int k = 0; k < DQ_TILE_K; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float val = acc * A_scales[row] + bias[col];
        C[row * N + col] = fmaxf(val, 0.0f);
    }
}

} // extern "C"
