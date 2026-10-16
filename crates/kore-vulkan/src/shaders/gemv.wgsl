struct GemmUniforms {
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    beta: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: GemmUniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

// Shared memory cache for up to 4 rows of A[0..M, 0..64]
var<workgroup> shared_a: array<f32, 256>;

// Batch-Invariant Small-Batch GEMV: C[M, N] = A[M, K] @ B[K, N] for M in 1..4
// Reads weight matrix B only ONCE from VRAM across up to 4 batch tokens (e.g. MTP verification)
// while keeping an identical K-reduction order regardless of batch size M.
@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let col = global_id.x;
    let lx = local_id.x;
    let m_len = params.m;
    let k_len = params.k;
    let n_len = params.n;

    var sum0: f32 = 0.0;
    var sum1: f32 = 0.0;
    var sum2: f32 = 0.0;
    var sum3: f32 = 0.0;
    let num_tiles = (k_len + 63u) / 64u;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_idx = t * 64u + lx;
        if (a_idx < k_len) {
            shared_a[lx] = A[a_idx];
            if (m_len > 1u) { shared_a[64u + lx] = A[k_len + a_idx]; }
            if (m_len > 2u) { shared_a[128u + lx] = A[2u * k_len + a_idx]; }
            if (m_len > 3u) { shared_a[192u + lx] = A[3u * k_len + a_idx]; }
        } else {
            shared_a[lx] = 0.0;
            shared_a[64u + lx] = 0.0;
            shared_a[128u + lx] = 0.0;
            shared_a[192u + lx] = 0.0;
        }
        workgroupBarrier();

        if (col < n_len) {
            let tile_k_end = min(64u, k_len - t * 64u);
            let base_k = t * 64u;
            var k: u32 = 0u;
            let k_unroll = (tile_k_end / 4u) * 4u;
            for (; k < k_unroll; k = k + 4u) {
                let w0 = B[(base_k + k) * n_len + col];
                let w1 = B[(base_k + k + 1u) * n_len + col];
                let w2 = B[(base_k + k + 2u) * n_len + col];
                let w3 = B[(base_k + k + 3u) * n_len + col];

                sum0 = sum0 + shared_a[k] * w0 + shared_a[k + 1u] * w1 + shared_a[k + 2u] * w2 + shared_a[k + 3u] * w3;
                if (m_len > 1u) {
                    sum1 = sum1 + shared_a[64u + k] * w0 + shared_a[64u + k + 1u] * w1 + shared_a[64u + k + 2u] * w2 + shared_a[64u + k + 3u] * w3;
                }
                if (m_len > 2u) {
                    sum2 = sum2 + shared_a[128u + k] * w0 + shared_a[128u + k + 1u] * w1 + shared_a[128u + k + 2u] * w2 + shared_a[128u + k + 3u] * w3;
                }
                if (m_len > 3u) {
                    sum3 = sum3 + shared_a[192u + k] * w0 + shared_a[192u + k + 1u] * w1 + shared_a[192u + k + 2u] * w2 + shared_a[192u + k + 3u] * w3;
                }
            }
            for (; k < tile_k_end; k = k + 1u) {
                let w = B[(base_k + k) * n_len + col];
                sum0 = sum0 + shared_a[k] * w;
                if (m_len > 1u) { sum1 = sum1 + shared_a[64u + k] * w; }
                if (m_len > 2u) { sum2 = sum2 + shared_a[128u + k] * w; }
                if (m_len > 3u) { sum3 = sum3 + shared_a[192u + k] * w; }
            }
        }
        workgroupBarrier();
    }

    if (col < n_len) {
        let alpha = params.alpha;
        let beta = params.beta;
        if (beta == 0.0) {
            C[col] = alpha * sum0;
            if (m_len > 1u) { C[n_len + col] = alpha * sum1; }
            if (m_len > 2u) { C[2u * n_len + col] = alpha * sum2; }
            if (m_len > 3u) { C[3u * n_len + col] = alpha * sum3; }
        } else {
            C[col] = alpha * sum0 + beta * C[col];
            if (m_len > 1u) { C[n_len + col] = alpha * sum1 + beta * C[n_len + col]; }
            if (m_len > 2u) { C[2u * n_len + col] = alpha * sum2 + beta * C[2u * n_len + col]; }
            if (m_len > 3u) { C[3u * n_len + col] = alpha * sum3 + beta * C[3u * n_len + col]; }
        }
    }
}
