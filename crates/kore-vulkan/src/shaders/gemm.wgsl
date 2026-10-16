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

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if (row < params.m && col < params.n) {
        var sum: f32 = 0.0;
        let k_len = params.k;
        let n_len = params.n;
        let a_row_offset = row * k_len;

        var k: u32 = 0u;
        let k_end = (k_len / 4u) * 4u;
        for (; k < k_end; k = k + 4u) {
            sum = sum + A[a_row_offset + k] * B[k * n_len + col]
                      + A[a_row_offset + k + 1u] * B[(k + 1u) * n_len + col]
                      + A[a_row_offset + k + 2u] * B[(k + 2u) * n_len + col]
                      + A[a_row_offset + k + 3u] * B[(k + 3u) * n_len + col];
        }
        for (; k < k_len; k = k + 1u) {
            sum = sum + A[a_row_offset + k] * B[k * n_len + col];
        }

        let idx = row * n_len + col;
        if (params.beta == 0.0) {
            C[idx] = params.alpha * sum;
        } else {
            C[idx] = params.alpha * sum + params.beta * C[idx];
        }
    }
}
