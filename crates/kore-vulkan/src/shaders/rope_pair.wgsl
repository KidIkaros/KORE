struct RopePairUniforms {
    pos: u32,
    head_dim: u32,
    half_dim: u32,
    rope_theta: f32,
    num_heads_q: u32,
    num_heads_k: u32,
    total_pairs_q: u32,   // (rows * nq * head_dim) / head_dim * half_dim
    total_pairs_k: u32,
}

@group(0) @binding(0) var<uniform> params: RopePairUniforms;
@group(0) @binding(1) var<storage, read> in_q: array<f32>;
@group(0) @binding(2) var<storage, read> in_k: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_q: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_k: array<f32>;

// Fused RoPE for Q and K in one dispatch: global pair index routes to the
// q buffer for pair_idx < total_pairs_q, else the k buffer.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_idx = global_id.x;
    if (pair_idx >= params.total_pairs_q + params.total_pairs_k) {
        return;
    }
    var p = pair_idx;
    var num_heads = params.num_heads_q;
    var is_q = true;
    if (pair_idx >= params.total_pairs_q) {
        p = pair_idx - params.total_pairs_q;
        num_heads = params.num_heads_k;
        is_q = false;
    }

    let h = p / params.half_dim;
    let i = p % params.half_dim;
    let seq_row = h / num_heads;

    let freq = 1.0 / pow(params.rope_theta, (2.0 * f32(i)) / f32(params.head_dim));
    let angle = f32(params.pos + seq_row) * freq;
    let cos_val = cos(angle);
    let sin_val = sin(angle);

    let idx0 = h * params.head_dim + 2u * i;
    let idx1 = h * params.head_dim + 2u * i + 1u;

    if (is_q) {
        let x0 = in_q[idx0];
        let x1 = in_q[idx1];
        out_q[idx0] = x0 * cos_val - x1 * sin_val;
        out_q[idx1] = x0 * sin_val + x1 * cos_val;
    } else {
        let x0 = in_k[idx0];
        let x1 = in_k[idx1];
        out_k[idx0] = x0 * cos_val - x1 * sin_val;
        out_k[idx1] = x0 * sin_val + x1 * cos_val;
    }
}
