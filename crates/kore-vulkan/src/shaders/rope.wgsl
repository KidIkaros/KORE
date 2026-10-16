struct RopeUniforms {
    pos: u32,
    num_heads: u32,
    head_dim: u32,
    half_dim: u32,
    rope_theta: f32,
    total_pairs: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: RopeUniforms;
@group(0) @binding(1) var<storage, read> In: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;

// Rotary Position Embedding (RoPE) on device.
// Interleaved rotation: (2i, 2i+1) rotated by angle = pos * (1 / theta^(2i/d))
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_idx = global_id.x;
    if (pair_idx >= params.total_pairs) {
        return;
    }

    let h = pair_idx / params.half_dim;
    let i = pair_idx % params.half_dim;
    let seq_row = h / params.num_heads;

    let freq = 1.0 / pow(params.rope_theta, (2.0 * f32(i)) / f32(params.head_dim));
    let angle = f32(params.pos + seq_row) * freq;
    let cos_val = cos(angle);
    let sin_val = sin(angle);

    let idx0 = h * params.head_dim + 2u * i;
    let idx1 = h * params.head_dim + 2u * i + 1u;

    let x0 = In[idx0];
    let x1 = In[idx1];

    Out[idx0] = x0 * cos_val - x1 * sin_val;
    Out[idx1] = x0 * sin_val + x1 * cos_val;
}
