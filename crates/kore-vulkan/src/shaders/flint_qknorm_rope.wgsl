struct FlintQkNormRopeUniforms {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    pos: u32,
    // 1 -> rotate the first head_dim elements; 0 -> NoPE (14 of Flint's 49
    // layers have no rotary embedding at all).
    rope: u32,
    _pad: u32,
    theta: f32,
    eps: f32,
}

@group(0) @binding(0) var<uniform> params: FlintQkNormRopeUniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read> QNorm: array<f32>;
@group(0) @binding(4) var<storage, read> KNorm: array<f32>;
@group(0) @binding(5) var<storage, read_write> QOut: array<f32>;
@group(0) @binding(6) var<storage, read_write> KOut: array<f32>;

var<workgroup> buf: array<f32, 256>;
var<workgroup> red: array<f32, 128>;

// Per-head RMS normalisation followed by split-half RoPE, for a plain
// `[n_heads * head_dim]` layout (qwen's `qknorm_rope` carries a q|gate pair and
// is fixed at head_dim 256, so this is a separate kernel rather than a flag).
//
// Element `i` of a head pairs with `i + head_dim/2` and rotates by
// `pos * theta^(-2i/head_dim)` -- the llama convention, which is what the
// reference implementation for this model uses. Every layer normalises; only
// the `rope` flag says whether it also rotates.
//
// One workgroup per head: workgroup x in [0, n_heads) is a query head and
// [n_heads, n_heads + n_kv) a key head. Each thread holds elements `tid` and
// `tid + 128` so a head of up to 256 elements needs no loop.
@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wg_id.x;
    let tid = lid.x;
    let nh = params.n_heads;
    let hd = params.head_dim;
    if (h >= nh + params.n_kv || tid >= hd) {
        return;
    }
    let is_q = h < nh;
    let base = select((h - nh) * hd, h * hd, is_q);

    var a = 0.0;
    var b = 0.0;
    if (tid < hd) {
        a = select(K[base + tid], Q[base + tid], is_q);
    }
    if (tid + 128u < hd) {
        b = select(K[base + tid + 128u], Q[base + tid + 128u], is_q);
    }

    red[tid] = a * a + b * b;
    workgroupBarrier();
    for (var s: u32 = 64u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            red[tid] = red[tid] + red[tid + s];
        }
        workgroupBarrier();
    }
    let rms = 1.0 / sqrt(red[0] / f32(hd) + params.eps);

    if (tid < hd) {
        let w = select(KNorm[tid], QNorm[tid], is_q);
        buf[tid] = a * rms * w;
    }
    if (tid + 128u < hd) {
        let w = select(KNorm[tid + 128u], QNorm[tid + 128u], is_q);
        buf[tid + 128u] = b * rms * w;
    }
    workgroupBarrier();

    if (params.rope == 0u) {
        if (tid < hd) {
            if (is_q) { QOut[base + tid] = buf[tid]; } else { KOut[base + tid] = buf[tid]; }
        }
        if (tid + 128u < hd) {
            if (is_q) { QOut[base + tid + 128u] = buf[tid + 128u]; }
            else { KOut[base + tid + 128u] = buf[tid + 128u]; }
        }
        return;
    }

    let half = hd / 2u;
    if (tid < half) {
        let freq = pow(params.theta, -2.0 * f32(tid) / f32(hd));
        let ang = f32(params.pos) * freq;
        let c = cos(ang);
        let s = sin(ang);
        let x0 = buf[tid];
        let x1 = buf[tid + half];
        let o0 = x0 * c - x1 * s;
        let o1 = x1 * c + x0 * s;
        if (is_q) {
            QOut[base + tid] = o0;
            QOut[base + tid + half] = o1;
        } else {
            KOut[base + tid] = o0;
            KOut[base + tid + half] = o1;
        }
    }
}
