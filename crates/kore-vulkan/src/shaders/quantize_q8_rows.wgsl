struct QuantQ8Uniforms {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: QuantQ8Uniforms;
@group(0) @binding(1) var<storage, read> X: array<f32>;
// Packed INT8 output: 4 bytes per u32 word, little-endian (matches flash_decode_gqa_q8).
@group(0) @binding(2) var<storage, read_write> Q: array<u32>;
@group(0) @binding(3) var<storage, read_write> S: array<f32>;

var<workgroup> red: array<f32, 256>;

// WGSL `round` resolves `.5` ties however the implementation likes (GLSL says
// so explicitly, and RADV reads 63.5 as 63 where the CPU recipe reads 64), so
// the tie is pinned here: half away from zero, which is what `f32::round` and
// every GGUF quantizer do.
//
// The *contract* is still "codes equal a CPU reference except within 1 code of
// a `.5` tie": this shader forms the reciprocal `127/absmax` on the device,
// Vulkan only promises 2.5 ULP for that division, and one ULP is enough to move
// an exactly-representable 63.5 to 63.4999. The returned `S` is the scale both
// lanes dequantize with, so the cache stays self-consistent.
fn round_half_away(x: f32) -> f32 {
    if (x >= 0.0) {
        return floor(x + 0.5);
    }
    return ceil(x - 0.5);
}

// Per-row symmetric INT8 quantization (offset-binary, scale = absmax/127).
// One workgroup per row: reduces absmax across the row, then each thread packs
// 4 consecutive int8 codes into a u32 word. `cols` must be a multiple of 4.
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lx: vec3<u32>) {
    let row = wg.x;
    if (row >= params.rows) { return; }
    let base = row * params.cols;

    var m: f32 = 0.0;
    for (var c: u32 = lx.x; c < params.cols; c += 256u) {
        m = max(m, abs(X[base + c]));
    }
    red[lx.x] = m;
    workgroupBarrier();

    var stride: u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lx.x < stride) { red[lx.x] = max(red[lx.x], red[lx.x + stride]); }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    let absmax = red[0];
    let scale = select(1.0, absmax / 127.0, absmax > 0.0);
    let inv = 1.0 / scale;
    if (lx.x == 0u) { S[row] = scale; }

    let nwords = params.cols >> 2u;
    for (var w: u32 = lx.x; w < nwords; w += 256u) {
        var packed: u32 = 0u;
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            // One correctly-rounded multiply. The reciprocal is not
            // reproducible across lanes (Vulkan promises 2.5 ULP for division),
            // so a code can differ from a CPU reference by 1 at a `.5` tie --
            // see the contract note on `round_half_away`.
            let xv = X[base + 4u * w + k] * inv;
            let q = clamp(round_half_away(xv), -127.0, 127.0);
            packed = packed | (u32(i32(q) + 128) << (8u * k));
        }
        Q[row * nwords + w] = packed;
    }
}
