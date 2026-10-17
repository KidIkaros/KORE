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
            let xv = X[base + 4u * w + k] * inv;
            let q = clamp(round(xv), -127.0, 127.0);
            packed = packed | (u32(i32(q) + 128) << (8u * k));
        }
        Q[row * nwords + w] = packed;
    }
}
