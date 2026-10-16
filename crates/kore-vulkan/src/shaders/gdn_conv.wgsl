struct GdnConvUniforms {
    conv_dim: u32,     // channels (qkv_dim)
    conv_k: u32,       // kernel size (4)
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: GdnConvUniforms;
@group(0) @binding(1) var<storage, read> x: array<f32>;          // [conv_dim] new qkv row
@group(0) @binding(2) var<storage, read> w: array<f32>;          // [conv_dim * conv_k]
@group(0) @binding(3) var<storage, read_write> hist: array<f32>; // [conv_dim * (conv_k-1)] rolling
@group(0) @binding(4) var<storage, read_write> out: array<f32>;  // [conv_dim] silu(conv)

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= params.conv_dim) {
        return;
    }
    let ck = params.conv_k;
    let hw = ck - 1u;
    var acc = 0.0;
    for (var k = 0u; k < hw; k = k + 1u) {
        acc += w[c * ck + k] * hist[c * hw + k];
    }
    acc += w[c * ck + hw] * x[c];
    // shift history left, append newest
    for (var k = 0u; k + 1u < hw; k = k + 1u) {
        hist[c * hw + k] = hist[c * hw + k + 1u];
    }
    hist[c * hw + hw - 1u] = x[c];
    out[c] = acc / (1.0 + exp(-acc));
}
