struct GdnGatesUniforms {
    hidden: u32,   // 2048
    n_v: u32,      // 16
    _pad0: u32,
    _pad1: u32,
}

// beta[j] = sigmoid(dot(w_beta[j], h))
// gate[j] = softplus(dot(w_alpha[j], h) + dt_bias[j]) * a[j]
// w_alpha/w_beta: [n_v, hidden] row-major (row j contiguous).
@group(0) @binding(0) var<uniform> params: GdnGatesUniforms;
@group(0) @binding(1) var<storage, read> h: array<f32>;
@group(0) @binding(2) var<storage, read> w_alpha: array<f32>;
@group(0) @binding(3) var<storage, read> w_beta: array<f32>;
@group(0) @binding(4) var<storage, read> dt_bias: array<f32>;
@group(0) @binding(5) var<storage, read> ssm_a: array<f32>;
@group(0) @binding(6) var<storage, read_write> beta_out: array<f32>;
@group(0) @binding(7) var<storage, read_write> gate_out: array<f32>;

var<workgroup> red_a: array<f32, 128>;
var<workgroup> red_b: array<f32, 128>;

// one workgroup per output j
@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let j = wg_id.x;
    if (j >= params.n_v) {
        return;
    }
    let tid = lid.x;
    let m = params.hidden;
    var sa = 0.0;
    var sb = 0.0;
    for (var i = tid; i < m; i = i + 128u) {
        let hv = h[i];
        sa += w_alpha[j * m + i] * hv;
        sb += w_beta[j * m + i] * hv;
    }
    red_a[tid] = sa;
    red_b[tid] = sb;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            red_a[tid] += red_a[tid + s];
            red_b[tid] += red_b[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        beta_out[j] = 1.0 / (1.0 + exp(-red_b[0]));
        let a_b = red_a[0] + dt_bias[j];
        // softplus with large-arg passthrough
        let sp = select(log(1.0 + exp(a_b)), a_b, a_b > 20.0);
        gate_out[j] = sp * ssm_a[j];
    }
}
