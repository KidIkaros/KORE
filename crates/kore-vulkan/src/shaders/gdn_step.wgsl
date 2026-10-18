struct GdnStepUniforms {
    n_v: u32,          // value heads (32 on the 9B, 16 on Bonsai)
    n_k: u32,          // key heads (16 on the 9B, == n_v on Bonsai)
    d_k: u32,          // key dim per head (128)
    d_v: u32,          // value dim per head (128)
    eps: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// qkv layout: [q: n_k*d_k | k: n_k*d_k | v: n_v*d_v]
// Value head `hv` reads key head `hv % n_k` -- the key heads are *tiled* up to
// the value head count, matching llama.cpp's `ggml_gated_delta_net`
// (`iq1 = iv1 % neq1`).
// state layout per head: s[j*d_v + i] = S[i][j]  (row j holds S[:, j])
@group(0) @binding(0) var<uniform> params: GdnStepUniforms;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;        // conv+silu output
@group(0) @binding(2) var<storage, read> z: array<f32>;          // [n_v*d_v] gate
@group(0) @binding(3) var<storage, read_write> state: array<f32>;// [n_v * d_v * d_v]
@group(0) @binding(4) var<storage, read> beta: array<f32>;       // [n_v] sigmoided
@group(0) @binding(5) var<storage, read> gate: array<f32>;       // [n_v] softplus*a
@group(0) @binding(6) var<storage, read> norm_w: array<f32>;     // [d_v] rmsnorm weight
@group(0) @binding(7) var<storage, read_write> out: array<f32>;  // [n_v*d_v]

var<workgroup> kq: array<f32, 512>;      // k[128] q[128] delta[128] v[128]
var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let hv = wg_id.x;
    if (hv >= params.n_v) {
        return;
    }
    let dk = params.d_k;
    let dv = params.d_v;
    let tid = lid.x;
    let hk = hv % params.n_k;
    let q_off = hk * dk;
    let k_off = params.n_k * dk + hk * dk;
    let v_off = 2u * params.n_k * dk + hv * dv;
    let s_base = hv * dv * dv;

    // load k, q, v into shared
    kq[tid] = qkv[k_off + tid];          // k
    kq[128u + tid] = qkv[q_off + tid];   // q
    kq[256u + tid] = qkv[v_off + tid];   // v
    workgroupBarrier();

    // l2-normalize k and q in place: x / max(sqrt(ss), eps)
    var ss = kq[tid] * kq[tid];
    red[tid] = ss;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (tid < s) { red[tid] += red[tid + s]; }
        workgroupBarrier();
    }
    let kinv = 1.0 / max(sqrt(red[0]), params.eps);
    workgroupBarrier();
    kq[tid] = kq[tid] * kinv;

    ss = kq[128u + tid] * kq[128u + tid];
    workgroupBarrier();
    red[tid] = ss;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (tid < s) { red[tid] += red[tid + s]; }
        workgroupBarrier();
    }
    let qinv = 1.0 / max(sqrt(red[0]), params.eps);
    workgroupBarrier();
    kq[128u + tid] = kq[128u + tid] * qinv;

    let decay = exp(gate[hv]);
    let b = beta[hv];

    // delta[j] = (v[j] - decay * (row_j . k)) * beta   (row_j = S[:, j] decayed)
    let j = tid;
    var dot = 0.0;
    for (var i = 0u; i < dv; i = i + 1u) {
        dot += state[s_base + j * dv + i] * kq[i];
    }
    kq[384u + j] = (kq[256u + j] - decay * dot) * b;
    workgroupBarrier();

    // S[i][j] = decay * S[i][j] + k[i] * delta[j]; row j update
    let dlt = kq[384u + j];
    for (var i = 0u; i < dv; i = i + 1u) {
        let idx = s_base + j * dv + i;
        state[idx] = decay * state[idx] + kq[i] * dlt;
    }

    // out[j] = (S_new . q)[j] / sqrt(d_v)
    var acc = 0.0;
    for (var i = 0u; i < dv; i = i + 1u) {
        acc += state[s_base + j * dv + i] * kq[128u + i];
    }
    let o = acc * inverseSqrt(f32(dv));
    workgroupBarrier();
    red[tid] = o * o;
    kq[256u + tid] = o;                  // stash raw head output
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (tid < s) { red[tid] += red[tid + s]; }
        workgroupBarrier();
    }
    // gated rmsnorm: rms_norm(o)*w * silu(z)
    let rms = 1.0 / sqrt(red[0] / f32(dv) + params.eps);
    let oi = kq[256u + tid] * rms * norm_w[tid];
    let zv = z[hv * dv + tid];
    let sig = 1.0 / (1.0 + exp(-zv));
    out[hv * dv + tid] = oi * zv * sig;
}
