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
// State layout per head: s[i*d_v + j] = S[i][j]. Lane `j` walks i, so every
// step of the state sweeps reads/writes `i*d_v + j` -- consecutive lanes
// touch consecutive words. The old transposed layout (`j*d_v + i`, one head row
// per lane) strided lanes by 512 B and fetched a full sector per 4 B used.
//
// The two state sweeps are split across two thread groups: group `g` owns rows
// `g*d_v/2 .. (g+1)*d_v/2` of every head, so each lane's dependent chain is half
// as long and a workgroup carries 8 warps instead of 4 (one workgroup per CU,
// 30 dispatches per step, so warps per CU is the resource that was missing).
// The two partial sums are added at the end of each sweep -- `dot` and the
// readout -- which reassociates a 128-term left fold into two 64-term folds.
// That is the one numerics change in this kernel: everything else (the state
// update per element, both L2 norms, the RMS reduction) keeps its original
// order, so the difference is a few ulp, measured by
// `test_vulkan_gdn_step_split_matches_reference`.
@group(0) @binding(0) var<uniform> params: GdnStepUniforms;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;        // conv+silu output
@group(0) @binding(2) var<storage, read> z: array<f32>;          // [n_v*d_v] gate
@group(0) @binding(3) var<storage, read_write> state: array<f32>;// [n_v * d_v * d_v]
@group(0) @binding(4) var<storage, read> beta: array<f32>;       // [n_v] sigmoided
@group(0) @binding(5) var<storage, read> gate: array<f32>;       // [n_v] softplus*a
@group(0) @binding(6) var<storage, read> norm_w: array<f32>;     // [d_v] rmsnorm weight
@group(0) @binding(7) var<storage, read_write> out: array<f32>;  // [n_v*d_v]

// k[128] q[128] delta[128] v[128], shared by both groups.
var<workgroup> kq: array<f32, 512>;
// One partial per group per lane: `part[g*128 + j]`.
var<workgroup> part: array<f32, 256>;
// Per-group reduction scratch; both groups run the same 128-lane tree.
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let hv = wg_id.x;
    if (hv >= params.n_v) {
        return;
    }
    let g = lid.x >> 7u;
    let tid = lid.x & 127u;
    let rb = g * 128u;
    let dk = params.d_k;
    let dv = params.d_v;
    let half_dv = dv / 2u;
    let hk = hv % params.n_k;
    let q_off = hk * dk;
    let k_off = params.n_k * dk + hk * dk;
    let v_off = 2u * params.n_k * dk + hv * dv;
    let s_base = hv * dv * dv;

    // load k, q, v into shared (one group is enough; both read the same copy)
    if (g == 0u) {
        kq[tid] = qkv[k_off + tid];          // k
        kq[128u + tid] = qkv[q_off + tid];   // q
        kq[256u + tid] = qkv[v_off + tid];   // v
    }
    workgroupBarrier();

    // l2-normalize k and q in place: x / max(sqrt(ss), eps). Both groups run
    // the same tree over the same values, so `kinv`/`qinv` agree exactly.
    var ss = kq[tid] * kq[tid];
    red[rb + tid] = ss;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (tid < s) { red[rb + tid] += red[rb + tid + s]; }
        workgroupBarrier();
    }
    let kinv = 1.0 / max(sqrt(red[rb]), params.eps);
    workgroupBarrier();
    if (g == 0u) {
        kq[tid] = kq[tid] * kinv;
    }

    ss = kq[128u + tid] * kq[128u + tid];
    workgroupBarrier();
    red[rb + tid] = ss;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (tid < s) { red[rb + tid] += red[rb + tid + s]; }
        workgroupBarrier();
    }
    let qinv = 1.0 / max(sqrt(red[rb]), params.eps);
    workgroupBarrier();
    if (g == 0u) {
        kq[128u + tid] = kq[128u + tid] * qinv;
    }
    workgroupBarrier();

    let decay = exp(gate[hv]);
    let b = beta[hv];

    // delta[j] = (v[j] - decay * (S[:, j] . k)) * beta, S decayed.
    // Each group sums its own half of the `i` range, then lane j adds the two.
    let j = tid;
    var dot_p = 0.0;
    let i0 = g * half_dv;
    for (var ii = 0u; ii < half_dv; ii = ii + 1u) {
        let i = i0 + ii;
        dot_p += state[s_base + i * dv + j] * kq[i];
    }
    part[rb + j] = dot_p;
    workgroupBarrier();
    let dot = part[j] + part[128u + j];
    let dlt = (kq[256u + j] - decay * dot) * b;
    workgroupBarrier();

    // S[i][j] = decay * S[i][j] + k[i] * delta[j], with the readout
    // out[j] = (S_new . q)[j] accumulated in the same walk (bit-identical to a
    // separate third sweep -- lane j reads back only its own writes).
    var acc_p = 0.0;
    for (var ii = 0u; ii < half_dv; ii = ii + 1u) {
        let i = i0 + ii;
        let idx = s_base + i * dv + j;
        let s = decay * state[idx] + kq[i] * dlt;
        state[idx] = s;
        acc_p += s * kq[128u + i];
    }
    part[rb + j] = acc_p;
    workgroupBarrier();
    let acc = part[j] + part[128u + j];
    let o = acc * inverseSqrt(f32(dv));
    workgroupBarrier();

    // Gated rmsnorm over j. Both groups hold the same `o`, so either can run
    // the tree; both do, and only group 0 writes the result.
    red[rb + tid] = o * o;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (tid < s) { red[rb + tid] += red[rb + tid + s]; }
        workgroupBarrier();
    }
    let rms = 1.0 / sqrt(red[rb] / f32(dv) + params.eps);
    let oi = o * rms * norm_w[tid];
    let zv = z[hv * dv + tid];
    let sig = 1.0 / (1.0 + exp(-zv));
    if (g == 0u) {
        out[hv * dv + tid] = oi * zv * sig;
    }
}
