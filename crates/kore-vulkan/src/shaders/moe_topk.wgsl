// Top-k router selection and renormalisation, entirely on device.
//
// Mirrors the host path (`router_probs` + `top_k_experts`): a max-subtracted
// softmax over *all* experts, then the k largest, renormalised to sum to one.
// Keeping the full softmax rather than a top-k softmax is what makes the
// weights match the host lane, not just the ordering.
//
// This used to be a single-lane workgroup. The selection is O(k*n) candidates
// and the old loop also re-ran the exclusion scan inside it, so a 256-expert
// layer spent ~0.5 ms here and forty layers spent ~22 ms of a 49 ms step. One
// workgroup of 256 lanes now owns one expert each: the max and the k argmax
// passes are tree reductions, and only the softmax normaliser stays sequential
// so its summation order -- and therefore the routing weights -- stay
// bit-identical to the host lane.
struct TopkUniforms {
    n: u32,
    k: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: TopkUniforms;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> idx: array<f32>;
@group(0) @binding(3) var<storage, read_write> w: array<f32>;

const NEG_INF: f32 = -3.402823466e38;
const WG: u32 = 256u;
const MAX_EXPERTS: u32 = 1024u;

var<workgroup> red_v: array<f32, 256>;
var<workgroup> red_i: array<u32, 256>;
var<workgroup> taken: array<u32, 1024>;
var<workgroup> picked: array<u32, 16>;
var<workgroup> kept_v: f32;

fn tree_max(t: u32) {
    var st = WG / 2u;
    loop {
        if (st == 0u) {
            break;
        }
        if (t < st) {
            red_v[t] = max(red_v[t], red_v[t + st]);
        }
        workgroupBarrier();
        st = st / 2u;
    }
}

/// Max reduce of `(red_v, red_i)`, ties going to the lower expert id -- the
/// same rule the host scan gets by visiting ids in ascending order.
fn tree_argmax(t: u32) {
    var st = WG / 2u;
    loop {
        if (st == 0u) {
            break;
        }
        if (t < st) {
            let av = red_v[t];
            let bv = red_v[t + st];
            let ai = red_i[t];
            let bi = red_i[t + st];
            if (bv > av || (bv == av && bi < ai)) {
                red_v[t] = bv;
                red_i[t] = bi;
            }
        }
        workgroupBarrier();
        st = st / 2u;
    }
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    let n = min(p.n, MAX_EXPERTS);
    let k = min(p.k, 16u);

    for (var i = t; i < n; i = i + WG) {
        taken[i] = 0u;
    }
    workgroupBarrier();

    // Softmax max, in parallel.
    var mx_t = NEG_INF;
    for (var i = t; i < n; i = i + WG) {
        mx_t = max(mx_t, logits[i]);
    }
    red_v[t] = mx_t;
    workgroupBarrier();
    tree_max(t);
    let mx = red_v[0];
    workgroupBarrier();

    // Normaliser: one lane, ascending id, exactly as the host accumulates it.
    if (t == 0u) {
        var total = 0.0;
        for (var i = 0u; i < n; i = i + 1u) {
            total = total + exp(logits[i] - mx);
        }
        red_v[0] = select(0.0, 1.0 / total, total > 0.0);
    }
    workgroupBarrier();
    let inv_total = red_v[0];

    // k rounds; each is a parallel argmax over the experts not yet taken.
    for (var s = 0u; s < k; s = s + 1u) {
        var best = NEG_INF;
        var bi = 0xFFFFFFFFu;
        for (var i = t; i < n; i = i + WG) {
            if (taken[i] == 0u) {
                let pr = exp(logits[i] - mx) * inv_total;
                if (pr > best || (pr == best && i < bi)) {
                    best = pr;
                    bi = i;
                }
            }
        }
        red_v[t] = best;
        red_i[t] = bi;
        workgroupBarrier();
        tree_argmax(t);
        if (t == 0u) {
            picked[s] = red_i[0];
            taken[red_i[0]] = 1u;
        }
        workgroupBarrier();
    }

    if (t == 0u) {
        var kept = 0.0;
        for (var s = 0u; s < k; s = s + 1u) {
            kept = kept + exp(logits[picked[s]] - mx) * inv_total;
        }
        kept_v = select(0.0, 1.0 / kept, kept > 0.0);
    }
    workgroupBarrier();
    if (t < k) {
        let e = picked[t];
        let pr = exp(logits[e] - mx) * inv_total;
        idx[t] = f32(e);
        w[t] = pr * kept_v;
    }
}
