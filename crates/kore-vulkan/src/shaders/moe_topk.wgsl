// Top-k router selection and renormalisation, entirely on device.
//
// Mirrors the host path (`router_probs` + `top_k_experts`): a max-subtracted
// softmax over *all* experts, then the k largest, renormalised to sum to one.
// Keeping the full softmax rather than a top-k softmax is what makes the
// weights match the host lane, not just the ordering.
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

@compute @workgroup_size(1)
fn main() {
    let n = p.n;
    let k = min(p.k, 16u);
    var mx = NEG_INF;
    for (var i = 0u; i < n; i = i + 1u) {
        mx = max(mx, logits[i]);
    }
    var total = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
        total = total + exp(logits[i] - mx);
    }
    let inv_total = select(0.0, 1.0 / total, total > 0.0);

    // O(k*n) selection. n is a few hundred and k is single digits, so this is
    // cheaper than any sorted structure and has an obvious tie rule (lowest id).
    var picked: array<u32, 16>;
    for (var s = 0u; s < k; s = s + 1u) {
        var best = -1.0;
        var bi = 0u;
        for (var i = 0u; i < n; i = i + 1u) {
            var taken = false;
            for (var j = 0u; j < s; j = j + 1u) {
                if (picked[j] == i) {
                    taken = true;
                }
            }
            let pr = exp(logits[i] - mx) * inv_total;
            if (!taken && pr > best) {
                best = pr;
                bi = i;
            }
        }
        picked[s] = bi;
    }

    var kept = 0.0;
    for (var s = 0u; s < k; s = s + 1u) {
        kept = kept + exp(logits[picked[s]] - mx) * inv_total;
    }
    let inv_kept = select(0.0, 1.0 / kept, kept > 0.0);
    for (var s = 0u; s < k; s = s + 1u) {
        let pr = exp(logits[picked[s]] - mx) * inv_total;
        idx[s] = f32(picked[s]);
        w[s] = pr * inv_kept;
    }
}
