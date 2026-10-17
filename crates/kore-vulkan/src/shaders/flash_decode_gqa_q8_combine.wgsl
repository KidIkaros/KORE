// Merge pass for split-KV flash decode (see flash_decode_gqa_q8_split.wgsl).
//
// Each shard wrote (max, sumexp, V-accumulator) computed against its *own* max.
// This rescales every shard by exp(shard_max - global_max) so the running
// normalization is consistent, then normalizes. One workgroup per head; thread
// `d` owns output dimension `d` (head_dim <= 256).
//
// seq_q == 1 only, so the output row base is head_idx * head_dim.

struct Q8CombineUniforms { head_dim: u32, n_splits: u32, num_heads: u32, _pad0: u32 }
@group(0) @binding(0) var<uniform> params: Q8CombineUniforms;
@group(0) @binding(1) var<storage, read> PartML: array<f32>;
@group(0) @binding(2) var<storage, read> PartAcc: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let head = wg_id.x;
    if (head >= params.num_heads) { return; }
    let d = local_id.x;
    let ns = params.n_splits; let d_len = params.head_dim;

    var m: f32 = -1e30;
    for (var s: u32 = 0u; s < ns; s = s + 1u) { m = max(m, PartML[(head * ns + s) * 2u]); }

    var l: f32 = 0.0;
    for (var s: u32 = 0u; s < ns; s = s + 1u) {
        let w = exp(PartML[(head * ns + s) * 2u] - m);
        l += PartML[(head * ns + s) * 2u + 1u] * w;
    }

    if (d < d_len) {
        var acc: f32 = 0.0;
        for (var s: u32 = 0u; s < ns; s = s + 1u) {
            let w = exp(PartML[(head * ns + s) * 2u] - m);
            acc += PartAcc[(head * ns + s) * d_len + d] * w;
        }
        Out[head * d_len + d] = select(0.0, acc / l, l > 0.0);
    }
}
