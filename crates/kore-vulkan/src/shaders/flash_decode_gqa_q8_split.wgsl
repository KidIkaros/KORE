// Split-KV flash decode attention with INT8-packed K/V (shard pass).
//
// `flash_decode_gqa_q8.wgsl` launches only `num_heads x seq_q` workgroups. At
// decode (seq_q == 1) that is 8 workgroups of 64 threads for the whole GPU, each
// walking the entire KV window with no way to hide memory latency, so attention
// runs at ~1.4 GB/s (memory-latency-bound) and dominates the long-context
// collapse. Measured on the RTX 2070 Max-Q: 8.4 ms/token of attention at
// seq_k = 8192 against a 17.3 ms/token total.
//
// This kernel shards the KV window across `n_splits` workgroups per head
// (grid = (num_heads, n_splits, 1)) and writes one
// (shard max, shard sumexp, shard V-accumulator) triple per shard.
// `flash_decode_gqa_q8_combine.wgsl` merges the shards with a log-sum-exp.
//
// The arithmetic per shard is identical to flash_decode_gqa_q8.wgsl; the only
// difference is that the online running max is replaced by the shard's own max,
// which the combine pass rescales. Shards partition [0, seq_k) exactly.

struct Q8GqaSplitUniforms { seq_q: u32, seq_k: u32, head_dim: u32, scale: f32, num_heads: u32, num_kv_heads: u32, kv_dim: u32, q_dim: u32, n_splits: u32, _pad0: u32, _pad1: u32, _pad2: u32 }
@group(0) @binding(0) var<uniform> params: Q8GqaSplitUniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
// K/V are packed INT8: 4 bytes per u32 word, little-endian (byte i lives in word i>>2).
@group(0) @binding(2) var<storage, read> K_q8: array<u32>;
@group(0) @binding(3) var<storage, read> V_q8: array<u32>;
@group(0) @binding(4) var<storage, read> KScale: array<f32>;
@group(0) @binding(5) var<storage, read> VScale: array<f32>;
// Per shard: [max, sumexp] pairs, index = (head * n_splits + split) * 2.
@group(0) @binding(6) var<storage, read_write> PartML: array<f32>;
// Per shard V accumulator, laid out (head * n_splits + split) * head_dim + d.
@group(0) @binding(7) var<storage, read_write> PartAcc: array<f32>;
var<workgroup> shared_q: array<f32, 256>;
var<workgroup> shared_dot: array<f32, 64>;
var<workgroup> shared_scores: array<f32, 8>;

// Extract byte `i` from the packed K/V words and dequantize (offset-binary: byte-128).
fn k_dequant(i: u32, scale: f32) -> f32 {
    let w = K_q8[i >> 2u];
    return (f32((w >> (8u * (i & 3u))) & 0xFFu) - 128.0) * scale;
}
fn v_dequant(i: u32, scale: f32) -> f32 {
    let w = V_q8[i >> 2u];
    return (f32((w >> (8u * (i & 3u))) & 0xFFu) - 128.0) * scale;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let head_idx = wg_id.x; let split = wg_id.y;
    if (head_idx >= params.num_heads || split >= params.n_splits) { return; }
    let lx = local_id.x; let d_len = params.head_dim; let seq_k = params.seq_k;
    let scale = abs(params.scale);
    // seq_q == 1 decode only: no causal masking, all seq_k keys visible.
    let chunk = (seq_k + params.n_splits - 1u) / params.n_splits;
    let kv_begin = split * chunk;
    let kv_end = min(seq_k, kv_begin + chunk);
    let ks = params.num_heads / params.num_kv_heads; let hki = head_idx / ks;
    let q_base = head_idx * d_len;
    if (lx < d_len) { shared_q[lx] = Q[q_base + lx]; }
    if ((lx+64u) < d_len) { shared_q[lx+64u] = Q[q_base + lx + 64u]; }
    if ((lx+128u) < d_len) { shared_q[lx+128u] = Q[q_base + lx + 128u]; }
    if ((lx+192u) < d_len) { shared_q[lx+192u] = Q[q_base + lx + 192u]; }
    workgroupBarrier();
    var a0: f32=0.0; var a1: f32=0.0; var a2: f32=0.0; var a3: f32=0.0;
    var mi: f32 = -1e30; var li: f32 = 0.0;
    let g = lx >> 3u; let lane = lx & 7u;
    for (var tb: u32 = kv_begin; tb < kv_end; tb = tb + 8u) {
        let tc = tb + g; var part: f32 = 0.0;
        if (tc < kv_end) { let kb = tc * params.kv_dim + hki * d_len; for (var i: u32 = lane; i < d_len; i += 8u) { part += shared_q[i] * k_dequant(kb + i, KScale[tc]); } }
        shared_dot[lx] = part; workgroupBarrier();
        if (lane < 4u) { shared_dot[lx] += shared_dot[lx+4u]; } workgroupBarrier();
        if (lane < 2u) { shared_dot[lx] += shared_dot[lx+2u]; } workgroupBarrier();
        if (lane == 0u) { shared_scores[g] = (shared_dot[lx] + shared_dot[lx+1u]) * scale; }
        workgroupBarrier();
        let tl = min(8u, kv_end - tb);
        for (var j: u32 = 0u; j < tl; j++) {
            let s = shared_scores[j]; let mn = max(mi, s); let al = exp(mi - mn); let p = exp(s - mn);
            li = li * al + p; a0 *= al; a1 *= al; a2 *= al; a3 *= al;
            let vb = (tb + j) * params.kv_dim + hki * d_len;
            if (lx < d_len) { a0 += p * v_dequant(vb + lx, VScale[tb + j]); }
            if ((lx+64u) < d_len) { a1 += p * v_dequant(vb + lx + 64u, VScale[tb + j]); }
            if ((lx+128u) < d_len) { a2 += p * v_dequant(vb + lx + 128u, VScale[tb + j]); }
            if ((lx+192u) < d_len) { a3 += p * v_dequant(vb + lx + 192u, VScale[tb + j]); }
            mi = mn;
        }
        workgroupBarrier();
    }
    // A shard with no keys leaves (mi, li) at their initial values; the combine
    // pass weights it by exp(-1e30 - m) == 0, so it contributes nothing.
    let shard = head_idx * params.n_splits + split;
    if (lx == 0u) { PartML[shard * 2u] = mi; PartML[shard * 2u + 1u] = li; }
    let ab = shard * d_len;
    if (lx < d_len) { PartAcc[ab + lx] = a0; }
    if ((lx+64u) < d_len) { PartAcc[ab + lx + 64u] = a1; }
    if ((lx+128u) < d_len) { PartAcc[ab + lx + 128u] = a2; }
    if ((lx+192u) < d_len) { PartAcc[ab + lx + 192u] = a3; }
}
