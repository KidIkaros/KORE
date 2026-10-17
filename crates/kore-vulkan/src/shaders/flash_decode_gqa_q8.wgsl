struct Q8GqaUniforms { seq_q: u32, seq_k: u32, head_dim: u32, scale: f32, num_heads: u32, num_kv_heads: u32, kv_dim: u32, q_dim: u32 }
@group(0) @binding(0) var<uniform> params: Q8GqaUniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
// K/V are packed INT8: 4 bytes per u32 word, little-endian (byte i lives in word i>>2).
@group(0) @binding(2) var<storage, read> K_q8: array<u32>;
@group(0) @binding(3) var<storage, read> V_q8: array<u32>;
@group(0) @binding(4) var<storage, read> KScale: array<f32>;
@group(0) @binding(5) var<storage, read> VScale: array<f32>;
@group(0) @binding(6) var<storage, read_write> Out: array<f32>;
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

// Fused GQA decode attention with INT8-packed K/V (4x less KV read traffic than F32).
// Mirrors flash_decode_gqa.wgsl: 8-token tiled scoring, online softmax,
// barrier-free V accumulation. Supports seq_q == 1 decode and causal multi-query batches.
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let head_idx = wg_id.x; let q_row = wg_id.y;
    if (head_idx >= params.num_heads || q_row >= params.seq_q) { return; }
    let lx = local_id.x; let d_len = params.head_dim; let seq_k = params.seq_k;
    let is_causal = params.scale < 0.0 || params.seq_q > 1u; let scale = abs(params.scale);
    let max_t = select(seq_k, (seq_k - params.seq_q) + q_row + 1u, is_causal && seq_k >= params.seq_q);
    let ks = params.num_heads / params.num_kv_heads; let hki = head_idx / ks;
    let q_base = q_row * params.q_dim + head_idx * d_len; let out_base = q_row * params.q_dim + head_idx * d_len;
    if (lx < d_len) { shared_q[lx] = Q[q_base + lx]; }
    if ((lx+64u) < d_len) { shared_q[lx+64u] = Q[q_base + lx + 64u]; }
    if ((lx+128u) < d_len) { shared_q[lx+128u] = Q[q_base + lx + 128u]; }
    if ((lx+192u) < d_len) { shared_q[lx+192u] = Q[q_base + lx + 192u]; }
    workgroupBarrier();
    var a0: f32=0.0; var a1: f32=0.0; var a2: f32=0.0; var a3: f32=0.0;
    var mi: f32 = -1e30; var li: f32 = 0.0;
    let g = lx >> 3u; let lane = lx & 7u;
    for (var tb: u32 = 0u; tb < max_t; tb = tb + 8u) {
        let tc = tb + g; var part: f32 = 0.0;
        if (tc < max_t) { let kb = tc * params.kv_dim + hki * d_len; for (var i: u32 = lane; i < d_len; i += 8u) { part += shared_q[i] * k_dequant(kb + i, KScale[tc]); } }
        shared_dot[lx] = part; workgroupBarrier();
        if (lane < 4u) { shared_dot[lx] += shared_dot[lx+4u]; } workgroupBarrier();
        if (lane < 2u) { shared_dot[lx] += shared_dot[lx+2u]; } workgroupBarrier();
        if (lane == 0u) { shared_scores[g] = (shared_dot[lx] + shared_dot[lx+1u]) * scale; }
        workgroupBarrier();
        let tl = min(8u, max_t - tb);
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
    let il = select(0.0, 1.0 / li, li > 0.0);
    if (lx < d_len) { Out[out_base + lx] = a0 * il; }
    if ((lx+64u) < d_len) { Out[out_base + lx + 64u] = a1 * il; }
    if ((lx+128u) < d_len) { Out[out_base + lx + 128u] = a2 * il; }
    if ((lx+192u) < d_len) { Out[out_base + lx + 192u] = a3 * il; }
}
