struct FlashDecodeUniforms {
    seq_q: u32,
    seq_k: u32,
    d: u32,
    scale: f32,
}

@group(0) @binding(0) var<uniform> params: FlashDecodeUniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> Out: array<f32>;

var<workgroup> shared_dot: array<f32, 64>;

// Fused Flash-Decoding Attention for decode & small-batch speculative verification (seq_q <= 4).
// Computes softmax(Q @ K^T * scale) @ V directly over row-major K[seq_k, d] and V[seq_k, d]
// without transposing K or materializing the [seq_q, seq_k] score matrix.
@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let q_row = wg_id.x;
    if (q_row >= params.seq_q) {
        return;
    }

    let lx = local_id.x;
    let d_len = params.d;
    let seq_k = params.seq_k;
    let is_causal = params.scale < 0.0;
    let scale = abs(params.scale);
    let max_t = select(seq_k, (seq_k - params.seq_q) + q_row + 1u, is_causal && seq_k >= params.seq_q);
    let q_base = q_row * d_len;

    // Each thread handles up to 4 feature dimensions: lx, lx + 64, lx + 128, lx + 192 (for d <= 256)
    let q0 = select(0.0, Q[q_base + lx], lx < d_len);
    let q1 = select(0.0, Q[q_base + lx + 64u], (lx + 64u) < d_len);
    let q2 = select(0.0, Q[q_base + lx + 128u], (lx + 128u) < d_len);
    let q3 = select(0.0, Q[q_base + lx + 192u], (lx + 192u) < d_len);

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;

    var m_i: f32 = -1e30;
    var l_i: f32 = 0.0;

    for (var t: u32 = 0u; t < max_t; t = t + 1u) {
        let kv_base = t * d_len;
        var partial: f32 = 0.0;
        if (lx < d_len) {
            partial = partial + q0 * K[kv_base + lx];
        }
        if ((lx + 64u) < d_len) {
            partial = partial + q1 * K[kv_base + lx + 64u];
        }
        if ((lx + 128u) < d_len) {
            partial = partial + q2 * K[kv_base + lx + 128u];
        }
        if ((lx + 192u) < d_len) {
            partial = partial + q3 * K[kv_base + lx + 192u];
        }
        // Handle any dimensions beyond 256
        for (var idx: u32 = lx + 256u; idx < d_len; idx = idx + 64u) {
            partial = partial + Q[q_base + idx] * K[kv_base + idx];
        }

        shared_dot[lx] = partial;
        workgroupBarrier();

        // Parallel tree reduction across 64 threads
        for (var stride: u32 = 32u; stride > 0u; stride = stride >> 1u) {
            if (lx < stride) {
                shared_dot[lx] = shared_dot[lx] + shared_dot[lx + stride];
            }
            workgroupBarrier();
        }

        let score = shared_dot[0] * scale;
        let m_new = max(m_i, score);
        let alpha = exp(m_i - m_new);
        let p = exp(score - m_new);

        l_i = l_i * alpha + p;
        acc0 = acc0 * alpha;
        acc1 = acc1 * alpha;
        acc2 = acc2 * alpha;
        acc3 = acc3 * alpha;

        if (lx < d_len) {
            acc0 = acc0 + p * V[kv_base + lx];
        }
        if ((lx + 64u) < d_len) {
            acc1 = acc1 + p * V[kv_base + lx + 64u];
        }
        if ((lx + 128u) < d_len) {
            acc2 = acc2 + p * V[kv_base + lx + 128u];
        }
        if ((lx + 192u) < d_len) {
            acc3 = acc3 + p * V[kv_base + lx + 192u];
        }
        m_i = m_new;
        workgroupBarrier();
    }

    let inv_l = 1.0 / max(l_i, 1e-20);
    if (lx < d_len) {
        Out[q_base + lx] = acc0 * inv_l;
    }
    if ((lx + 64u) < d_len) {
        Out[q_base + lx + 64u] = acc1 * inv_l;
    }
    if ((lx + 128u) < d_len) {
        Out[q_base + lx + 128u] = acc2 * inv_l;
    }
    if ((lx + 192u) < d_len) {
        Out[q_base + lx + 192u] = acc3 * inv_l;
    }
}
