struct FlashDecodeGqaSinkUniforms {
    seq_q: u32,
    seq_k: u32,
    head_dim: u32,
    scale: f32,
    num_heads: u32,
    num_kv_heads: u32,
    kv_dim: u32,
    q_dim: u32,
    // Index of the first key in the window. K/V are the whole cache, so this is
    // what makes a sliding window free: the kernel reads
    // `(kv_start + t) * kv_dim`, and the caller sets kv_start to
    // `seen - window`.
    kv_start: u32,
    // 0 -> no sink term (plain softmax); 1 -> `Sinks[head]` joins the
    // denominator with the score it is written as.
    has_sink: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: FlashDecodeGqaSinkUniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> Out: array<f32>;
@group(0) @binding(5) var<storage, read> Sinks: array<f32>;

var<workgroup> shared_q: array<f32, 256>;
var<workgroup> shared_dot: array<f32, 64>;
var<workgroup> shared_scores: array<f32, 8>;

// `flash_decode_gqa` plus a learned attention sink and a window offset.
//
// The sink is one extra term in the denominator with no V of its own:
//
//     p_i = exp(s_i - m) / (exp(sink - m) + sum_j exp(s_j - m))
//
// which online softmax reproduces exactly by *starting* at `m = sink, l = 1`
// instead of `m = -inf, l = 0`, as the reopen below does. Seeding it that way
// also keeps a very negative sink from being an overflow hazard.
@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let head_idx = wg_id.x;
    let q_row = wg_id.y;
    if (head_idx >= params.num_heads || q_row >= params.seq_q) {
        return;
    }

    let lx = local_id.x;
    let d_len = params.head_dim;
    let seq_k = params.seq_k;
    let is_causal = params.scale < 0.0 || params.seq_q > 1u;
    let scale = abs(params.scale);
    let max_t = select(seq_k, (seq_k - params.seq_q) + q_row + 1u, is_causal && seq_k >= params.seq_q);

    let kv_group_size = params.num_heads / params.num_kv_heads;
    let kv_head_idx = head_idx / kv_group_size;

    let q_base = q_row * params.q_dim + head_idx * d_len;
    let out_base = q_row * params.q_dim + head_idx * d_len;

    if (lx < d_len) {
        shared_q[lx] = Q[q_base + lx];
    }
    if ((lx + 64u) < d_len) {
        shared_q[lx + 64u] = Q[q_base + lx + 64u];
    }
    if ((lx + 128u) < d_len) {
        shared_q[lx + 128u] = Q[q_base + lx + 128u];
    }
    if ((lx + 192u) < d_len) {
        shared_q[lx + 192u] = Q[q_base + lx + 192u];
    }
    workgroupBarrier();

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;

    // Online-softmax state, seeded with the sink when there is one.
    var m_i: f32 = -1e30;
    var l_i: f32 = 0.0;
    if (params.has_sink != 0u) {
        m_i = Sinks[head_idx];
        l_i = 1.0;
    }

    let g = lx >> 3u;
    let lane = lx & 7u;
    let kv_first = params.kv_start * params.kv_dim + kv_head_idx * d_len;

    for (var t_base: u32 = 0u; t_base < max_t; t_base = t_base + 8u) {
        let t_cur = t_base + g;
        var partial: f32 = 0.0;
        if (t_cur < max_t) {
            let k_base = kv_first + t_cur * params.kv_dim;
            for (var idx: u32 = lane; idx < d_len; idx = idx + 8u) {
                partial = partial + shared_q[idx] * K[k_base + idx];
            }
        }

        shared_dot[lx] = partial;
        workgroupBarrier();

        if (lane < 4u) {
            shared_dot[lx] = shared_dot[lx] + shared_dot[lx + 4u];
        }
        workgroupBarrier();

        if (lane < 2u) {
            shared_dot[lx] = shared_dot[lx] + shared_dot[lx + 2u];
        }
        workgroupBarrier();

        if (lane == 0u) {
            shared_scores[g] = (shared_dot[lx] + shared_dot[lx + 1u]) * scale;
        }
        workgroupBarrier();

        let tile_len = min(8u, max_t - t_base);
        for (var j: u32 = 0u; j < tile_len; j = j + 1u) {
            let score = shared_scores[j];
            let m_new = max(m_i, score);
            let alpha = exp(m_i - m_new);
            let p = exp(score - m_new);

            l_i = l_i * alpha + p;
            acc0 = acc0 * alpha;
            acc1 = acc1 * alpha;
            acc2 = acc2 * alpha;
            acc3 = acc3 * alpha;

            let v_base = kv_first + (t_base + j) * params.kv_dim;
            if (lx < d_len) {
                acc0 = acc0 + p * V[v_base + lx];
            }
            if ((lx + 64u) < d_len) {
                acc1 = acc1 + p * V[v_base + lx + 64u];
            }
            if ((lx + 128u) < d_len) {
                acc2 = acc2 + p * V[v_base + lx + 128u];
            }
            if ((lx + 192u) < d_len) {
                acc3 = acc3 + p * V[v_base + lx + 192u];
            }
            m_i = m_new;
        }
        workgroupBarrier();
    }

    let inv_l = select(0.0, 1.0 / l_i, l_i > 0.0);
    if (lx < d_len) {
        Out[out_base + lx] = acc0 * inv_l;
    }
    if ((lx + 64u) < d_len) {
        Out[out_base + lx + 64u] = acc1 * inv_l;
    }
    if ((lx + 128u) < d_len) {
        Out[out_base + lx + 128u] = acc2 * inv_l;
    }
    if ((lx + 192u) < d_len) {
        Out[out_base + lx + 192u] = acc3 * inv_l;
    }
}
