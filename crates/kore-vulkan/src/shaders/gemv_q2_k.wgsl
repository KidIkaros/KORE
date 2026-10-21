struct Q2KUniforms {
    m: u32,
    n: u32,
    k: u32,
    blocks_per_row: u32,
    // Base of this matrix inside `w`, in u32 words (a stacked expert bank).
    w_off: u32,
    // Output base in f32 elements, and whether to add onto it instead of
    // overwriting. The routed `down` projection accumulates its experts here,
    // which removes one dispatch per expert.
    y_off: u32,
    accum: u32,
    _pad0: u32,
    // Expert-strided mode: when nonzero, output row `r` reads expert
    // `r / rows_per_expert`, whose weights start `expert_words` words in and
    // whose activation is row `r / rows_per_expert` of `x`.
    id: u32,
    rows_per_expert: u32,
    expert_words: u32,
    // Indexed mode: output row `r` belongs to position `r / rows_per_expert`,
    // whose expert id is `idx[idx_off + position]`. `x_stride` is the
    // activation row stride in vec4 units (0 when every position shares the
    // same input row, as gate/up do).
    idx_off: u32,
    x_stride: u32,
    // Number of consecutive routing positions that share one activation row:
    // `n_used` for a verify batch, 1 for a single token.
    pos_div: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Q2KUniforms;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
// Expert ids from `moe_topk`, as f32 (tensors are f32 in this lane; ids are
// exact integers below 2^24). Only read in indexed mode.
@group(0) @binding(4) var<storage, read> idx: array<f32>;

var<workgroup> sdata: array<vec4<f32>, 128>;

// Batch-Invariant Small-Batch Q2_K GEMV (@workgroup_size(128) = 4 rows per workgroup).
//
// A Q2_K block is 84 bytes: scales[16], qs[64], then (d, dmin) as two f16. It holds
// 256 weights in sixteen 16-wide sub-blocks. `scales[i]` is one byte: the low nibble
// scales sub-block i and the high nibble is its minimum.
//
// `qs` is packed so that four sub-blocks share a 16-byte window, one per 2-bit
// field, and the window alternates 0/16 inside each 32-byte bank:
//
//     byte = (i / 8) * 32 + (i % 2) * 16 + l      shift = 2 * ((i % 8) / 2)
//
// Lane t covers the same two weight quads as the Q4_K kernel -- {4t, 128+4t} -- so
// the two kernels share an x access pattern: quad1 is sub-block t/4 and quad2 is
// sub-block t/4 + 8, one from each bank. Each quad is therefore four consecutive
// bytes at one shift, which is exactly one u32, and one scale pair.
@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let sub_warp = local_id.x / 32u;
    let tid = local_id.x % 32u;
    let row = workgroup_id.x * 4u + sub_warp;
    let m_base = workgroup_id.y * 4u;
    let base_sdata = sub_warp * 32u;

    if (m_base >= params.m) {
        return;
    }
    let m_rem = min(4u, params.m - m_base);
    let kv = params.k / 4u;

    var sum0: f32 = 0.0;
    var sum1: f32 = 0.0;
    var sum2: f32 = 0.0;
    var sum3: f32 = 0.0;

    if (row < params.n) {
        let i1 = tid / 4u;                        // 0..7   quad1's sub-block
        let i2 = i1 + 8u;                         // 8..15  quad2's sub-block
        // Both scale bytes live in the first four words (16 bytes) of the block.
        let nshift = 2u * (tid / 8u);             // 0, 2, 4, 6 -> the four fields
        let u1 = ((tid / 4u) % 2u) * 4u + (tid % 4u); // qs word for quad1
        let u2 = u1 + 8u;                         // qs word for quad2 (second bank)
        var row_off: u32;
        var x_base: u32;
        if (params.id != 0u) {
            // One dispatch covers every selected expert. The output row names
            // a *position* in the routing order; `idx` maps that position to
            // an expert in the resident bank, so the weights are read in
            // place and no gather is needed.
            let pos = row / params.rows_per_expert;
            let expert = u32(idx[params.idx_off + pos]);
            let local = row - pos * params.rows_per_expert;
            row_off = expert * params.expert_words + local * params.blocks_per_row * 21u;
            x_base = (pos / params.pos_div) * params.x_stride;
        } else {
            row_off = params.w_off + row * params.blocks_per_row * 21u;
            x_base = 0u;
        }

        for (var b: u32 = 0u; b < params.blocks_per_row; b = b + 1u) {
            let bo = row_off + b * 21u;
            let d_dmin = unpack2x16float(w[bo + 20u]);
            let b1 = (w[bo + i1 / 4u] >> (8u * (i1 % 4u))) & 0xFFu;
            let b2 = (w[bo + i2 / 4u] >> (8u * (i2 % 4u))) & 0xFFu;
            let d = d_dmin.x;
            let dmin = d_dmin.y;

            let qa = w[bo + 4u + u1];
            let qb = w[bo + 4u + u2];
            let qa4 = vec4<f32>(unpack4xU8((qa >> nshift) & 0x03030303u));
            let qb4 = vec4<f32>(unpack4xU8((qb >> nshift) & 0x03030303u));

            let dl1 = d * f32(b1 & 0x0Fu);
            let ml1 = dmin * f32(b1 >> 4u);
            let dl2 = d * f32(b2 & 0x0Fu);
            let ml2 = dmin * f32(b2 >> 4u);

            // x vec4 indices: quad1 -> b*64 + tid, quad2 -> b*64 + 32 + tid
            let xi_a = b * 64u + tid;
            let xi_b = b * 64u + 32u + tid;
            let ones = vec4<f32>(1.0);

            {
                let xb = x_base + (m_base + 0u) * kv;
                let xa = x[xb + xi_a];
                let xc = x[xb + xi_b];
                sum0 += dl1 * dot(qa4, xa) + dl2 * dot(qb4, xc)
                      - ml1 * dot(xa, ones) - ml2 * dot(xc, ones);
            }
            if (m_rem > 1u) {
                let xb = x_base + (m_base + 1u) * kv;
                let xa = x[xb + xi_a];
                let xc = x[xb + xi_b];
                sum1 += dl1 * dot(qa4, xa) + dl2 * dot(qb4, xc)
                      - ml1 * dot(xa, ones) - ml2 * dot(xc, ones);
            }
            if (m_rem > 2u) {
                let xb = x_base + (m_base + 2u) * kv;
                let xa = x[xb + xi_a];
                let xc = x[xb + xi_b];
                sum2 += dl1 * dot(qa4, xa) + dl2 * dot(qb4, xc)
                      - ml1 * dot(xa, ones) - ml2 * dot(xc, ones);
            }
            if (m_rem > 3u) {
                let xb = x_base + (m_base + 3u) * kv;
                let xa = x[xb + xi_a];
                let xc = x[xb + xi_b];
                sum3 += dl1 * dot(qa4, xa) + dl2 * dot(qb4, xc)
                      - ml1 * dot(xa, ones) - ml2 * dot(xc, ones);
            }
        }
    }

    sdata[base_sdata + tid] = vec4<f32>(sum0, sum1, sum2, sum3);
    workgroupBarrier();

    if (tid < 16u) {
        sdata[base_sdata + tid] += sdata[base_sdata + tid + 16u];
    }
    workgroupBarrier();

    if (tid < 8u) {
        sdata[base_sdata + tid] += sdata[base_sdata + tid + 8u];
    }
    workgroupBarrier();

    if (tid < 4u) {
        sdata[base_sdata + tid] += sdata[base_sdata + tid + 4u];
    }
    workgroupBarrier();

    if (tid < 2u) {
        sdata[base_sdata + tid] += sdata[base_sdata + tid + 2u];
    }
    workgroupBarrier();

    if (tid == 0u && row < params.n) {
        let total = sdata[base_sdata] + sdata[base_sdata + 1u];
        let n_len = params.n;
        let o0 = params.y_off + m_base * n_len + row;
        if (params.accum == 0u) { y[o0] = total.x; } else { y[o0] += total.x; }
        if (m_rem > 1u) {
            let o1 = params.y_off + (m_base + 1u) * n_len + row;
            if (params.accum == 0u) { y[o1] = total.y; } else { y[o1] += total.y; }
        }
        if (m_rem > 2u) {
            let o2 = params.y_off + (m_base + 2u) * n_len + row;
            if (params.accum == 0u) { y[o2] = total.z; } else { y[o2] += total.z; }
        }
        if (m_rem > 3u) {
            let o3 = params.y_off + (m_base + 3u) * n_len + row;
            if (params.accum == 0u) { y[o3] = total.w; } else { y[o3] += total.w; }
        }
    }
}
