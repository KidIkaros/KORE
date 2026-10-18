struct Q3KUniforms {
    m: u32,
    n: u32,
    k: u32,
    blocks_per_row: u32,
}

@group(0) @binding(0) var<uniform> params: Q3KUniforms;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> sdata: array<vec4<f32>, 128>;

// Six-bit signed sub-block scale: four low bits from the first eight bytes
// (one nibble each) and two high bits packed four-per-byte in bytes 8..12.
fn q3_k_scale(word: u32, top: u32, i: u32, high: bool) -> f32 {
    let sh = 8u * (i % 4u) + select(0u, 4u, high);
    let nib = (word >> sh) & 0x0Fu;
    let t = (top >> (8u * (i % 4u) + 2u * (i / 4u))) & 0x03u;
    return f32(i32(nib | (t << 4u)) - 32);
}

// Batch-Invariant Small-Batch Q3_K GEMV (@workgroup_size(128) = 4 rows per workgroup).
//
// The host pads each 110-byte block to 112 so every block is word-aligned:
//
//     words  0..7   hmask[32]
//     words  8..23  qs[64]
//     words 24..26  scales[12]
//     word  27      d as f16 in the low half, two pad bytes in the high half
//
// Three bits per weight: two from `qs` (same layout as Q2_K) and one from
// `hmask`, where a *set* mask bit means the high bit is zero. Lane t covers the
// same two weight quads as the Q2_K and Q4_K kernels -- {4t, 128+4t} -- so all
// three share an x access pattern, and each quad is one u32 at one 2-bit field.
//
// The four mask bits of a quad are four consecutive bits of one hmask word, so a
// word shifted right by the bit index and masked with 0x01010101 yields them in
// weight order. Quad1 reads bit t/8 and quad2 reads bit t/8 + 4 of the same word.
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
        let nshift = 2u * (tid / 8u);             // 0, 2, 4, 6
        let u1 = ((tid / 4u) % 2u) * 4u + (tid % 4u);
        let u2 = u1 + 8u;
        let hm = 4u * ((tid / 4u) % 2u) + (tid % 4u);  // word 0..7
        let bit1 = tid / 8u;                           // 0..3
        let bit2 = bit1 + 4u;                          // 4..7
        let row_off = row * params.blocks_per_row * 28u;

        for (var b: u32 = 0u; b < params.blocks_per_row; b = b + 1u) {
            let bo = row_off + b * 28u;
            let d = unpack2x16float(w[bo + 27u]).x;

            // The nibbles live at bytes 96..104 (words 24..25); the two-bit high
            // parts at bytes 104..108 (word 26).
            let dl1 = d * q3_k_scale(w[bo + 24u + i1 / 4u], w[bo + 26u], i1, false);
            let dl2 = d * q3_k_scale(w[bo + 24u + (i2 - 8u) / 4u], w[bo + 26u], i2, true);

            let qa = w[bo + 8u + u1];
            let qb = w[bo + 8u + u2];
            let ql1 = vec4<f32>(unpack4xU8((qa >> nshift) & 0x03030303u));
            let ql2 = vec4<f32>(unpack4xU8((qb >> nshift) & 0x03030303u));

            let hw = w[bo + hm];
            let hi1 = vec4<f32>(unpack4xU8((hw >> bit1) & 0x01010101u));
            let hi2 = vec4<f32>(unpack4xU8((hw >> bit2) & 0x01010101u));
            let off1 = vec4<f32>(1.0) - hi1;
            let off2 = vec4<f32>(1.0) - hi2;

            // x vec4 indices: quad1 -> b*64 + tid, quad2 -> b*64 + 32 + tid
            let xi_a = b * 64u + tid;
            let xi_b = b * 64u + 32u + tid;

            {
                let xb = (m_base + 0u) * kv;
                let xa = x[xb + xi_a];
                let xc = x[xb + xi_b];
                sum0 += dl1 * (dot(ql1, xa) - 4.0 * dot(off1, xa))
                      + dl2 * (dot(ql2, xc) - 4.0 * dot(off2, xc));
            }
            if (m_rem > 1u) {
                let xb = (m_base + 1u) * kv;
                let xa = x[xb + xi_a];
                let xc = x[xb + xi_b];
                sum1 += dl1 * (dot(ql1, xa) - 4.0 * dot(off1, xa))
                      + dl2 * (dot(ql2, xc) - 4.0 * dot(off2, xc));
            }
            if (m_rem > 2u) {
                let xb = (m_base + 2u) * kv;
                let xa = x[xb + xi_a];
                let xc = x[xb + xi_b];
                sum2 += dl1 * (dot(ql1, xa) - 4.0 * dot(off1, xa))
                      + dl2 * (dot(ql2, xc) - 4.0 * dot(off2, xc));
            }
            if (m_rem > 3u) {
                let xb = (m_base + 3u) * kv;
                let xa = x[xb + xi_a];
                let xc = x[xb + xi_b];
                sum3 += dl1 * (dot(ql1, xa) - 4.0 * dot(off1, xa))
                      + dl2 * (dot(ql2, xc) - 4.0 * dot(off2, xc));
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
        y[(m_base + 0u) * n_len + row] = total.x;
        if (m_rem > 1u) { y[(m_base + 1u) * n_len + row] = total.y; }
        if (m_rem > 2u) { y[(m_base + 2u) * n_len + row] = total.z; }
        if (m_rem > 3u) { y[(m_base + 3u) * n_len + row] = total.w; }
    }
}
