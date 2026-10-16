struct Q4KUniforms {
    m: u32,
    n: u32,
    k: u32,
    blocks_per_row: u32,
}

@group(0) @binding(0) var<uniform> params: Q4KUniforms;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> sdata: array<vec4<f32>, 128>;

// Extract (d*sc_lo, dm*mn_lo, d*sc_hi, dm*mn_hi) for sub-blocks `s_lo` in 0..3
// (direct 6-bit fields) and `s_hi` = s_lo+4 in 4..7 (interleaved packing).
fn q4_k_scales(w0: u32, w1: u32, w2: u32, d: f32, dmin: f32, s_lo: u32) -> vec4<f32> {
    let sh8 = s_lo * 8u;   // s_lo in 0..3; s_hi = s_lo + 4 -> byte j = s_lo
    let sc_lo = f32((w0 >> sh8) & 0x3Fu);
    let mn_lo = f32((w1 >> sh8) & 0x3Fu);
    let sc_hi = f32(((w2 >> sh8) & 0x0Fu) | (((w0 >> (sh8 + 6u)) & 0x03u) << 4u));
    let mn_hi = f32(((w2 >> (sh8 + 4u)) & 0x0Fu) | (((w1 >> (sh8 + 6u)) & 0x03u) << 4u));
    return vec4<f32>(d * sc_lo, dmin * mn_lo, d * sc_hi, dmin * mn_hi);
}

// Batch-Invariant Small-Batch Q4_K GEMV (@workgroup_size(128) = 4 rows per workgroup).
// Lane t covers the two weight quads at positions {4t, 128+4t} of each block so that
// per-instruction lane x-reads are 16B-contiguous across the whole warp (no shared-mem
// staging needed). Nibble half (lo/hi) is (t>>3)&1; sub-blocks are (t>>3) and (t>>3)+4.
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
        let s_lo = tid >> 3u;             // 0..3 -> sub-blocks s_lo and s_lo+4
        let hi_nib = (tid >> 3u) & 1u;    // 0 for t<8 and 16-23, 1 otherwise? see below
        let nshift = hi_nib * 4u;         // nibble shift: 0 = lo, 4 = hi
        let u1 = (tid & 7u) + ((tid >> 4u) * 8u); // payload u32 for quad1
        let u2 = 16u + u1;                // payload u32 for quad2
        let row_off = row * params.blocks_per_row * 36u;

        for (var b: u32 = 0u; b < params.blocks_per_row; b = b + 1u) {
            let bo = row_off + b * 36u;
            let d_dmin = unpack2x16float(w[bo]);
            let sc = q4_k_scales(w[bo + 1u], w[bo + 2u], w[bo + 3u], d_dmin.x, d_dmin.y, s_lo);

            let qa = w[bo + 4u + u1];
            let qb = w[bo + 4u + u2];
            let qa4 = vec4<f32>(unpack4xU8((qa >> nshift) & 0x0F0F0F0Fu));
            let qb4 = vec4<f32>(unpack4xU8((qb >> nshift) & 0x0F0F0F0Fu));

            // x vec4 indices: quad1 -> b*64 + tid, quad2 -> b*64 + 32 + tid
            let xi_a = b * 64u + tid;
            let xi_b = b * 64u + 32u + tid;

            {
                let xb = (m_base + 0u) * kv;
                let xa = x[xb + xi_a];
                let xc = x[xb + xi_b];
                sum0 += sc.x * dot(qa4, xa) + sc.z * dot(qb4, xc)
                      - sc.y * dot(xa, vec4<f32>(1.0)) - sc.w * dot(xc, vec4<f32>(1.0));
            }
            if (m_rem > 1u) {
                let xb = (m_base + 1u) * kv;
                let xa = x[xb + xi_a];
                let xc = x[xb + xi_b];
                sum1 += sc.x * dot(qa4, xa) + sc.z * dot(qb4, xc)
                      - sc.y * dot(xa, vec4<f32>(1.0)) - sc.w * dot(xc, vec4<f32>(1.0));
            }
            if (m_rem > 2u) {
                let xb = (m_base + 2u) * kv;
                let xa = x[xb + xi_a];
                let xc = x[xb + xi_b];
                sum2 += sc.x * dot(qa4, xa) + sc.z * dot(qb4, xc)
                      - sc.y * dot(xa, vec4<f32>(1.0)) - sc.w * dot(xc, vec4<f32>(1.0));
            }
            if (m_rem > 3u) {
                let xb = (m_base + 3u) * kv;
                let xa = x[xb + xi_a];
                let xc = x[xb + xi_b];
                sum3 += sc.x * dot(qa4, xa) + sc.z * dot(qb4, xc)
                      - sc.y * dot(xa, vec4<f32>(1.0)) - sc.w * dot(xc, vec4<f32>(1.0));
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
