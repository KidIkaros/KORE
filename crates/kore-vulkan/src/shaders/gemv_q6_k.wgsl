struct Q6KUniforms {
    m: u32,
    n: u32,
    k: u32,
    blocks_per_row: u32,
}

@group(0) @binding(0) var<uniform> params: Q6KUniforms;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> sdata: array<vec4<f32>, 128>;

// Batch-Invariant Small-Batch Q6_K GEMV (@workgroup_size(128) = 4 rows per workgroup).
// Weights are repacked once at upload: each 210B Q6_K block padded to 224B (56 u32) so
// all payload reads are aligned. Lane t covers quads at positions {4t, 128+4t} so lane
// x-reads are 16B-contiguous across the warp. ql nibble select = (t>>4), qh 2-bit field
// = (t>>3), sub-block scales = (t>>2) and (t>>2)+8.
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
        let nshift = (tid >> 4u) * 4u;            // ql nibble: 0=lo, 4=hi
        let fshift = ((tid >> 3u) & 3u) * 2u;     // qh 2-bit field
        let fvec = vec4<u32>(fshift);
        let u_ql = tid & 15u;                     // ql u32 idx (quad1); +16 for quad2
        let u_qh = tid & 7u;                      // qh u32 idx (quad1); +8 for quad2
        let s_lo = tid >> 2u;                     // sub-block idx 0..7; +8 for quad2
        let scw_idx = 48u + (s_lo >> 2u);         // scale u32 for subs 0-3 or 4-7
        let scw_idx_hi = 50u + (s_lo >> 2u);      // scale u32 for subs 8-11 or 12-15
        let sc_byte_sh = (s_lo & 3u) * 8u;
        let row_off = row * params.blocks_per_row * 56u;

        for (var b: u32 = 0u; b < params.blocks_per_row; b = b + 1u) {
            let bo = row_off + b * 56u;

            let d = unpack2x16float(w[bo + 52u] & 0xFFFFu).x;
            let raw_lo = (w[bo + scw_idx] >> sc_byte_sh) & 0xFFu;
            let raw_hi = (w[bo + scw_idx_hi] >> sc_byte_sh) & 0xFFu;
            let s1 = d * select(f32(raw_lo), f32(raw_lo) - 256.0, raw_lo >= 128u);
            let s2 = d * select(f32(raw_hi), f32(raw_hi) - 256.0, raw_hi >= 128u);

            let qa_ql = w[bo + u_ql];
            let qb_ql = w[bo + u_ql + 16u];
            let qa_qh = w[bo + 32u + u_qh];
            let qb_qh = w[bo + 32u + u_qh + 8u];

            let lo_a = unpack4xU8((qa_ql >> nshift) & 0x0F0F0F0Fu);
            let lo_b = unpack4xU8((qb_ql >> nshift) & 0x0F0F0F0Fu);
            let hi_a = (unpack4xU8(qa_qh) >> fvec) & vec4<u32>(3u);
            let hi_b = (unpack4xU8(qb_qh) >> fvec) & vec4<u32>(3u);
            let qa4 = vec4<f32>(vec4<i32>(lo_a | (hi_a << vec4<u32>(4u))) - vec4<i32>(32));
            let qb4 = vec4<f32>(vec4<i32>(lo_b | (hi_b << vec4<u32>(4u))) - vec4<i32>(32));

            let xi_a = b * 64u + tid;
            let xi_b = b * 64u + 32u + tid;

            {
                let xb = (m_base + 0u) * kv;
                let xa = x[xb + xi_a];
                let xc = x[xb + xi_b];
                sum0 += s1 * dot(qa4, xa) + s2 * dot(qb4, xc);
            }
            if (m_rem > 1u) {
                let xb = (m_base + 1u) * kv;
                let xa = x[xb + xi_a];
                let xc = x[xb + xi_b];
                sum1 += s1 * dot(qa4, xa) + s2 * dot(qb4, xc);
            }
            if (m_rem > 2u) {
                let xb = (m_base + 2u) * kv;
                let xa = x[xb + xi_a];
                let xc = x[xb + xi_b];
                sum2 += s1 * dot(qa4, xa) + s2 * dot(qb4, xc);
            }
            if (m_rem > 3u) {
                let xb = (m_base + 3u) * kv;
                let xa = x[xb + xi_a];
                let xc = x[xb + xi_b];
                sum3 += s1 * dot(qa4, xa) + s2 * dot(qb4, xc);
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
