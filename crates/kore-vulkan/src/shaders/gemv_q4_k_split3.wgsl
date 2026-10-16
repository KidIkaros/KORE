struct Q4KSplit3Uniforms {
    m: u32,
    n: u32,          // total rows = nq + nk + nv
    k: u32,
    blocks_per_row: u32,
    nq: u32,         // rows 0..nq          -> y_q
    nk: u32,         // rows nq..nq+nk      -> y_k
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Q4KSplit3Uniforms;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<u32>;   // q|k|v rows concatenated
@group(0) @binding(3) var<storage, read_write> y_q: array<f32>;
@group(0) @binding(4) var<storage, read_write> y_k: array<f32>;
@group(0) @binding(5) var<storage, read_write> y_v: array<f32>;

var<workgroup> sdata: array<vec4<f32>, 128>;

fn q4_k_scales(w0: u32, w1: u32, w2: u32, d: f32, dmin: f32, s_lo: u32) -> vec4<f32> {
    let sh8 = s_lo * 8u;
    let sc_lo = f32((w0 >> sh8) & 0x3Fu);
    let mn_lo = f32((w1 >> sh8) & 0x3Fu);
    let sc_hi = f32(((w2 >> sh8) & 0x0Fu) | (((w0 >> (sh8 + 6u)) & 0x03u) << 4u));
    let mn_hi = f32(((w2 >> (sh8 + 4u)) & 0x0Fu) | (((w1 >> (sh8 + 6u)) & 0x03u) << 4u));
    return vec4<f32>(d * sc_lo, dmin * mn_lo, d * sc_hi, dmin * mn_hi);
}

// Fused Q|K|V projection: same coalesced-x lane mapping as gemv_q4_k over the
// concatenated weight rows; each sub-warp's output row is routed to y_q/y_k/y_v.
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
        let s_lo = tid >> 3u;
        let hi_nib = (tid >> 3u) & 1u;
        let nshift = hi_nib * 4u;
        let u1 = (tid & 7u) + ((tid >> 4u) * 8u);
        let u2 = 16u + u1;
        let row_off = row * params.blocks_per_row * 36u;

        for (var b: u32 = 0u; b < params.blocks_per_row; b = b + 1u) {
            let bo = row_off + b * 36u;
            let d_dmin = unpack2x16float(w[bo]);
            let sc = q4_k_scales(w[bo + 1u], w[bo + 2u], w[bo + 3u], d_dmin.x, d_dmin.y, s_lo);

            let qa = w[bo + 4u + u1];
            let qb = w[bo + 4u + u2];
            let qa4 = vec4<f32>(unpack4xU8((qa >> nshift) & 0x0F0F0F0Fu));
            let qb4 = vec4<f32>(unpack4xU8((qb >> nshift) & 0x0F0F0F0Fu));

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
        // Route output row into the q/k/v partitions.
        var dst_off: u32 = row;
        var n_len: u32 = params.nq;
        var sel: u32 = 0u;
        if (row >= params.nq) {
            dst_off = row - params.nq;
            n_len = params.nk;
            sel = 1u;
        }
        if (row >= params.nq + params.nk) {
            dst_off = row - params.nq - params.nk;
            n_len = params.n - params.nq - params.nk;
            sel = 2u;
        }
        if (sel == 0u) {
            y_q[(m_base + 0u) * n_len + dst_off] = total.x;
            if (m_rem > 1u) { y_q[(m_base + 1u) * n_len + dst_off] = total.y; }
            if (m_rem > 2u) { y_q[(m_base + 2u) * n_len + dst_off] = total.z; }
            if (m_rem > 3u) { y_q[(m_base + 3u) * n_len + dst_off] = total.w; }
        } else if (sel == 1u) {
            y_k[(m_base + 0u) * n_len + dst_off] = total.x;
            if (m_rem > 1u) { y_k[(m_base + 1u) * n_len + dst_off] = total.y; }
            if (m_rem > 2u) { y_k[(m_base + 2u) * n_len + dst_off] = total.z; }
            if (m_rem > 3u) { y_k[(m_base + 3u) * n_len + dst_off] = total.w; }
        } else {
            y_v[(m_base + 0u) * n_len + dst_off] = total.x;
            if (m_rem > 1u) { y_v[(m_base + 1u) * n_len + dst_off] = total.y; }
            if (m_rem > 2u) { y_v[(m_base + 2u) * n_len + dst_off] = total.z; }
            if (m_rem > 3u) { y_v[(m_base + 3u) * n_len + dst_off] = total.w; }
        }
    }
}
