// Ternary (Q2_0) matrix-vector product.
//
// Repacked row layout (u32-aligned, same byte count as the on-disk Q2_0 row):
//   codes:  block b's 16 code bytes at word offset b*4 (elems 16b..16b+15 are
//           bytes 0..15 of the block, i.e. word b*4 + 0..3)
//   scales: u16 at byte bpr*16 + 2b -> u32 idx bpr*4 + b/2, half (b&1)*16
//
// Each warp accumulates RPW output rows together. y[o] = sum_k x[k]*W[o][k],
// so for a fixed k the same x values feed every row: reading x once and reusing
// it across the row group cuts the kernel's dominant traffic by RPW. x costs
// 4 bytes/element against 0.28 for a Q2_0 weight, so x reads dominate the bytes
// this kernel moves.
//
// A lane owns one code *word* -- 16 k-elements, a quarter of a block -- for
// each of the RPW rows, so a warp covers 512 contiguous elements per
// k-iteration:
//
//   lane t: elements [i*512 + 16t, i*512 + 16t + 16)   (4 coalesced x vec4)
//           block  i*8 + t/4   -> code word i*32 + t    (fully coalesced)
//           scale  word i*4 + t/8, half ((t/4) & 1)*16  (8-way broadcast)
//
// That is 12 loads per 64 weights. The earlier shape gave each lane a single
// code *byte* (4 elements) and spent 4 code + 4 scale loads per 16 weights,
// i.e. 36 per 64. The load stream, not the arithmetic, caps this kernel: the
// same shape at m=4 (which amortises those loads over four x columns) runs 2.5x
// more efficiently per weight, and a two-term cost model fit to that pair puts
// 58% of the m=1 cost on the load side.
//
// The m columns accumulate one at a time into separate components of the same
// accumulators, and each component's arithmetic is independent of m_rem: same
// lane coverage, same rows, same per-row summation order. An output row is
// therefore bit-identical at m=1 and inside a wider batch, so decode and
// batched spec-decode verification select the same token.

struct Q2Uniforms {
    m: u32,
    n: u32,
    k: u32,
    blocks_per_row: u32,
}

@group(0) @binding(0) var<uniform> params: Q2Uniforms;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

// Rows per warp; a workgroup is 4 sub-warps, so 4*RPW rows per workgroup.
const RPW: u32 = 4u;

// Reduction scratch: 4 sub-warps x RPW rows x 32 lanes.
var<workgroup> sdata: array<vec4<f32>, 512>;

// The four code deltas (c_i - 1) held in byte `j` of code word `ww`.
fn byte_deltas(ww: u32, j: u32) -> vec4<f32> {
    let cb = (ww >> (j * 8u)) & 0xFFu;
    return vec4<f32>(
        f32(i32(cb & 3u) - 1),
        f32(i32((cb >> 2u) & 3u) - 1),
        f32(i32((cb >> 4u) & 3u) - 1),
        f32(i32((cb >> 6u) & 3u) - 1),
    );
}

// Dot of one code word (16 elements) against their 16 x values.
fn dot_wide(ww: u32, xa: vec4<f32>, xb: vec4<f32>, xc: vec4<f32>, xd: vec4<f32>) -> f32 {
    let p = byte_deltas(ww, 0u) * xa + byte_deltas(ww, 1u) * xb
        + byte_deltas(ww, 2u) * xc + byte_deltas(ww, 3u) * xd;
    return p.x + p.y + p.z + p.w;
}

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let lx = local_id.x;
    let sub_warp = lx / 32u;      // 0..3
    let tid = lx % 32u;           // 0..31
    let row0 = workgroup_id.x * (4u * RPW) + sub_warp * RPW;
    let m_base = workgroup_id.y * 4u;

    if (m_base >= params.m) {
        return;
    }
    let m_rem = min(4u, params.m - m_base);
    let kv = params.k / 4u;
    let bpr = params.blocks_per_row;
    let row_words = bpr * 18u / 4u;

    var acc0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var acc1 = acc0;
    var acc2 = acc0;
    var acc3 = acc0;

    // ---- one code word (16 elements) per lane per row ----
    //
    // A lane owns one code word of one block for each of the RPW rows, so a
    // warp covers 512 contiguous elements per k-iteration with 4 code loads,
    // 4 scale loads and 4 x vec4 loads: 12 loads per 64 weights. The earlier
    // shape gave each lane a single code *byte* (4 elements) and spent 4 code
    // + 4 scale loads per 16 weights, i.e. 36 per 64.
    //
    // The m columns are accumulated one at a time into separate components of
    // the same accumulators, and each component's arithmetic is independent of
    // m_rem: same lane coverage, same rows, same per-row summation order. An
    // output row is therefore bit-identical at m=1 and inside a wider batch, so
    // decode and batched spec-decode verification select the same token.
    let q = tid >> 2u;                 // block within an 8-block group
    let sw = bpr * 4u + (q >> 1u);     // this block's scale word
    let sw_sh = (q & 1u) * 16u;        // and which half of it
    let n_iter_w = (bpr + 7u) / 8u;
    let n_len = params.n;

    for (var i = 0u; i < n_iter_w; i = i + 1u) {
        let b = i * 8u + q;
        if (b >= bpr) {
            continue;
        }
        let xi = i * 128u + 4u * tid;
        let cw = i * 32u + tid;        // this lane's code word
        let swo = sw + i * 4u;

        // The lane's rows: code word and scale, read once and reused by every
        // m column.
        let row1 = row0 + 1u;
        let row2 = row0 + 2u;
        let row3 = row0 + 3u;
        var ww0 = 0u;
        var ww1 = 0u;
        var ww2 = 0u;
        var ww3 = 0u;
        var dd0 = 0.0;
        var dd1 = 0.0;
        var dd2 = 0.0;
        var dd3 = 0.0;
        if (row0 < n_len) {
            let rw = row0 * row_words;
            ww0 = w[rw + cw];
            dd0 = unpack2x16float(w[rw + swo] >> sw_sh).x;
        }
        if (row1 < n_len) {
            let rw = row1 * row_words;
            ww1 = w[rw + cw];
            dd1 = unpack2x16float(w[rw + swo] >> sw_sh).x;
        }
        if (row2 < n_len) {
            let rw = row2 * row_words;
            ww2 = w[rw + cw];
            dd2 = unpack2x16float(w[rw + swo] >> sw_sh).x;
        }
        if (row3 < n_len) {
            let rw = row3 * row_words;
            ww3 = w[rw + cw];
            dd3 = unpack2x16float(w[rw + swo] >> sw_sh).x;
        }

        // Column 0.
        let xa0 = x[xi];
        let xb0 = x[xi + 1u];
        let xc0 = x[xi + 2u];
        let xd0 = x[xi + 3u];
        if (row0 < n_len) { acc0.x += dd0 * dot_wide(ww0, xa0, xb0, xc0, xd0); }
        if (row1 < n_len) { acc1.x += dd1 * dot_wide(ww1, xa0, xb0, xc0, xd0); }
        if (row2 < n_len) { acc2.x += dd2 * dot_wide(ww2, xa0, xb0, xc0, xd0); }
        if (row3 < n_len) { acc3.x += dd3 * dot_wide(ww3, xa0, xb0, xc0, xd0); }

        if (m_rem > 1u) {
            let xa1 = x[xi + kv];
            let xb1 = x[xi + kv + 1u];
            let xc1 = x[xi + kv + 2u];
            let xd1 = x[xi + kv + 3u];
            if (row0 < n_len) { acc0.y += dd0 * dot_wide(ww0, xa1, xb1, xc1, xd1); }
            if (row1 < n_len) { acc1.y += dd1 * dot_wide(ww1, xa1, xb1, xc1, xd1); }
            if (row2 < n_len) { acc2.y += dd2 * dot_wide(ww2, xa1, xb1, xc1, xd1); }
            if (row3 < n_len) { acc3.y += dd3 * dot_wide(ww3, xa1, xb1, xc1, xd1); }
        }
        if (m_rem > 2u) {
            let xa2 = x[xi + 2u * kv];
            let xb2 = x[xi + 2u * kv + 1u];
            let xc2 = x[xi + 2u * kv + 2u];
            let xd2 = x[xi + 2u * kv + 3u];
            if (row0 < n_len) { acc0.z += dd0 * dot_wide(ww0, xa2, xb2, xc2, xd2); }
            if (row1 < n_len) { acc1.z += dd1 * dot_wide(ww1, xa2, xb2, xc2, xd2); }
            if (row2 < n_len) { acc2.z += dd2 * dot_wide(ww2, xa2, xb2, xc2, xd2); }
            if (row3 < n_len) { acc3.z += dd3 * dot_wide(ww3, xa2, xb2, xc2, xd2); }
        }
        if (m_rem > 3u) {
            let xa3 = x[xi + 3u * kv];
            let xb3 = x[xi + 3u * kv + 1u];
            let xc3 = x[xi + 3u * kv + 2u];
            let xd3 = x[xi + 3u * kv + 3u];
            if (row0 < n_len) { acc0.w += dd0 * dot_wide(ww0, xa3, xb3, xc3, xd3); }
            if (row1 < n_len) { acc1.w += dd1 * dot_wide(ww1, xa3, xb3, xc3, xd3); }
            if (row2 < n_len) { acc2.w += dd2 * dot_wide(ww2, xa3, xb3, xc3, xd3); }
            if (row3 < n_len) { acc3.w += dd3 * dot_wide(ww3, xa3, xb3, xc3, xd3); }
        }
    }

    // Reduce each of the RPW rows across this warp's 32 lanes.
    let sb = sub_warp * (RPW * 32u);
    sdata[sb + tid] = acc0;
    sdata[sb + 32u + tid] = acc1;
    sdata[sb + 64u + tid] = acc2;
    sdata[sb + 96u + tid] = acc3;
    workgroupBarrier();

    if (tid < 16u) {
        sdata[sb + tid] += sdata[sb + tid + 16u];
        sdata[sb + 32u + tid] += sdata[sb + 32u + tid + 16u];
        sdata[sb + 64u + tid] += sdata[sb + 64u + tid + 16u];
        sdata[sb + 96u + tid] += sdata[sb + 96u + tid + 16u];
    }
    workgroupBarrier();
    if (tid < 8u) {
        sdata[sb + tid] += sdata[sb + tid + 8u];
        sdata[sb + 32u + tid] += sdata[sb + 32u + tid + 8u];
        sdata[sb + 64u + tid] += sdata[sb + 64u + tid + 8u];
        sdata[sb + 96u + tid] += sdata[sb + 96u + tid + 8u];
    }
    workgroupBarrier();
    if (tid < 4u) {
        sdata[sb + tid] += sdata[sb + tid + 4u];
        sdata[sb + 32u + tid] += sdata[sb + 32u + tid + 4u];
        sdata[sb + 64u + tid] += sdata[sb + 64u + tid + 4u];
        sdata[sb + 96u + tid] += sdata[sb + 96u + tid + 4u];
    }
    workgroupBarrier();
    if (tid < 2u) {
        sdata[sb + tid] += sdata[sb + tid + 2u];
        sdata[sb + 32u + tid] += sdata[sb + 32u + tid + 2u];
        sdata[sb + 64u + tid] += sdata[sb + 64u + tid + 2u];
        sdata[sb + 96u + tid] += sdata[sb + 96u + tid + 2u];
    }
    workgroupBarrier();

    if (tid == 0u) {
        let n_len = params.n;
        if (row0 < n_len) {
            let total = sdata[sb] + sdata[sb + 1u];
            y[(m_base + 0u) * n_len + row0] = total.x;
            if (m_rem > 1u) { y[(m_base + 1u) * n_len + row0] = total.y; }
            if (m_rem > 2u) { y[(m_base + 2u) * n_len + row0] = total.z; }
            if (m_rem > 3u) { y[(m_base + 3u) * n_len + row0] = total.w; }
        }
        if (row0 + 1u < n_len) {
            let total = sdata[sb + 32u] + sdata[sb + 33u];
            y[(m_base + 0u) * n_len + row0 + 1u] = total.x;
            if (m_rem > 1u) { y[(m_base + 1u) * n_len + row0 + 1u] = total.y; }
            if (m_rem > 2u) { y[(m_base + 2u) * n_len + row0 + 1u] = total.z; }
            if (m_rem > 3u) { y[(m_base + 3u) * n_len + row0 + 1u] = total.w; }
        }
        if (row0 + 2u < n_len) {
            let total = sdata[sb + 64u] + sdata[sb + 65u];
            y[(m_base + 0u) * n_len + row0 + 2u] = total.x;
            if (m_rem > 1u) { y[(m_base + 1u) * n_len + row0 + 2u] = total.y; }
            if (m_rem > 2u) { y[(m_base + 2u) * n_len + row0 + 2u] = total.z; }
            if (m_rem > 3u) { y[(m_base + 3u) * n_len + row0 + 2u] = total.w; }
        }
        if (row0 + 3u < n_len) {
            let total = sdata[sb + 96u] + sdata[sb + 97u];
            y[(m_base + 0u) * n_len + row0 + 3u] = total.x;
            if (m_rem > 1u) { y[(m_base + 1u) * n_len + row0 + 3u] = total.y; }
            if (m_rem > 2u) { y[(m_base + 2u) * n_len + row0 + 3u] = total.z; }
            if (m_rem > 3u) { y[(m_base + 3u) * n_len + row0 + 3u] = total.w; }
        }
    }
}
