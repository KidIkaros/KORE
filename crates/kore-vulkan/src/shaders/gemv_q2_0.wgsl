// Ternary (Q2_0) matrix-vector product.
//
// Repacked row layout (u32-aligned, same byte count as the on-disk Q2_0 row):
//   codes:  block b's 16 code bytes at word offset b*4 (elems 16b..16b+15 are
//           bytes 0..15 of the block, i.e. word b*4 + 0..3)
//   scales: u16 at byte bpr*16 + 2b -> u32 idx bpr*4 + b/2, half (b&1)*16
//
// Blocks are stored contiguously so a warp's code reads stay coalesced.
// A warp covers 2 blocks (128 contiguous elements) per k-iteration: lane t
// reads the vec4 at index i*32 + t, i.e. elements i*128 + 4t (block
// 2i + t/16, code byte t%16); x is read as a vec4 (fully coalesced).
//
// Each warp accumulates RPW output rows together. y[o] = sum_k x[k]*W[o][k],
// so for a fixed k the same x values feed every row: reading x once and
// reusing it across the row group cuts the kernel's dominant traffic by RPW.
// x costs 4 bytes/element against 0.28 for a Q2_0 weight, so x reads were
// ~93% of the bytes this kernel moved.

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

// The four code deltas (c_i - 1) packed into one byte of a block.
fn code_deltas(row_w: u32, b: u32, cw_j: u32, cw_sh: u32) -> vec4<f32> {
    let cb = (w[row_w + b * 4u + cw_j] >> cw_sh) & 0xFFu;
    return vec4<f32>(
        f32(i32(cb & 3u) - 1),
        f32(i32((cb >> 2u) & 3u) - 1),
        f32(i32((cb >> 4u) & 3u) - 1),
        f32(i32((cb >> 6u) & 3u) - 1),
    );
}

// Block b's fp16 scale for a row.
fn block_scale(row_w: u32, b: u32, bpr: u32) -> f32 {
    return unpack2x16float(w[row_w + bpr * 4u + (b >> 1u)] >> ((b & 1u) * 16u)).x;
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

    // lane's block-within-8 and code-byte constants
    let b_in8 = tid >> 4u;           // 0..1
    let byte_p = tid & 15u;          // code byte within block
    let cw_j = byte_p >> 2u;         // which code u32 word
    let cw_sh = (byte_p & 3u) * 8u;  // byte position within the word

    var acc0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var acc1 = acc0;
    var acc2 = acc0;
    var acc3 = acc0;

    let n_iter = (bpr + 1u) / 2u;
    for (var i = 0u; i < n_iter; i = i + 1u) {
        let b = i * 2u + b_in8;
        if (b >= bpr) {
            break;
        }
        let xi = i * 32u + tid;

        // One x read, reused by all RPW rows.
        let xv0 = x[xi];
        var xv1 = xv0;
        var xv2 = xv0;
        var xv3 = xv0;
        if (m_rem > 1u) { xv1 = x[xi + kv]; }
        if (m_rem > 2u) { xv2 = x[xi + 2u * kv]; }
        if (m_rem > 3u) { xv3 = x[xi + 3u * kv]; }

        if (row0 < params.n) {
            let rw = row0 * row_words;
            let d = block_scale(rw, b, bpr);
            let lv = code_deltas(rw, b, cw_j, cw_sh);
            let t = lv * xv0;
            acc0.x += d * (t.x + t.y + t.z + t.w);
            if (m_rem > 1u) { let a = lv * xv1; acc0.y += d * (a.x + a.y + a.z + a.w); }
            if (m_rem > 2u) { let a = lv * xv2; acc0.z += d * (a.x + a.y + a.z + a.w); }
            if (m_rem > 3u) { let a = lv * xv3; acc0.w += d * (a.x + a.y + a.z + a.w); }
        }
        let row1 = row0 + 1u;
        if (row1 < params.n) {
            let rw = row1 * row_words;
            let d = block_scale(rw, b, bpr);
            let lv = code_deltas(rw, b, cw_j, cw_sh);
            let t = lv * xv0;
            acc1.x += d * (t.x + t.y + t.z + t.w);
            if (m_rem > 1u) { let a = lv * xv1; acc1.y += d * (a.x + a.y + a.z + a.w); }
            if (m_rem > 2u) { let a = lv * xv2; acc1.z += d * (a.x + a.y + a.z + a.w); }
            if (m_rem > 3u) { let a = lv * xv3; acc1.w += d * (a.x + a.y + a.z + a.w); }
        }
        let row2 = row0 + 2u;
        if (row2 < params.n) {
            let rw = row2 * row_words;
            let d = block_scale(rw, b, bpr);
            let lv = code_deltas(rw, b, cw_j, cw_sh);
            let t = lv * xv0;
            acc2.x += d * (t.x + t.y + t.z + t.w);
            if (m_rem > 1u) { let a = lv * xv1; acc2.y += d * (a.x + a.y + a.z + a.w); }
            if (m_rem > 2u) { let a = lv * xv2; acc2.z += d * (a.x + a.y + a.z + a.w); }
            if (m_rem > 3u) { let a = lv * xv3; acc2.w += d * (a.x + a.y + a.z + a.w); }
        }
        let row3 = row0 + 3u;
        if (row3 < params.n) {
            let rw = row3 * row_words;
            let d = block_scale(rw, b, bpr);
            let lv = code_deltas(rw, b, cw_j, cw_sh);
            let t = lv * xv0;
            acc3.x += d * (t.x + t.y + t.z + t.w);
            if (m_rem > 1u) { let a = lv * xv1; acc3.y += d * (a.x + a.y + a.z + a.w); }
            if (m_rem > 2u) { let a = lv * xv2; acc3.z += d * (a.x + a.y + a.z + a.w); }
            if (m_rem > 3u) { let a = lv * xv3; acc3.w += d * (a.x + a.y + a.z + a.w); }
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
