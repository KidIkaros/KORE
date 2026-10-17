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

var<workgroup> sdata: array<vec4<f32>, 128>;

// Repacked Q2_0 row layout (u32-aligned, same byte count):
//   codes:  block b's 16 code bytes at word offset b*4 (elems 16b..16b+15 are
//           bytes 0..15 of the block, i.e. word b*4 + 0..3)
//   scales: u16 at byte bpr*16 + 2b -> u32 idx bpr*4 + b/2, half (b&1)*16
//
// Blocks are stored contiguously so a warp's code reads stay coalesced.
// One warp (32 lanes) owns one output row. Per iteration the warp covers 2
// blocks (128 contiguous elements): lane t owns elements i*128 + 4t .. +3,
// i.e. block (2i + t/16) and code byte (t%16). x is read as a vec4 at index
// i*32 + t (fully coalesced).

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let lx = local_id.x;

    let sub_warp = lx / 32u;
    let tid = lx % 32u;
    let row = workgroup_id.x * 4u + sub_warp;
    let m_base = workgroup_id.y * 4u;
    let base_sdata = sub_warp * 32u;

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

    var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    if (row < params.n) {
        let row_w = row * row_words;
        // Each warp (32 lanes) owns one output row and covers 2 blocks (128
        // elements) per iteration: lane t reads the vec4 at i*32 + t, which is
        // elements i*128 + 4t (block i*2 + t/16, code byte t%16).
        let n_iter = (bpr + 1u) / 2u;
        for (var i = 0u; i < n_iter; i = i + 1u) {
            let b = i * 2u + b_in8;
            if (b >= bpr) {
                break;
            }
            let sw = w[row_w + bpr * 4u + (b >> 1u)];
            let d = unpack2x16float(sw >> ((b & 1u) * 16u)).x;
            let cb = (w[row_w + b * 4u + cw_j] >> cw_sh) & 0xFFu;
            let lv = vec4<f32>(
                f32(i32(cb & 3u) - 1),
                f32(i32((cb >> 2u) & 3u) - 1),
                f32(i32((cb >> 4u) & 3u) - 1),
                f32(i32((cb >> 6u) & 3u) - 1),
            );
            let xi = i * 32u + tid;

            var a = lv * x[xi];
            sum.x += d * (a.x + a.y + a.z + a.w);
            if (m_rem > 1u) {
                a = lv * x[xi + kv];
                sum.y += d * (a.x + a.y + a.z + a.w);
            }
            if (m_rem > 2u) {
                a = lv * x[xi + 2u * kv];
                sum.z += d * (a.x + a.y + a.z + a.w);
            }
            if (m_rem > 3u) {
                a = lv * x[xi + 3u * kv];
                sum.w += d * (a.x + a.y + a.z + a.w);
            }
        }
    }

    sdata[base_sdata + tid] = sum;
    workgroupBarrier();

    if (tid < 16u) { sdata[base_sdata + tid] += sdata[base_sdata + tid + 16u]; }
    workgroupBarrier();
    if (tid < 8u) { sdata[base_sdata + tid] += sdata[base_sdata + tid + 8u]; }
    workgroupBarrier();
    if (tid < 4u) { sdata[base_sdata + tid] += sdata[base_sdata + tid + 4u]; }
    workgroupBarrier();
    if (tid < 2u) { sdata[base_sdata + tid] += sdata[base_sdata + tid + 2u]; }
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
