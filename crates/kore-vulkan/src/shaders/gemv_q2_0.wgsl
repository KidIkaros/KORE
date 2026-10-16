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

// Q2_0: 64 elems/block, 18 bytes = fp16 scale + 16 bytes of 2-bit codes.
// Code c in {0,1,2,3} -> value c-1 in {-1,0,+1,+2}, times scale.
// Blocks are byte-addressed (18 is not a multiple of 4).

fn fp16_to_f32(h: u32) -> f32 {
    let sign = (h >> 15u) & 1u;
    let exp = (h >> 10u) & 0x1Fu;
    let mant = h & 0x3FFu;
    var f: f32;
    if (exp == 0u) {
        f = f32(mant) * 0x1p-24;
    } else if (exp == 31u) {
        f = select(1e30, 0.0, mant != 0u);
    } else {
        f = (1024.0 + f32(mant)) * exp2(f32(i32(exp) - 25));
    }
    return select(f, -f, sign == 1u);
}

fn get_byte(row_off: u32, byte_idx: u32) -> u32 {
    let addr = row_off + byte_idx;
    return (w[addr >> 2u] >> ((addr & 3u) * 8u)) & 0xFFu;
}

// Lane t covers elements {2t, 2t+1} of each block: code byte 2+(t>>1),
// shift (t&1)*4. x pair = one vec4 element (xv.xy or xv.zw) per block pos.
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

    var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    if (row < params.n) {
        let row_off = row * params.blocks_per_row * 18u;
        let byte_idx = 2u + (tid >> 1u);          // code byte within block
        let shift = (tid & 1u) * 4u;
        let xvec_idx = tid >> 1u;                 // vec4 index within block
        let xlo = (tid & 1u) == 0u;               // .xy or .zw

        for (var b: u32 = 0u; b < params.blocks_per_row; b = b + 1u) {
            let bo = row_off + b * 18u;
            let lo = get_byte(bo, 0u);
            let hi = get_byte(bo, 1u);
            let d = fp16_to_f32(lo | (hi << 8u));
            let cb = get_byte(bo, byte_idx);
            let c0 = f32(i32((cb >> shift) & 3u) - 1);
            let c1 = f32(i32((cb >> (shift + 2u)) & 3u) - 1);

            let xi = b * 16u + xvec_idx;          // vec4 index into x row 0
            var xv = x[xi];
            var xp = select(xv.zw, xv.xy, xlo);
            sum.x += d * (c0 * xp.x + c1 * xp.y);
            if (m_rem > 1u) {
                xv = x[xi + kv];
                xp = select(xv.zw, xv.xy, xlo);
                sum.y += d * (c0 * xp.x + c1 * xp.y);
            }
            if (m_rem > 2u) {
                xv = x[xi + 2u * kv];
                xp = select(xv.zw, xv.xy, xlo);
                sum.z += d * (c0 * xp.x + c1 * xp.y);
            }
            if (m_rem > 3u) {
                xv = x[xi + 3u * kv];
                xp = select(xv.zw, xv.xy, xlo);
                sum.w += d * (c0 * xp.x + c1 * xp.y);
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
