struct Q80Uniforms {
    m: u32,
    n: u32,
    k: u32,
    blocks_per_row: u32,
}

@group(0) @binding(0) var<uniform> params: Q80Uniforms;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> sdata: array<vec4<f32>, 128>;

// Batch-Invariant Q8_0 GEMV (@workgroup_size(128) = 4 rows per workgroup).
//
// A Q8_0 block is 32 int8 quants under one f16 scale -- an eighth of a K-quant
// block -- so the lane map is per-quad rather than per-block: lane `tid` owns
// quants [4*(tid%8), 4*(tid%8)+4) of the blocks whose index is `tid/8` mod 4,
// and walks them step 4. One warp therefore covers four blocks per iteration,
// and each of those blocks gets its eight quads read by eight lanes, so the x
// vec4s a warp loads for one block are 128 contiguous bytes. The reduction
// below is the same 32-lane tree the K-quant kernels use, which is what keeps
// the result independent of the batch width `m`.
//
// The payload is a repacked copy (`kore_kernels::q8_0::repack_q8_0`): the quants
// come first, eight words of them, so a lane's quad sits at word `quad`, and the
// f16 scale rides in the low half of word 8. The file's own 34-byte layout is
// not word-aligned, hence the two-byte pad.
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
        let quad = tid % 8u;      // which 4 of the block's 32 quants
        let stride = tid / 8u;    // four blocks in flight per warp
        let row_off = row * params.blocks_per_row * 9u;  // 9 u32 per packed block

        for (var b: u32 = stride; b < params.blocks_per_row; b = b + 4u) {
            let bo = row_off + b * 9u;

            let d = unpack2x16float(w[bo + 8u] & 0xFFFFu).x;
            let u4 = unpack4xU8(w[bo + quad]);
            // The quants are signed bytes; the 0x80 bit is the sign, so the
            // unsigned view has to be biased back down before it is a value.
            let q4 = vec4<f32>(select(
                vec4<i32>(u4),
                vec4<i32>(u4) - vec4<i32>(256),
                u4 >= vec4<u32>(128u)
            ));

            let xi = b * 8u + quad;   // x as vec4: one block is eight vec4s
            {
                let xa = x[(m_base + 0u) * kv + xi];
                sum0 += d * dot(q4, xa);
            }
            if (m_rem > 1u) {
                let xa = x[(m_base + 1u) * kv + xi];
                sum1 += d * dot(q4, xa);
            }
            if (m_rem > 2u) {
                let xa = x[(m_base + 2u) * kv + xi];
                sum2 += d * dot(q4, xa);
            }
            if (m_rem > 3u) {
                let xa = x[(m_base + 3u) * kv + xi];
                sum3 += d * dot(q4, xa);
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
