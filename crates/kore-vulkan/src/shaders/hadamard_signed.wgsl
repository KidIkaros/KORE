struct HadamardSignedUniforms {
    total_blocks: u32,
    block_size: u32,   // 512
    scale: f32,        // 1/sqrt(block)
    mode: u32,         // 0 = signs-then-WHT (forward), 1 = WHT-then-signs (restore)
}

@group(0) @binding(0) var<uniform> params: HadamardSignedUniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> signs: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// One workgroup per 512-block; 128 threads x 4 elements each.
var<workgroup> buf: array<f32, 512>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let block_idx = wg_id.x;
    if (block_idx >= params.total_blocks) {
        return;
    }
    let bsize = params.block_size;
    let lx = local_id.x;
    let base = block_idx * bsize;
    let per = bsize / 128u;

    // load (mode 0: apply signs on the way in)
    for (var i = 0u; i < per; i = i + 1u) {
        let j = lx * per + i;
        var v = input[base + j];
        if (params.mode == 0u) {
            v = v * signs[base + j];
        }
        buf[j] = v;
    }
    workgroupBarrier();

    // in-place butterfly: all reads into registers, barrier, then writes
    var h = 1u;
    loop {
        if (h >= bsize) {
            break;
        }
        var vals: array<f32, 4>;
        for (var i = 0u; i < per; i = i + 1u) {
            let j = lx * per + i;
            let p = j ^ h;
            let a = buf[j];
            let b = buf[p];
            if ((j & h) == 0u) {
                vals[i] = a + b;
            } else {
                vals[i] = b - a;
            }
        }
        workgroupBarrier();
        for (var i = 0u; i < per; i = i + 1u) {
            buf[lx * per + i] = vals[i];
        }
        workgroupBarrier();
        h = h * 2u;
    }

    for (var i = 0u; i < per; i = i + 1u) {
        let j = lx * per + i;
        var v = buf[j] * params.scale;
        if (params.mode == 1u) {
            v = v * signs[base + j];
        }
        output[base + j] = v;
    }
}
