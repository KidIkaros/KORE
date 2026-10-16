struct HadamardUniforms {
    total_blocks: u32,
    block_size: u32,
    scale: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: HadamardUniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> shared_buf: array<f32, 256>;

@compute @workgroup_size(256)
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

    if (lx < bsize) {
        shared_buf[lx] = input[base + lx];
    }
    workgroupBarrier();

    var h: u32 = 1u;
    loop {
        if (h >= bsize) {
            break;
        }
        var val: f32 = 0.0;
        if (lx < bsize) {
            let partner = lx ^ h;
            let a = shared_buf[lx];
            let b = shared_buf[partner];
            if ((lx & h) == 0u) {
                val = a + b;
            } else {
                val = b - a;
            }
        }
        workgroupBarrier();
        if (lx < bsize) {
            shared_buf[lx] = val;
        }
        workgroupBarrier();
        h = h * 2u;
    }

    if (lx < bsize) {
        output[base + lx] = shared_buf[lx] * params.scale;
    }
}
