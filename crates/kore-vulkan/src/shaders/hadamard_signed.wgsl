struct HadamardSignedUniforms {
    total_blocks: u32,
    block_size: u32,   // 512
    scale: f32,        // 1/sqrt(block)
    mode: u32,         // 0 = signs-then-WHT (forward), 1 = WHT-then-signs (restore)
    cols: u32,         // row width; sign index is row-local (block base % cols)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: HadamardSignedUniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> signs: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// Two shared buffers so every butterfly stage is a single barrier with no
// intermediate register array. The previous in-place version staged the four
// per-thread values in `var vals: array<f32,4>` indexed by a loop counter; that
// dynamic indexing spills to local memory and cost more than the transform.
//
// dst[j] = src[j] + src[j^h] for j with bit h clear, src[j^h] - src[j] otherwise:
// the same per-element arithmetic, so results are bit-identical to the in-place
// form. The stages are unrolled explicitly because naga rejects workgroup-memory
// pointers as function parameters.
var<workgroup> buf_a: array<f32, 512>;
var<workgroup> buf_b: array<f32, 512>;

// One workgroup per 512-block; 128 threads x 4 elements each.
@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let block_idx = wg_id.x;
    if (block_idx >= params.total_blocks) {
        return;
    }
    let lx = local_id.x;
    let base = block_idx * params.block_size;
    // Row-local sign base: the sign table covers one row, so an [rows, cols]
    // input reuses the same pattern on every row.
    let sbase = (base % params.cols);

    // load (mode 0: apply signs on the way in)
    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = lx * 4u + i;
        var v = input[base + j];
        if (params.mode == 0u) {
            v = v * signs[sbase + j];
        }
        buf_a[j] = v;
    }
    workgroupBarrier();

    // h = 1  (a -> b)
    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = lx * 4u + i; let p = j ^ 1u;
        let a = buf_a[j]; let b = buf_a[p];
        if ((j & 1u) == 0u) { buf_b[j] = a + b; } else { buf_b[j] = b - a; }
    }
    workgroupBarrier();
    // h = 2  (b -> a)
    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = lx * 4u + i; let p = j ^ 2u;
        let a = buf_b[j]; let b = buf_b[p];
        if ((j & 2u) == 0u) { buf_a[j] = a + b; } else { buf_a[j] = b - a; }
    }
    workgroupBarrier();
    // h = 4  (a -> b)
    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = lx * 4u + i; let p = j ^ 4u;
        let a = buf_a[j]; let b = buf_a[p];
        if ((j & 4u) == 0u) { buf_b[j] = a + b; } else { buf_b[j] = b - a; }
    }
    workgroupBarrier();
    // h = 8  (b -> a)
    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = lx * 4u + i; let p = j ^ 8u;
        let a = buf_b[j]; let b = buf_b[p];
        if ((j & 8u) == 0u) { buf_a[j] = a + b; } else { buf_a[j] = b - a; }
    }
    workgroupBarrier();
    // h = 16 (a -> b)
    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = lx * 4u + i; let p = j ^ 16u;
        let a = buf_a[j]; let b = buf_a[p];
        if ((j & 16u) == 0u) { buf_b[j] = a + b; } else { buf_b[j] = b - a; }
    }
    workgroupBarrier();
    // h = 32 (b -> a)
    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = lx * 4u + i; let p = j ^ 32u;
        let a = buf_b[j]; let b = buf_b[p];
        if ((j & 32u) == 0u) { buf_a[j] = a + b; } else { buf_a[j] = b - a; }
    }
    workgroupBarrier();
    // h = 64 (a -> b)
    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = lx * 4u + i; let p = j ^ 64u;
        let a = buf_a[j]; let b = buf_a[p];
        if ((j & 64u) == 0u) { buf_b[j] = a + b; } else { buf_b[j] = b - a; }
    }
    workgroupBarrier();
    // h = 128 (b -> a)
    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = lx * 4u + i; let p = j ^ 128u;
        let a = buf_b[j]; let b = buf_b[p];
        if ((j & 128u) == 0u) { buf_a[j] = a + b; } else { buf_a[j] = b - a; }
    }
    workgroupBarrier();
    // h = 256 (a -> b)
    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = lx * 4u + i; let p = j ^ 256u;
        let a = buf_a[j]; let b = buf_a[p];
        if ((j & 256u) == 0u) { buf_b[j] = a + b; } else { buf_b[j] = b - a; }
    }
    workgroupBarrier();

    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = lx * 4u + i;
        var v = buf_b[j] * params.scale;
        if (params.mode == 1u) {
            v = v * signs[sbase + j];
        }
        output[base + j] = v;
    }
}
