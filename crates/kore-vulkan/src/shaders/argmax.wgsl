struct ArgmaxUniforms {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: ArgmaxUniforms;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_indices: array<u32>;

var<workgroup> shared_val: array<f32, 256>;
var<workgroup> shared_idx: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = wg_id.x;
    if (row >= params.rows) {
        return;
    }

    let lx = local_id.x;
    let cols = params.cols;
    let base = row * cols;

    var best_val: f32 = -1e38;
    var best_idx: u32 = 0u;

    for (var i: u32 = lx; i < cols; i = i + 256u) {
        let v = logits[base + i];
        // NaN-safe comparison: only update when v > best_val
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    shared_val[lx] = best_val;
    shared_idx[lx] = best_idx;
    workgroupBarrier();

    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (lx < stride) {
            let other_val = shared_val[lx + stride];
            let other_idx = shared_idx[lx + stride];
            if (other_val > shared_val[lx] || (other_val == shared_val[lx] && other_idx < shared_idx[lx])) {
                shared_val[lx] = other_val;
                shared_idx[lx] = other_idx;
            }
        }
        workgroupBarrier();
    }

    if (lx == 0u) {
        out_indices[row] = shared_idx[0];
    }
}
