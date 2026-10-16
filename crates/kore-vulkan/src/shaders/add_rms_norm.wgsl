struct AddRmsNormUniforms {
    rows: u32,
    cols: u32,
    eps: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: AddRmsNormUniforms;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> weight: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_res: array<f32>;     // a + b (residual)
@group(0) @binding(5) var<storage, read_write> out_norm: array<f32>;   // rms_norm(a + b) * weight

var<workgroup> shared_sq: array<f32, 256>;

// Fused residual add + RMSNorm: writes both the residual stream (needed by the
// next residual add) and its normalized version in one pass.
@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = wg_id.x;
    if (row >= params.rows) {
        return;
    }

    let tid = local_id.x;
    let cols = params.cols;
    let base_idx = row * cols;

    var sum_sq: f32 = 0.0;
    for (var c: u32 = tid; c < cols; c = c + 256u) {
        let val = a[base_idx + c] + b[base_idx + c];
        out_res[base_idx + c] = val;
        sum_sq = sum_sq + val * val;
    }

    shared_sq[tid] = sum_sq;
    workgroupBarrier();

    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_sq[tid] = shared_sq[tid] + shared_sq[tid + stride];
        }
        workgroupBarrier();
    }

    let rms = inverseSqrt(shared_sq[0] / f32(cols) + params.eps);
    for (var c: u32 = tid; c < cols; c = c + 256u) {
        out_norm[base_idx + c] = out_res[base_idx + c] * rms * weight[c];
    }
}
