struct SoftmaxUniforms {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<uniform> params: SoftmaxUniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row < params.rows) {
        let base_idx = row * params.cols;
        var max_val: f32 = -1e30;
        for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
            let v = input[base_idx + c];
            if (v > max_val) {
                max_val = v;
            }
        }

        var sum_exp: f32 = 0.0;
        for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
            let e = exp(input[base_idx + c] - max_val);
            output[base_idx + c] = e;
            sum_exp = sum_exp + e;
        }

        let inv_sum = 1.0 / sum_exp;
        for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
            output[base_idx + c] = output[base_idx + c] * inv_sum;
        }
    }
}
