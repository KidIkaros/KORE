struct TransposeUniforms {
    out_rows: u32,
    out_cols: u32,
    offset: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: TransposeUniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let s = global_id.x; // out_col (0..out_cols)
    let d = global_id.y; // out_row (0..out_rows)

    if (d < params.out_rows && s < params.out_cols) {
        // Output is row-major [out_rows, out_cols]
        let out_idx = d * params.out_cols + s;
        // Input had swapped dimensions [out_cols, out_rows] in row-major memory
        let in_idx = params.offset + s * params.out_rows + d;
        output[out_idx] = input[in_idx];
    }
}
