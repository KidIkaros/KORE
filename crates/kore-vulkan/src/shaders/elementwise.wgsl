struct ElementwiseUniforms {
    numel: u32,
    op_type: u32, // 0=add, 1=mul, 2=sub, 3=div, 4=add_scalar, 5=mul_scalar, 6=silu, 7=swiglu, 8=copy_offset, 9=swiglu_split, 10=x_sigmoid_gate
    scalar: f32,
    a_numel: u32,
    b_numel: u32,
    dst_offset: u32,
    src_offset: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: ElementwiseUniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.numel) {
        if (params.op_type == 8u) {
            C[params.dst_offset + idx] = A[params.src_offset + idx];
            return;
        }
        if (params.op_type == 9u) {
            let inter = params.a_numel;
            let row = idx / inter;
            let col = idx % inter;
            let base = row * (inter * 2u) + col;
            let g_val = A[base];
            let u_val = A[base + inter];
            C[idx] = (g_val / (1.0 + exp(-g_val))) * u_val;
            return;
        }
        var a_idx = idx;
        if (params.a_numel != params.numel) {
            a_idx = idx % params.a_numel;
        }
        var b_idx = idx;
        if (params.b_numel != params.numel) {
            b_idx = idx % params.b_numel;
        }
        let a = A[a_idx];
        let b = B[b_idx];
        if (params.op_type == 0u) {
            C[idx] = a + b;
        } else if (params.op_type == 1u) {
            C[idx] = a * b;
        } else if (params.op_type == 2u) {
            C[idx] = a - b;
        } else if (params.op_type == 3u) {
            C[idx] = a / b;
        } else if (params.op_type == 4u) {
            C[idx] = a + params.scalar;
        } else if (params.op_type == 5u) {
            C[idx] = a * params.scalar;
        } else if (params.op_type == 6u) {
            let sig = 1.0 / (1.0 + exp(-a));
            C[idx] = a * sig;
        } else if (params.op_type == 7u) {
            let sig = 1.0 / (1.0 + exp(-a));
            C[idx] = a * sig * b;
        } else if (params.op_type == 10u) {
            C[idx] = a / (1.0 + exp(-b));
        }
    }
}

