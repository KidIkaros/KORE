struct KvAppendPairUniforms {
    numel_k: u32,       // num_new * cols_k
    numel_v: u32,       // num_new * cols_v
    k_dst_off: u32,     // row_offset * cols_k
    v_dst_off: u32,     // row_offset * cols_v
}

@group(0) @binding(0) var<uniform> params: KvAppendPairUniforms;
@group(0) @binding(1) var<storage, read> src_k: array<f32>;
@group(0) @binding(2) var<storage, read> src_v: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst_k: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst_v: array<f32>;

// Fused KV-cache append: copies new K rows and new V rows into their resident
// caches at the append offset in a single dispatch.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.numel_k) {
        dst_k[params.k_dst_off + idx] = src_k[idx];
    } else if (idx < params.numel_k + params.numel_v) {
        let i = idx - params.numel_k;
        dst_v[params.v_dst_off + i] = src_v[i];
    }
}
