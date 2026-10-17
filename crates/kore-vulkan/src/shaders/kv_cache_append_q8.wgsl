struct KvAppendQ8Uniforms {
    numel_k: u32,       // bytes: num_new * cols_k
    dst_k_offset: u32,  // bytes: row_offset * cols_k
    src_k_offset: u32,  // bytes
    numel_v: u32,       // bytes: num_new * cols_v
    dst_v_offset: u32,  // bytes: row_offset * cols_v
    src_v_offset: u32,  // bytes
}

@group(0) @binding(0) var<uniform> params: KvAppendQ8Uniforms;
// Packed byte buffers (4 bytes per u32 word, little-endian). All offsets/counts are
// 4-aligned because the host asserts cols % 4 == 0.
@group(0) @binding(1) var<storage, read> src_k: array<u32>;
@group(0) @binding(2) var<storage, read> src_v: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst_k: array<u32>;
@group(0) @binding(4) var<storage, read_write> dst_v: array<u32>;

// Q8 (INT8) KV-cache append: copies new K rows and new V rows into their resident
// caches at the append offset in a single dispatch, word-addressable. Per-token
// scales are appended separately by the elementwise copy path.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x; // word index within this dispatch
    if (idx < (params.numel_k >> 2u)) {
        dst_k[(params.dst_k_offset >> 2u) + idx] = src_k[(params.src_k_offset >> 2u) + idx];
    }
    if (idx < (params.numel_v >> 2u)) {
        dst_v[(params.dst_v_offset >> 2u) + idx] = src_v[(params.src_v_offset >> 2u) + idx];
    }
}
