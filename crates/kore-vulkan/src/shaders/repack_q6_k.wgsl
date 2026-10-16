struct RepackParams {
    blocks_per_row: u32,
    n_rows: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: RepackParams;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<u32>;

fn load_u32_at(byte_off: u32) -> u32 {
    let word = w[byte_off >> 2u];
    let sh = (byte_off & 3u) * 8u;
    if (sh == 0u) {
        return word;
    }
    return (word >> sh) | (w[(byte_off >> 2u) + 1u] << (32u - sh));
}

// Copies each 210-byte Q6_K block into a 224-byte (56 u32) slot so downstream GEMV
// payload reads are u32-aligned. Each workgroup covers ROWS_PER_WG rows.
const ROWS_PER_WG: u32 = 8u;
@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row_base = workgroup_id.x * ROWS_PER_WG;
    let n_out = ROWS_PER_WG * params.blocks_per_row * 56u;
    for (var i: u32 = local_id.x; i < n_out; i = i + 64u) {
        let row = row_base + i / (params.blocks_per_row * 56u);
        if (row >= params.n_rows) {
            continue;
        }
        let rem = i % (params.blocks_per_row * 56u);
        let b = rem / 56u;
        let j = rem % 56u;
        let src = (row * params.blocks_per_row + b) * 210u + j * 4u;
        var v: u32 = 0u;
        if (j < 52u) {
            v = load_u32_at(src);
        } else if (j == 52u) {
            v = load_u32_at(src) & 0xFFFFu;
        }
        out_buf[(row * params.blocks_per_row + b) * 56u + j] = v;
    }
}
