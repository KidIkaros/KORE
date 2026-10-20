// Device-side expert gather: dst[s] = bank[idx[s]] for s in 0..k.
//
// This is the piece that removes the routing readback from the critical path:
// the expert ids never leave the device, so a layer can route, gather and
// compute in one command stream. `idx` holds f32 ids (tensors are f32 in this
// lane); they are exact integers below 2^24.
//
// One workgroup per *chunk* of an expert, not one per expert: an expert is a
// few hundred kB and a single workgroup cannot saturate memory, so a one-group
// gather is latency-bound and dominates the step. `groups` is chosen on the
// host from the expert size.
struct GatherUniforms {
    per_words: u32,
    k: u32,
    idx_off: u32,
    groups: u32,
    chunk: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> p: GatherUniforms;
@group(0) @binding(1) var<storage, read> bank: array<u32>;
@group(0) @binding(2) var<storage, read> idx: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lx: vec3<u32>,
) {
    let s = wg.x / p.groups;
    let g = wg.x % p.groups;
    if (s >= p.k) {
        return;
    }
    let e = u32(idx[p.idx_off + s]);
    let base_s = e * p.per_words;
    let base_d = s * p.per_words;
    let start = g * p.chunk;
    let end = min(start + p.chunk, p.per_words);
    for (var i = start + lx.x; i < end; i = i + 256u) {
        dst[base_d + i] = bank[base_s + i];
    }
}
