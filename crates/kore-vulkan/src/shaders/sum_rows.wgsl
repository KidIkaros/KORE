// Sum `rows` contiguous rows of `n` f32 values into one row.
//
// The routed MoE `down` projection produces one `[n]` row per selected expert;
// this collapses them in a single dispatch instead of an add per expert.

struct SumRowsUniforms {
    rows: u32,
    n: u32,
    /// When set, `bias[i]` is added to the collapsed row. The routed `down`
    /// sum and the shared expert's output used to be two dispatches.
    has_bias: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: SumRowsUniforms;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) {
        return;
    }
    var acc = 0.0;
    for (var r = 0u; r < params.rows; r = r + 1u) {
        acc = acc + x[r * params.n + i];
    }
    if (params.has_bias != 0u) {
        y[i] = acc + bias[i];
    } else {
        y[i] = acc;
    }
}
