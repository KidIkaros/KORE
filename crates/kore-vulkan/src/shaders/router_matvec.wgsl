// Router matvec for the device-routed MoE lane.
//
// The routing projection is the one F32 matrix in the step: a
// `[n_experts, hidden]` GGUF tensor, row-major, so each expert's row is
// contiguous. The generic GEMV stores it transposed (`[hidden, n_experts]`)
// and gives every output column its own workgroup; the B reads then stride by
// `n_experts` and touch one 4-byte word per cache line. At 256 experts that
// costs ~0.18 ms a layer, and two of them a layer is ~15 ms of a 30 ms step.
//
// Here one workgroup owns one expert and walks its row contiguously, so the
// loads coalesce: 128 lanes read 128 consecutive floats per iteration and the
// reduction is a tree over shared memory.
//
// The k-reduction order differs from the generic GEMV (and therefore from the
// host lane) by float reassociation; the router logits move by ~1e-6 and the
// token stream is checked against the host lane after the change.

struct RouterUniforms {
    n: u32,
    k: u32,
    // Activation rows. A verify batch routes every column through the same
    // weight rows, so `wg.y` selects the column and the reduction order per
    // column is untouched -- a batched pass is bit-identical to one call per
    // column.
    cols: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: RouterUniforms;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wg.x;
    let col = wg.y;
    let t = lid.x;
    let base = row * p.k;
    let xb = col * p.k;

    var acc = 0.0;
    for (var i = t; i < p.k; i = i + 128u) {
        acc = acc + w[base + i] * x[xb + i];
    }
    red[t] = acc;
    workgroupBarrier();

    var st = 64u;
    loop {
        if (st == 0u) {
            break;
        }
        if (t < st) {
            red[t] = red[t] + red[t + st];
        }
        workgroupBarrier();
        st = st / 2u;
    }

    if (t == 0u) {
        y[col * p.n + row] = red[0];
    }
}
