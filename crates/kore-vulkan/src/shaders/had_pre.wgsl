// Fused "pre-op + Hadamard" kernel.
//
// Each decode-layer norm is followed immediately by a signed Walsh-Hadamard
// transform feeding the next mat-vec. Doing them as two dispatches costs two
// launch/drain latencies in the strictly serial decode chain; this merges the
// pair into one kernel.
//
// Modes:
//   0 = add + rms_norm + hadamard   (residual add folded into the norm)
//   1 = rms_norm + hadamard
//   2 = swiglu + hadamard
//   3 = sigmoid_mul + hadamard
//
// Outputs:
//   mode 0 -> out_res  = a + b            out_norm = rms_norm(a+b)*w   out_had = WHT(signs*h)
//   mode 1 ->                              out_norm = rms_norm(a)*w     out_had = WHT(signs*h)
//   mode 2 ->                                                           out_had = WHT(signs*swiglu(a,b))
//   mode 3 ->                                                           out_had = WHT(signs*(a/(1+exp(-b))))
//
// Every arithmetic step reproduces the corresponding standalone shader
// (rms_norm.wgsl / add_rms_norm.wgsl / elementwise.wgsl / hadamard_signed.wgsl)
// operation-for-operation so results are bit-identical.
//
// Grid: one workgroup per (row, 512-block). Modes 0/1 need the whole-row sum of
// squares; each workgroup recomputes it independently (deterministic, so all
// workgroups agree) which keeps every workgroup on the same critical path
// instead of serialising the blocks inside one workgroup.

struct HadPreUniforms {
    rows: u32,
    cols: u32,
    blocks: u32,   // cols / 512
    mode: u32,     // 0..3
    eps: f32,
    scale: f32,    // 1/sqrt(512)
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: HadPreUniforms;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> weight: array<f32>;
@group(0) @binding(4) var<storage, read> signs: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_res: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_norm: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_had: array<f32>;

var<workgroup> shared_sq: array<f32, 256>;
var<workgroup> buf_a: array<f32, 512>;
var<workgroup> buf_b: array<f32, 512>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let gid = wg_id.x;
    let blocks = params.blocks;
    let row = gid / blocks;
    let blk = gid % blocks;
    if (row >= params.rows) {
        return;
    }

    let cols = params.cols;
    let base = row * cols;
    let bb = base + blk * 512u;
    let cidx = blk * 512u + tid * 4u; // first column of this thread's group
    let mode = params.mode;

    var rms: f32 = 1.0;
    if (mode <= 1u) {
        // Same accumulation order as rms_norm.wgsl / add_rms_norm.wgsl: 256
        // logical accumulators, each summing columns with stride 256, then the
        // 128->1 tree. 128 threads therefore carry two accumulators each.
        var s0: f32 = 0.0;
        var s1: f32 = 0.0;
        for (var c: u32 = tid; c < cols; c = c + 256u) {
            var v = a[base + c];
            if (mode == 0u) {
                v = v + b[base + c];
            }
            s0 = s0 + v * v;
        }
        for (var c: u32 = tid + 128u; c < cols; c = c + 256u) {
            var v = a[base + c];
            if (mode == 0u) {
                v = v + b[base + c];
            }
            s1 = s1 + v * v;
        }
        shared_sq[tid] = s0;
        shared_sq[tid + 128u] = s1;
        workgroupBarrier();

        for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
            if (tid < stride) {
                shared_sq[tid] = shared_sq[tid] + shared_sq[tid + stride];
            }
            workgroupBarrier();
        }
        rms = inverseSqrt(shared_sq[0] / f32(cols) + params.eps);
    }

    // ---- pre-op + sign application into buf_a ----
    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = tid * 4u + i;
        var v: f32 = 0.0;
        if (mode == 0u) {
            let r = a[bb + j] + b[bb + j];
            out_res[bb + j] = r;
            let h = r * rms * weight[cidx + i];
            out_norm[bb + j] = h;
            v = h;
        } else if (mode == 1u) {
            let h = a[bb + j] * rms * weight[cidx + i];
            out_norm[bb + j] = h;
            v = h;
        } else if (mode == 2u) {
            let gv = a[bb + j];
            let sig = 1.0 / (1.0 + exp(-gv));
            v = gv * sig * b[bb + j];
        } else {
            v = a[bb + j] / (1.0 + exp(-b[bb + j]));
        }
        buf_a[j] = v * signs[bb + j];
    }
    workgroupBarrier();

    // ---- 9 butterfly stages (512 = 2^9), one barrier each ----
    // h = 1  (a -> b)
    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = tid * 4u + i; let p = j ^ 1u;
        let x = buf_a[j]; let y = buf_a[p];
        if ((j & 1u) == 0u) { buf_b[j] = x + y; } else { buf_b[j] = y - x; }
    }
    workgroupBarrier();
    // h = 2  (b -> a)
    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = tid * 4u + i; let p = j ^ 2u;
        let x = buf_b[j]; let y = buf_b[p];
        if ((j & 2u) == 0u) { buf_a[j] = x + y; } else { buf_a[j] = y - x; }
    }
    workgroupBarrier();
    // h = 4  (a -> b)
    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = tid * 4u + i; let p = j ^ 4u;
        let x = buf_a[j]; let y = buf_a[p];
        if ((j & 4u) == 0u) { buf_b[j] = x + y; } else { buf_b[j] = y - x; }
    }
    workgroupBarrier();
    // h = 8  (b -> a)
    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = tid * 4u + i; let p = j ^ 8u;
        let x = buf_b[j]; let y = buf_b[p];
        if ((j & 8u) == 0u) { buf_a[j] = x + y; } else { buf_a[j] = y - x; }
    }
    workgroupBarrier();
    // h = 16 (a -> b)
    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = tid * 4u + i; let p = j ^ 16u;
        let x = buf_a[j]; let y = buf_a[p];
        if ((j & 16u) == 0u) { buf_b[j] = x + y; } else { buf_b[j] = y - x; }
    }
    workgroupBarrier();
    // h = 32 (b -> a)
    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = tid * 4u + i; let p = j ^ 32u;
        let x = buf_b[j]; let y = buf_b[p];
        if ((j & 32u) == 0u) { buf_a[j] = x + y; } else { buf_a[j] = y - x; }
    }
    workgroupBarrier();
    // h = 64 (a -> b)
    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = tid * 4u + i; let p = j ^ 64u;
        let x = buf_a[j]; let y = buf_a[p];
        if ((j & 64u) == 0u) { buf_b[j] = x + y; } else { buf_b[j] = y - x; }
    }
    workgroupBarrier();
    // h = 128 (b -> a)
    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = tid * 4u + i; let p = j ^ 128u;
        let x = buf_b[j]; let y = buf_b[p];
        if ((j & 128u) == 0u) { buf_a[j] = x + y; } else { buf_a[j] = y - x; }
    }
    workgroupBarrier();
    // h = 256 (a -> b)
    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = tid * 4u + i; let p = j ^ 256u;
        let x = buf_a[j]; let y = buf_a[p];
        if ((j & 256u) == 0u) { buf_b[j] = x + y; } else { buf_b[j] = y - x; }
    }
    workgroupBarrier();

    for (var i = 0u; i < 4u; i = i + 1u) {
        let j = tid * 4u + i;
        out_had[bb + j] = buf_b[j] * params.scale;
    }
}
