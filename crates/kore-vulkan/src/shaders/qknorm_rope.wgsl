struct QkNormRopeUniforms {
    n_heads: u32,    // q heads (8)
    n_kv: u32,       // kv heads (2)
    head_dim: u32,   // 256
    rope_dim: u32,   // 64
    pos: u32,        // position of row 0
    theta: f32,
    eps: f32,
    rows: u32,       // batch rows; row r sits at position pos + r
}

// qg: [rows, n_heads, 2, head_dim] rows = per-head [q | gate] interleaved
// kk: [rows, n_kv, head_dim]
// workgroup x 0..n_heads-1 -> q head; n_heads..n_heads+n_kv-1 -> k head
// workgroup y             -> batch row
//
// `rows` exists for multi-column decode (a verify pass or a chunked prefill):
// the m queries in such a batch each carry their own RoPE position, `pos + row`.
// At rows = 1 every offset below is `row * stride == 0` and the arithmetic is
// the same expression it always was, so single-token decode is unchanged.
@group(0) @binding(0) var<uniform> params: QkNormRopeUniforms;
@group(0) @binding(1) var<storage, read> qg: array<f32>;
@group(0) @binding(2) var<storage, read> kk: array<f32>;
@group(0) @binding(3) var<storage, read> q_norm: array<f32>;
@group(0) @binding(4) var<storage, read> k_norm: array<f32>;
@group(0) @binding(5) var<storage, read_write> q_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> k_out: array<f32>;
@group(0) @binding(7) var<storage, read_write> g_out: array<f32>;

var<workgroup> buf: array<f32, 256>;
var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wg_id.x;
    let row = wg_id.y;
    let nh = params.n_heads;
    if (h >= nh + params.n_kv || row >= params.rows) {
        return;
    }
    let hd = params.head_dim;
    let tid = lid.x;
    let is_q = h < nh;

    // Absolute bases for this (row, head). `kh` is 0 for q heads so that `kb`
    // stays in range while never being read.
    let qb = row * (nh * hd) + h * hd;
    let kh = select(h - nh, 0u, is_q);
    let kb = row * (params.n_kv * hd) + kh * hd;

    // load source (2 elems/thread) and gate passthrough
    if (is_q) {
        let base = row * (2u * nh * hd) + h * 2u * hd;
        buf[tid] = qg[base + tid];
        buf[128u + tid] = qg[base + 128u + tid];
        if (tid < 128u) {
            g_out[qb + tid] = qg[base + hd + tid];
            g_out[qb + 128u + tid] = qg[base + hd + 128u + tid];
        }
    } else {
        let base = row * (params.n_kv * hd) + (h - nh) * hd;
        buf[tid] = kk[base + tid];
        buf[128u + tid] = kk[base + 128u + tid];
    }
    workgroupBarrier();

    // rms reduce over head_dim
    var ss = buf[tid] * buf[tid] + buf[128u + tid] * buf[128u + tid];
    red[tid] = ss;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (tid < s) { red[tid] += red[tid + s]; }
        workgroupBarrier();
    }
    let rms = 1.0 / sqrt(red[0] / f32(hd) + params.eps);

    // normalize (per-head weight) + rope on first rope_dim elements.
    // NEOX pairing: (j, j + rope_dim/2), freq theta^(-2j/rope_dim).
    let rd = params.rope_dim;
    let half = rd / 2u;
    var v0 = buf[tid] * rms;
    var v1 = buf[128u + tid] * rms;
    if (is_q) {
        v0 = v0 * q_norm[tid];
        v1 = v1 * q_norm[128u + tid];
    } else {
        v0 = v0 * k_norm[tid];
        v1 = v1 * k_norm[128u + tid];
    }

    if (tid < half) {
        // element j=tid in first half of rope dim; partner j+half in second
        let freq = pow(params.theta, -2.0 * f32(tid) / f32(rd));
        let ang = f32(params.pos + row) * freq;
        let c = cos(ang);
        let s = sin(ang);
        let a = v0;
        let b = buf[tid + half] * rms * select(k_norm[tid + half], q_norm[tid + half], is_q);
        v0 = a * c - b * s;
        // write partner separately below
        q_or_k_out(is_q, qb, kb, tid, v0);
        let vp = b * c + a * s;
        q_or_k_out(is_q, qb, kb, tid + half, vp);
    }
    // non-roped elements (tid >= half) and elements >= rd handled below
    if (tid >= rd && tid < hd / 2u) {
        q_or_k_out(is_q, qb, kb, tid, v0);
    }
    // second half elements beyond rope dim
    let j1 = 128u + tid;
    if (j1 >= rd && j1 < hd) {
        q_or_k_out(is_q, qb, kb, j1, v1);
    }
    // rope partner elements in [half, rd) are written by the tid<half branch.
}

fn q_or_k_out(is_q: bool, qb: u32, kb: u32, j: u32, v: f32) {
    if (is_q) {
        q_out[qb + j] = v;
    } else {
        k_out[kb + j] = v;
    }
}
