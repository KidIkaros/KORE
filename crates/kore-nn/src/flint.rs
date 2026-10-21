//! Flint (40 B-A2.5 B MoE) -- GGUF loader and host reference path.
//!
//! Flint is a hybrid-attention MoE. 49 layers, the first two dense and the rest
//! routed; 32 query heads over 4 KV heads at `head_dim` 128; a 513-token
//! sliding window on 35 layers and full causal attention on 14 -- and those 14
//! are the ones carrying *no* positional encoding. Every attention layer also
//! has a learned per-head sink: an extra logit appended to the softmax whose
//! value is zero, i.e. `p_i = exp(s_i) / (sum_j exp(s_j) + exp(sink))`, which
//! lets a head attend to nothing at all. Routing is sigmoid-scored, top-6 of
//! 256, renormalised over the selected six, plus one dense shared expert.
//!
//! Everything here is the host path: one f32 token at a time over the mmap'd
//! weights. It is the oracle the device kernels will be gated against, and it
//! is also the path that answers "what does this model actually say".
//!
//! Two details are worth spelling out because they are easy to get wrong:
//!
//! * Q and K are RMS-normalised **per head** (`attn_q_norm` / `attn_k_norm` are
//!   128-wide, shared by every head) *before* RoPE, not after the projection as
//!   a whole.
//! * The sink enters the denominator only. Treating it as an extra value with a
//!   zero vector is the same thing, but only if the value matrix row it implies
//!   really is zero -- and the checkpoint has no such row.

use std::path::Path;
use std::sync::Arc;

use rayon::prelude::*;

use kore_core::{KoreError, Result};

use crate::gguf_min::{MetaVal, MiniGguf};
use crate::qwen35::{QKind, QW};
use oxillama_gguf::{ByteOwner, SharedBytes};

fn err(msg: impl Into<String>) -> KoreError {
    KoreError::StorageError(msg.into())
}

// ---------- config ----------

#[derive(Debug, Clone)]
pub struct FlintConfig {
    pub n_layers: usize,
    pub hidden: usize,
    /// Dense-layer feed-forward width (`feed_forward_length`, 6144).
    pub ffn: usize,
    /// Routed-expert width (`expert_feed_forward_length`, 512).
    pub moe_ffn: usize,
    /// Shared-expert width (`expert_shared_feed_forward_length`, 1024).
    pub shexp_ffn: usize,
    pub n_experts: usize,
    pub n_experts_used: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub sliding_window: usize,
    /// Layers `[0, n_dense)` run a plain FFN; the rest route.
    pub n_dense: usize,
    /// Per layer: does it get RoPE? Empty means "all of them".
    pub rope: Vec<bool>,
    /// Per layer: 513-token window (true) or full causal (false)?
    pub sliding: Vec<bool>,
    pub theta: f32,
    pub eps: f32,
    pub vocab: usize,
    pub eos: Vec<u32>,
}

impl FlintConfig {
    fn is_rope(&self, layer: usize) -> bool {
        self.rope.get(layer).copied().unwrap_or(true)
    }
    fn is_sliding(&self, layer: usize) -> bool {
        self.sliding.get(layer).copied().unwrap_or(false)
    }
    /// KV slots a layer has to keep: the window for a sliding layer, the whole
    /// context for a full-attention one.
    fn kv_cap(&self, layer: usize, max_ctx: usize) -> usize {
        if self.is_sliding(layer) {
            self.sliding_window.min(max_ctx)
        } else {
            max_ctx
        }
    }
}

// ---------- weights ----------

/// `oxillama`'s zero-copy view over one tensor of a [`MiniGguf`] map. Same
/// contract as the qwen35 twin: read-only, and the mapping outlives the view.
struct FlintBytes {
    gg: Arc<MiniGguf>,
    name: String,
}

// SAFETY: the bytes are a read-only slice of the file's `Mmap`, owned by `gg`
// and stable for as long as `FlintBytes` lives. Nothing writes through it.
unsafe impl ByteOwner for FlintBytes {
    fn owned_bytes(&self) -> &[u8] {
        self.gg.tensor_bytes(&self.name).unwrap_or(&[])
    }
}

/// Load one tensor as a packed matrix, sized from its GGUF dims.
///
/// A stacked `[.., .., n_experts]` tensor spans every expert slice here; the
/// individual experts are windows into it (`QW::expert`).
fn load_qw(gg: &Arc<MiniGguf>, name: &str) -> Result<QW> {
    let t = gg.tensor(name)?;
    let kind = QKind::from_type_id(t.type_id).ok_or_else(|| {
        err(format!(
            "flint: tensor '{name}' has GGML type {}, which this loader has no \
             kernel for (supported: F32=0, F16=1, Q8_0=8, Q2_K=10, Q3_K=11, \
             Q4_K=12, Q5_K=13, Q6_K=14)",
            t.type_id
        ))
    })?;
    let (m, n) = match t.dims.as_slice() {
        [m] => (*m, 1),
        [m, n, ..] => (*m, *n),
        _ => return Err(err(format!("flint: tensor '{name}' has no dimensions"))),
    };
    if m == 0 || n == 0 || m % kind.block() != 0 {
        return Err(err(format!(
            "flint: tensor '{name}' has row length {m}, not a multiple of {}",
            kind.block()
        )));
    }
    let row = (m / kind.block()) * kind.block_bytes();
    let want = n * row;
    let slices = t.dims.get(2).copied().unwrap_or(1).max(1);
    let total = want
        .checked_mul(slices)
        .ok_or_else(|| err(format!("flint: tensor '{name}' size overflows")))?;
    if t.nbytes < total {
        return Err(err(format!(
            "flint: tensor '{name}' carries {} bytes, {total} required",
            t.nbytes
        )));
    }
    let owner: Arc<dyn ByteOwner> = Arc::new(FlintBytes {
        gg: gg.clone(),
        name: name.to_string(),
    });
    let shared = SharedBytes::from_owner(owner.clone(), 0, total)
        .ok_or_else(|| err(format!("flint: tensor '{name}': {total} bytes out of range")))?;
    Ok(QW {
        name: name.to_string(),
        m,
        n,
        kind,
        off: 0,
        shared,
        owner,
    })
}

struct AttnW {
    norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    sinks: Vec<f32>,
    wq: QW,
    wk: QW,
    wv: QW,
    wo: QW,
    rope: bool,
    sliding: bool,
}

struct DenseW {
    norm: Vec<f32>,
    gate: QW,
    up: QW,
    down: QW,
}

struct MoeW {
    norm: Vec<f32>,
    router: QW,
    gate_exps: QW,
    up_exps: QW,
    down_exps: QW,
    sh_gate: QW,
    sh_up: QW,
    sh_down: QW,
    /// Byte stride between adjacent experts in a stacked tensor.
    gate_stride: usize,
    down_stride: usize,
}

enum Mlp {
    Dense(DenseW),
    Moe(MoeW),
}

struct Layer {
    attn: AttnW,
    mlp: Mlp,
}

// ---------- KV cache ----------

/// One layer's keys and values, as a ring when the layer is windowed.
///
/// A sliding layer's history is bounded by construction, so `cap` is the
/// window and slot `p % cap` holds position `p`; a full-attention layer sets
/// `cap` to the context and the same modular index degenerates to `p`.
pub struct FlintKv {
    k: Vec<f32>,
    v: Vec<f32>,
    kvd: usize,
    cap: usize,
    len: usize,
}

impl FlintKv {
    fn new(cap: usize, kv_dim: usize) -> Self {
        Self {
            k: vec![0.0; cap * kv_dim],
            v: vec![0.0; cap * kv_dim],
            kvd: kv_dim,
            cap,
            len: 0,
        }
    }
    fn push(&mut self, k: &[f32], v: &[f32], pos: usize) {
        debug_assert_eq!(k.len(), self.kvd);
        let slot = pos % self.cap;
        self.k[slot * self.kvd..(slot + 1) * self.kvd].copy_from_slice(k);
        self.v[slot * self.kvd..(slot + 1) * self.kvd].copy_from_slice(v);
        self.len = self.len.max(pos + 1);
    }
}

/// Per-sequence state: one KV ring per layer.
pub struct FlintState {
    kv: Vec<FlintKv>,
}

impl FlintState {
    pub fn len(&self) -> usize {
        self.kv.first().map(|k| k.len).unwrap_or(0)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------- math ----------

fn rms_into(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let mut ss = 0.0f32;
    for v in x {
        ss += v * v;
    }
    let inv = 1.0 / (ss / x.len() as f32 + eps).sqrt();
    for ((o, v), wi) in out.iter_mut().zip(x).zip(w) {
        *o = v * inv * wi;
    }
}

fn rms_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let mut ss = 0.0f32;
    for v in x.iter() {
        ss += v * v;
    }
    let inv = 1.0 / (ss / x.len() as f32 + eps).sqrt();
    for (v, wi) in x.iter_mut().zip(w) {
        *v = *v * inv * wi;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Llama-convention RoPE (`traditional=False` in MLX): inside a head, element
/// `i` pairs with `i + head_dim/2`, and each pair rotates by `pos * theta^(-2i/d)`.
///
/// `KORE_FLINT_ROPE=pairs` switches to the interleaved (GPT-J) pairing, which
/// is the same rotation over consecutive elements. It exists only to settle
/// which layout this GGUF was written in; the split form is what the reference
/// implementation uses and what the file should carry.
fn rope_inplace(x: &mut [f32], n_heads: usize, hd: usize, pos: usize, theta: f32) {
    let half = hd / 2;
    let pairs = std::env::var("KORE_FLINT_ROPE").ok().as_deref() == Some("pairs");
    let mut cos = vec![0.0f32; half];
    let mut sin = vec![0.0f32; half];
    for i in 0..half {
        let freq = theta.powf(-(2.0 * i as f32) / hd as f32);
        let a = pos as f32 * freq;
        cos[i] = a.cos();
        sin[i] = a.sin();
    }
    for h in 0..n_heads {
        let base = h * hd;
        for i in 0..half {
            let (ia, ib) = if pairs {
                (base + 2 * i, base + 2 * i + 1)
            } else {
                (base + i, base + i + half)
            };
            let a = x[ia];
            let b = x[ib];
            x[ia] = a * cos[i] - b * sin[i];
            x[ib] = b * cos[i] + a * sin[i];
        }
    }
}

/// `out = W x`, cut across threads by row block when the matrix is wide enough
/// to pay for the fork/join. Rows are independent and each row keeps one
/// sequential accumulator, so the result does not depend on the block count.
fn gemv_split(w: &QW, x: &[f32], out: &mut [f32]) -> Result<()> {
    const PAR_MIN: usize = 1 << 21;
    let n = w.n;
    if n * w.m < PAR_MIN || out.len() != n {
        return w.gemv(x, out);
    }
    let blocks = rayon::current_num_threads().clamp(1, 16).min(n.div_ceil(512));
    let per = n.div_ceil(blocks);
    out.par_chunks_mut(per).enumerate().try_for_each(|(b, o)| {
        let rows = o.len();
        w.rows(b * per, rows).gemv(x, o)
    })
}

// ---------- model ----------

pub struct FlintModel {
    pub cfg: FlintConfig,
    embd: QW,
    out_norm: Vec<f32>,
    head: QW,
    layers: Vec<Layer>,
}

impl FlintModel {
    pub fn from_gguf(path: &Path) -> Result<Self> {
        let gg = MiniGguf::load(path)?;
        let meta = &gg.meta;

        let arch = gg.get_str("general.architecture").unwrap_or("");
        if arch != "flint" {
            return Err(err(format!(
                "flint: file declares architecture '{arch}', not 'flint'"
            )));
        }
        let ku = |k: &str| -> Result<usize> {
            meta.get(&format!("{arch}.{k}"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| err(format!("flint: missing meta '{arch}.{k}'")))
        };
        let kopt = |k: &str| -> Option<usize> { ku(k).ok() };
        let kf = |k: &str, d: f64| -> f64 {
            meta.get(&format!("{arch}.{k}"))
                .and_then(|v| v.as_f64())
                .unwrap_or(d)
        };
        let flags = |k: &str| -> Vec<bool> {
            match meta.get(&format!("{arch}.{k}")) {
                Some(MetaVal::I64Arr(v)) => v.iter().map(|&x| x != 0).collect(),
                Some(MetaVal::F32Arr(v)) => v.iter().map(|&x| x != 0.0).collect(),
                _ => Vec::new(),
            }
        };

        let n_layers = ku("block_count")?;
        let mut rope = flags("attention.rope_pattern");
        if rope.len() != n_layers {
            rope = Vec::new();
        }
        let mut sliding = flags("attention.sliding_window_pattern");
        if sliding.len() != n_layers {
            sliding = Vec::new();
        }
        // The two patterns are the same partition in this family (full
        // attention is the NoPE one), so either can stand in for the other if a
        // writer dropped one of them.
        if rope.is_empty() && !sliding.is_empty() {
            rope = sliding.clone();
        }
        if sliding.is_empty() && !rope.is_empty() {
            sliding = rope.clone();
        }

        let eos = match meta.get("tokenizer.ggml.eos_token_id") {
            Some(MetaVal::U64(v)) => vec![*v as u32],
            Some(MetaVal::I64Arr(v)) => v.iter().map(|&x| x as u32).collect(),
            _ => Vec::new(),
        };

        // The vocabulary size comes from the untied head: `tokenizer.ggml.*`
        // arrays are deliberately not stored by the reader (150k strings), so
        // the head is the only place the full width is spelled out.
        let head = load_qw(&gg, "output.weight")?;

        let cfg = FlintConfig {
            n_layers,
            hidden: ku("embedding_length")?,
            ffn: ku("feed_forward_length")?,
            moe_ffn: ku("expert_feed_forward_length")?,
            shexp_ffn: kopt("expert_shared_feed_forward_length").unwrap_or(0),
            n_experts: ku("expert_count")?,
            n_experts_used: ku("expert_used_count")?,
            n_heads: ku("attention.head_count")?,
            n_kv: ku("attention.head_count_kv")?,
            head_dim: ku("attention.key_length")?,
            sliding_window: ku("attention.sliding_window")?,
            n_dense: kopt("leading_dense_block_count").unwrap_or(0),
            rope,
            sliding,
            theta: kf("rope.freq_base", 1e4) as f32,
            eps: kf("attention.layer_norm_rms_epsilon", 1e-6) as f32,
            vocab: head.n,
            eos,
        };

        let f32w = |name: &str| -> Result<Vec<f32>> { gg.tensor_f32(name) };

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = |s: &str| format!("blk.{i}.{s}");
            let attn = AttnW {
                norm: f32w(&p("attn_norm.weight"))?,
                q_norm: f32w(&p("attn_q_norm.weight"))?,
                k_norm: f32w(&p("attn_k_norm.weight"))?,
                sinks: f32w(&p("attn_sinks.weight"))?,
                wq: load_qw(&gg, &p("attn_q.weight"))?,
                wk: load_qw(&gg, &p("attn_k.weight"))?,
                wv: load_qw(&gg, &p("attn_v.weight"))?,
                wo: load_qw(&gg, &p("attn_output.weight"))?,
                rope: cfg.is_rope(i),
                sliding: cfg.is_sliding(i),
            };
            // Which FFN a layer carries is a property of the file, not of its
            // index: `leading_dense_block_count` says where the dense run ends,
            // but a router tensor is proof.
            let routed = gg.tensor(&p("ffn_gate_inp.weight")).is_ok();
            let mlp = if routed {
                let gate_exps = load_qw(&gg, &p("ffn_gate_exps.weight"))?;
                let up_exps = load_qw(&gg, &p("ffn_up_exps.weight"))?;
                let down_exps = load_qw(&gg, &p("ffn_down_exps.weight"))?;
                let gate_stride = gate_exps.n * gate_exps.row_bytes();
                let down_stride = down_exps.n * down_exps.row_bytes();
                // The view spans every expert slice, so the bank is `n_experts`
                // strides wide; `byte_len` is only one slice of it.
                let bank = gate_exps.shared.as_slice().len() - gate_exps.off;
                if gate_stride * cfg.n_experts > bank {
                    return Err(err(format!(
                        "flint: layer {i} expert bank holds {bank} bytes but {} experts of {} \
                         rows need {}",
                        cfg.n_experts,
                        gate_exps.n,
                        gate_stride * cfg.n_experts
                    )));
                }
                Mlp::Moe(MoeW {
                    norm: f32w(&p("ffn_norm.weight"))?,
                    router: load_qw(&gg, &p("ffn_gate_inp.weight"))?,
                    gate_exps,
                    up_exps,
                    down_exps,
                    sh_gate: load_qw(&gg, &p("ffn_gate_shexp.weight"))?,
                    sh_up: load_qw(&gg, &p("ffn_up_shexp.weight"))?,
                    sh_down: load_qw(&gg, &p("ffn_down_shexp.weight"))?,
                    gate_stride,
                    down_stride,
                })
            } else {
                Mlp::Dense(DenseW {
                    norm: f32w(&p("ffn_norm.weight"))?,
                    gate: load_qw(&gg, &p("ffn_gate.weight"))?,
                    up: load_qw(&gg, &p("ffn_up.weight"))?,
                    down: load_qw(&gg, &p("ffn_down.weight"))?,
                })
            };
            layers.push(Layer { attn, mlp });
        }

        Ok(Self {
            embd: load_qw(&gg, "token_embd.weight")?,
            out_norm: f32w("output_norm.weight")?,
            head,
            layers,
            cfg,
        })
    }

    pub fn new_state(&self, max_ctx: usize) -> FlintState {
        let kvd = self.cfg.n_kv * self.cfg.head_dim;
        FlintState {
            kv: (0..self.cfg.n_layers)
                .map(|i| FlintKv::new(self.cfg.kv_cap(i, max_ctx), kvd))
                .collect(),
        }
    }

    fn attn_step(
        &self,
        l: &AttnW,
        x: &mut [f32],
        pos: usize,
        kv: &mut FlintKv,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let hd = cfg.head_dim;
        let kn = cfg.n_kv * hd;
        let qn = cfg.n_heads * hd;

        let mut xn = vec![0.0f32; cfg.hidden];
        rms_into(x, &l.norm, cfg.eps, &mut xn);

        let mut q = vec![0.0f32; qn];
        let mut k = vec![0.0f32; kn];
        let mut v = vec![0.0f32; kn];
        gemv_split(&l.wq, &xn, &mut q)?;
        l.wk.gemv(&xn, &mut k)?;
        l.wv.gemv(&xn, &mut v)?;

        for h in 0..cfg.n_heads {
            rms_inplace(&mut q[h * hd..(h + 1) * hd], &l.q_norm, cfg.eps);
        }
        for h in 0..cfg.n_kv {
            rms_inplace(&mut k[h * hd..(h + 1) * hd], &l.k_norm, cfg.eps);
        }
        if l.rope {
            rope_inplace(&mut q, cfg.n_heads, hd, pos, cfg.theta);
            rope_inplace(&mut k, cfg.n_kv, hd, pos, cfg.theta);
        }

        kv.push(&k, &v, pos);
        let cap = kv.cap;
        let seen = pos + 1;
        let win = if l.sliding {
            cfg.sliding_window.min(seen)
        } else {
            seen
        };
        let start = seen - win;
        let slots: Vec<usize> = (0..win).map(|j| (start + j) % cap).collect();
        let scale = 1.0 / (hd as f32).sqrt();
        let reps = cfg.n_heads / cfg.n_kv;

        let mut out = vec![0.0f32; qn];
        out.par_chunks_mut(hd).enumerate().for_each(|(h, o)| {
            let kvh = h / reps;
            let qh = &q[h * hd..(h + 1) * hd];
            let mut p = vec![0.0f32; win];
            for (j, &slot) in slots.iter().enumerate() {
                let kb = &kv.k[slot * kn + kvh * hd..][..hd];
                let mut acc = 0.0f32;
                for d in 0..hd {
                    acc += qh[d] * kb[d];
                }
                p[j] = acc * scale;
            }
            // Softmax over the scores plus the learned sink, which occupies one
            // extra slot in the denominator and contributes nothing to the
            // output. `m` starts at the sink so the sink's own exponential is
            // exactly 1 rather than an overflow hazard.
            let sink = l.sinks[h];
            let mut m = sink;
            for &s in &p {
                if s > m {
                    m = s;
                }
            }
            let mut den = (sink - m).exp();
            for s in p.iter_mut() {
                *s = (*s - m).exp();
                den += *s;
            }
            let inv = 1.0 / den;
            for (d, od) in o.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for (j, &pj) in p.iter().enumerate() {
                    acc += pj * kv.v[slots[j] * kn + kvh * hd + d];
                }
                *od = acc * inv;
            }
        });

        let mut proj = vec![0.0f32; cfg.hidden];
        gemv_split(&l.wo, &out, &mut proj)?;
        for (xv, pv) in x.iter_mut().zip(&proj) {
            *xv += pv;
        }
        Ok(())
    }

    fn dense_ffn(&self, l: &DenseW, x: &mut [f32]) -> Result<()> {
        let cfg = &self.cfg;
        let mut xn = vec![0.0f32; cfg.hidden];
        rms_into(x, &l.norm, cfg.eps, &mut xn);
        let wide = l.gate.n;
        let mut g = vec![0.0f32; wide];
        let mut u = vec![0.0f32; wide];
        gemv_split(&l.gate, &xn, &mut g)?;
        gemv_split(&l.up, &xn, &mut u)?;
        for (gi, ui) in g.iter_mut().zip(&u) {
            *gi = silu(*gi) * *ui;
        }
        let mut d = vec![0.0f32; cfg.hidden];
        gemv_split(&l.down, &g, &mut d)?;
        for (xv, dv) in x.iter_mut().zip(&d) {
            *xv += dv;
        }
        Ok(())
    }

    fn moe_ffn(&self, l: &MoeW, x: &mut [f32], trace: Option<&mut Vec<usize>>) -> Result<()> {
        let cfg = &self.cfg;
        let mut xn = vec![0.0f32; cfg.hidden];
        rms_into(x, &l.norm, cfg.eps, &mut xn);

        // Router: raw sigmoid scores, top-k by score, renormalised over the
        // chosen set. (`expert_bias` exists in the training code but the
        // checkpoint ships it as zeros and the GGUF does not carry it at all.)
        let mut logits = vec![0.0f32; cfg.n_experts];
        l.router.gemv(&xn, &mut logits)?;
        let scores: Vec<f32> = logits.iter().map(|&z| sigmoid(z)).collect();
        let k = cfg.n_experts_used;
        let mut picked: Vec<usize> = Vec::with_capacity(k);
        for _ in 0..k {
            let mut best = usize::MAX;
            let mut best_v = f32::NEG_INFINITY;
            for (e, &s) in scores.iter().enumerate() {
                if s > best_v && !picked.contains(&e) {
                    best_v = s;
                    best = e;
                }
            }
            picked.push(best);
        }
        let mut weights: Vec<f32> = picked.iter().map(|&e| scores[e]).collect();
        let sum: f32 = weights.iter().sum();
        let inv = 1.0 / (sum + 1e-20);
        for w in weights.iter_mut() {
            *w *= inv;
        }

        // Shared expert runs on every token, unconditionally, and is added
        // rather than gated.
        let sh = cfg.shexp_ffn;
        let mut sg = vec![0.0f32; sh];
        let mut su = vec![0.0f32; sh];
        l.sh_gate.gemv(&xn, &mut sg)?;
        l.sh_up.gemv(&xn, &mut su)?;
        for (gi, ui) in sg.iter_mut().zip(&su) {
            *gi = silu(*gi) * *ui;
        }
        let mut acc = vec![0.0f32; cfg.hidden];
        gemv_split(&l.sh_down, &sg, &mut acc)?;

        // The six experts are independent, so they run in parallel; the results
        // are then summed in expert order, which keeps the answer independent
        // of the thread count.
        let parts: Vec<Result<Vec<f32>>> = (0..k)
            .into_par_iter()
            .map(|j| {
                let e = picked[j];
                let mut g = vec![0.0f32; cfg.moe_ffn];
                let mut u = vec![0.0f32; cfg.moe_ffn];
                l.gate_exps.expert(e, l.gate_stride).gemv(&xn, &mut g)?;
                l.up_exps.expert(e, l.gate_stride).gemv(&xn, &mut u)?;
                for (gi, ui) in g.iter_mut().zip(&u) {
                    *gi = silu(*gi) * *ui;
                }
                let mut d = vec![0.0f32; cfg.hidden];
                l.down_exps.expert(e, l.down_stride).gemv(&g, &mut d)?;
                Ok(d)
            })
            .collect();
        for (j, part) in parts.into_iter().enumerate() {
            let d = part?;
            let w = weights[j];
            for (a, dv) in acc.iter_mut().zip(&d) {
                *a += w * dv;
            }
        }
        if let Some(t) = trace {
            t.extend_from_slice(&picked);
        }
        for (xv, av) in x.iter_mut().zip(&acc) {
            *xv += av;
        }
        Ok(())
    }

    /// Logits for one hidden state.
    pub fn logits(&self, x: &[f32]) -> Result<Vec<f32>> {
        let mut h = vec![0.0f32; self.cfg.hidden];
        rms_into(x, &self.out_norm, self.cfg.eps, &mut h);
        let mut logits = vec![0.0f32; self.head.n];
        gemv_split(&self.head, &h, &mut logits)?;
        Ok(logits)
    }

    /// One token at `pos`; returns the logits for the next one.
    pub fn forward(&self, token: u32, pos: usize, st: &mut FlintState) -> Result<Vec<f32>> {
        self.forward_traced(token, pos, st, None)
    }

    fn forward_traced(
        &self,
        token: u32,
        pos: usize,
        st: &mut FlintState,
        mut trace: Option<&mut Vec<usize>>,
    ) -> Result<Vec<f32>> {
        let mut x = self.embd.row_f32(token as usize)?;
        for (i, layer) in self.layers.iter().enumerate() {
            self.attn_step(&layer.attn, &mut x, pos, &mut st.kv[i])?;
            match &layer.mlp {
                Mlp::Dense(l) => self.dense_ffn(l, &mut x)?,
                Mlp::Moe(l) => self.moe_ffn(l, &mut x, trace.as_deref_mut())?,
            }
        }
        self.logits(&x)
    }

    pub fn argmax(logits: &[f32]) -> u32 {
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > bv {
                bv = v;
                best = i;
            }
        }
        best as u32
    }

    /// Greedy decode of `n_new` tokens after `tokens`; returns the new ids.
    pub fn generate_greedy(
        &self,
        tokens: &[u32],
        n_new: usize,
        max_ctx: usize,
    ) -> Result<Vec<u32>> {
        let mut st = self.new_state(max_ctx);
        let mut logits = Vec::new();
        for (i, &t) in tokens.iter().enumerate() {
            logits = self.forward(t, i, &mut st)?;
        }
        let mut out = Vec::with_capacity(n_new);
        for k in 0..n_new {
            let tok = Self::argmax(&logits);
            if self.cfg.eos.contains(&tok) {
                break;
            }
            out.push(tok);
            logits = self.forward(tok, tokens.len() + k, &mut st)?;
        }
        Ok(out)
    }

    /// Expert ids chosen on the last forward that was traced (diagnostics).
    pub fn route_trace(&self, tokens: &[u32], max_ctx: usize) -> Result<Vec<usize>> {
        let mut st = self.new_state(max_ctx);
        let mut trace = Vec::new();
        for (i, &t) in tokens.iter().enumerate() {
            let _ = self.forward_traced(t, i, &mut st, Some(&mut trace))?;
        }
        Ok(trace)
    }
}
