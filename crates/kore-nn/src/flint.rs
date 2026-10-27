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

// ---------- phase profile (diagnostics only) ----------
//
// `KORE_PROFILE=1` accumulates wall clock per phase so the host path can be
// attributed before any kernel work starts. Off by default and free when off
// (one cached env lookup, no timers).

mod prof {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;
    use std::time::Instant;

    pub const ATTN: usize = 0;
    pub const DENSE: usize = 1;
    pub const MOE: usize = 2;
    /// Router + shared gate/up: one fused parallel region inside the MoE.
    pub const PRE: usize = 3;
    /// Shared expert down-projection.
    pub const SHARED: usize = 4;
    pub const EXPERTS: usize = 5;
    pub const HEAD: usize = 6;
    pub const NAMES: [&str; 7] = [
        "attn",
        "dense_ffn",
        "  moe_total",
        "  pre",
        "  sh_down",
        "  experts",
        "head",
    ];

    /// Histogram of `|predicted ∩ actual|` for next-layer routing, and the
    /// number of scored layers. `KORE_PREDICT=1` only.
    static PRED_HIST: [AtomicU64; 8] = [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ];

    static PRED: OnceLock<bool> = OnceLock::new();

    pub fn predict_on() -> bool {
        *PRED.get_or_init(|| {
            std::env::var("KORE_PREDICT")
                .map(|v| v != "0")
                .unwrap_or(false)
        })
    }

    pub fn pred_score(pred: &[usize], actual: &[usize]) {
        let hit = pred.iter().filter(|e| actual.contains(e)).count();
        PRED_HIST[hit.min(7)].fetch_add(1, Ordering::Relaxed);
    }

    pub fn pred_dump() {
        if !predict_on() {
            return;
        }
        let n: u64 = PRED_HIST.iter().map(|c| c.load(Ordering::Relaxed)).sum();
        if n == 0 {
            return;
        }
        let mean: f64 = PRED_HIST
            .iter()
            .enumerate()
            .map(|(i, c)| i as f64 * c.load(Ordering::Relaxed) as f64)
            .sum::<f64>()
            / n as f64;
        let hist: Vec<String> = PRED_HIST
            .iter()
            .enumerate()
            .map(|(i, c)| {
                format!(
                    "{i}:{:.0}%",
                    100.0 * c.load(Ordering::Relaxed) as f64 / n as f64
                )
            })
            .collect();
        eprintln!(
            "next-layer route prediction (router of L+1 on L's input): mean {mean:.2} of 6 over {n} layers  [{}]",
            hist.join(" ")
        );
    }

    static ON: OnceLock<bool> = OnceLock::new();
    static NS: [AtomicU64; 7] = [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ];

    pub fn on() -> bool {
        *ON.get_or_init(|| {
            std::env::var("KORE_PROFILE")
                .map(|v| v != "0")
                .unwrap_or(false)
        })
    }

    pub struct T(pub Instant);

    pub fn t() -> Option<T> {
        if on() {
            Some(T(Instant::now()))
        } else {
            None
        }
    }

    pub fn acc(slot: usize, t: Option<T>) {
        if let Some(T(t0)) = t {
            NS[slot].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }

    pub fn dump(steps: usize) {
        if !on() || steps == 0 {
            return;
        }
        let get = |i: usize| NS[i].load(Ordering::Relaxed) as f64;
        // ROUTER/SHARED/EXPERTS are nested inside MOE, so they are reported
        // against the MoE total rather than against the step.
        let total: f64 = [ATTN, DENSE, MOE, HEAD].iter().map(|&i| get(i)).sum();
        let moe = get(MOE).max(1.0);
        eprintln!(
            "--- flint host profile: {steps} steps, {:.2} ms/step ---",
            total / 1e6 / steps as f64
        );
        let rows = [
            (ATTN, total),
            (DENSE, total),
            (MOE, total),
            (PRE, moe),
            (SHARED, moe),
            (EXPERTS, moe),
            (HEAD, total),
        ];
        for (i, denom) in rows {
            let ns = get(i);
            eprintln!(
                "  {:<12} {:8.2} ms/step  {:5.1}%",
                NAMES[i],
                ns / 1e6 / steps as f64,
                100.0 * ns / denom.max(1.0)
            );
        }
    }

    pub fn reset() {
        for a in NS.iter() {
            a.store(0, Ordering::Relaxed);
        }
    }
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
    let shared = SharedBytes::from_owner(owner.clone(), 0, total).ok_or_else(|| {
        err(format!(
            "flint: tensor '{name}': {total} bytes out of range"
        ))
    })?;
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

/// Top-`k` expert ids by score, highest first. Ties break towards the lower
/// index, which is what the reference's `argpartition`-then-sort does on the
/// scores this router produces.
fn topk_ids(logits: &[f32], k: usize) -> Vec<usize> {
    let scores: Vec<f32> = logits.iter().map(|&z| sigmoid(z)).collect();
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
    picked
}

/// `out = W x`.
///
/// **Rows are split inside the kernel, and that is the only lane that pays.**
/// `oxillama`'s `gemv` hands its row loop to a dedicated pool sized to the
/// machine, so cutting the matrix into outer chunks here nests two pools: the
/// outer workers block in `ThreadPool::install` while the inner pool serves
/// them. Measured on the i5-14400F, the nested form ran the attention
/// projections at 10.7 GB/s where one call reaches 22.9 GB/s. A hand-rolled
/// row lane with the kernel pinned to one thread was slower still (14.9 GB/s),
/// because rayon's own fork/join costs about as much as the kernel's.
///
/// What this leaves is the shape of the call, not the threading: 512-row
/// matvecs top out near 12 GB/s and 150k-row ones reach 35 GB/s, which is the
/// box's practical DRAM ceiling (`docs/inference-handoff.md` §14.4).
fn gemv_par(w: &QW, x: &[f32], out: &mut [f32]) -> Result<()> {
    w.gemv(x, out)
}

/// Same, for one stacked MoE tensor: `out` is `ids.len()` row-blocks,
/// expert-major. The outer split is exact (one chunk per expert), so nesting
/// the inner row split is safe.
fn gemv_experts(
    t: &QW,
    stride: usize,
    ids: &[usize],
    rows: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    out.par_chunks_mut(rows)
        .zip(ids.par_iter())
        .try_for_each(|(o, &e)| gemv_par(&t.expert(e, stride), x, o))
}

// ---------- route profile ----------

/// Per-(layer, expert) routing counts over a real workload.
///
/// 47 x 256 u32 is 48 KiB, so this is cheap enough to keep resident and to
/// persist (`--expert-profile-save` in the Maya engine is the same idea). What
/// it answers is whether an expert tier can pay at all: if the top 40 % of the
/// experts take 90 % of the reads, pinning them is worth it; if the distribution
/// is flat, it is not, and the only lever left is cheaper misses.
pub struct RouteProfile {
    pub n_layers: usize,
    pub n_experts: usize,
    /// `n_layers * n_experts`, row-major by layer.
    pub counts: Vec<u32>,
    /// Routed tokens seen (each contributes `n_moe * k` reads).
    pub tokens: u64,
    /// Histogram of `|previous token's set ∩ this token's set|`, one entry per
    /// layer per token. Length `k + 1`.
    pub overlap: Vec<u64>,
    /// Bytes one expert occupies (gate + up + down), for the budget table.
    pub expert_bytes: u64,
    /// Number of routed layers, for the summary.
    pub moe_layers: usize,
    /// Experts chosen per token.
    pub k: usize,
}

impl RouteProfile {
    pub fn new(n_layers: usize, n_experts: usize) -> Self {
        Self {
            n_layers,
            n_experts,
            counts: vec![0; n_layers * n_experts],
            tokens: 0,
            overlap: vec![0; 8],
            expert_bytes: 0,
            moe_layers: 0,
            k: 0,
        }
    }

    /// One token's trace: `moe_layers.len() * k` ids, layer-major.
    fn absorb(&mut self, trace: &[usize], moe_layers: &[usize], prev: &mut [Vec<usize>], k: usize) {
        if k == 0 || trace.len() != moe_layers.len() * k {
            return;
        }
        for (j, &layer) in moe_layers.iter().enumerate() {
            let ids = &trace[j * k..(j + 1) * k];
            for &e in ids {
                if e < self.n_experts {
                    self.counts[layer * self.n_experts + e] += 1;
                }
            }
            let p = &mut prev[layer];
            if p.len() == k {
                let hit = p.iter().filter(|e| ids.contains(e)).count();
                if hit < self.overlap.len() {
                    self.overlap[hit] += 1;
                }
            }
            *p = ids.to_vec();
        }
        self.tokens += 1;
    }

    /// Total route reads, and the counts sorted per layer.
    pub fn total(&self) -> u64 {
        self.counts.iter().map(|&c| c as u64).sum()
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut buf = Vec::with_capacity(24 + self.counts.len() * 4);
        buf.extend_from_slice(b"KRP2");
        buf.extend_from_slice(&(self.n_layers as u32).to_le_bytes());
        buf.extend_from_slice(&(self.n_experts as u32).to_le_bytes());
        buf.extend_from_slice(&(self.k as u32).to_le_bytes());
        buf.extend_from_slice(&(self.tokens as u64).to_le_bytes());
        for &c in &self.counts {
            buf.extend_from_slice(&c.to_le_bytes());
        }
        std::fs::write(path, buf).map_err(|e| err(format!("flint: profile save: {e}")))
    }

    pub fn load(path: &Path) -> Result<Self> {
        let buf = std::fs::read(path).map_err(|e| err(format!("flint: profile load: {e}")))?;
        if buf.len() < 24 || &buf[..4] != b"KRP2" {
            return Err(err("flint: profile: not a KRP2 file"));
        }
        let rd32 = |o: usize| u32::from_le_bytes(buf[o..o + 4].try_into().unwrap()) as usize;
        let rd64 = |o: usize| u64::from_le_bytes(buf[o..o + 8].try_into().unwrap());
        let (n_layers, n_experts) = (rd32(4), rd32(8));
        if buf.len() != 24 + n_layers * n_experts * 4 {
            return Err(err("flint: profile: truncated"));
        }
        let counts = buf[24..]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(Self {
            n_layers,
            n_experts,
            counts,
            k: rd32(12),
            tokens: rd64(16),
            overlap: vec![0; 8],
            expert_bytes: 0,
            moe_layers: 0,
        })
    }

    /// The `n` busiest (layer, expert) pairs, most used first.
    pub fn hot_set_n(&self, n: usize) -> Vec<u32> {
        let mut idx: Vec<u32> = (0..self.counts.len() as u32).collect();
        idx.sort_unstable_by_key(|&i| std::cmp::Reverse(self.counts[i as usize]));
        idx.truncate(n.min(idx.len()));
        idx
    }

    /// Share of *this* profile's reads that `hot` serves. Applying one profile's
    /// hot set to another profile is the only honest test of a pinned set: a
    /// fit on the same routes it is measured on always looks good.
    pub fn coverage(&self, hot: &[u32]) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        let hit: u64 = hot
            .iter()
            .filter(|&&i| (i as usize) < self.counts.len())
            .map(|&i| self.counts[i as usize] as u64)
            .sum();
        100.0 * hit as f64 / total as f64
    }

    /// The table that decides the hot-set size: for a pinned budget, the share
    /// of routed reads it serves.
    pub fn summary(&self) {
        let total = self.total().max(1);
        let mut all: Vec<u32> = self.counts.clone();
        all.sort_unstable_by(|a, b| b.cmp(a));
        let mut cum = vec![0u64; all.len() + 1];
        for (i, &c) in all.iter().enumerate() {
            cum[i + 1] = cum[i] + c as u64;
        }
        eprintln!(
            "routes: {} tokens x {} MoE layers x {} experts = {} reads over {} (layer, expert) pairs",
            self.tokens,
            self.moe_layers,
            self.k,
            total,
            self.counts.len()
        );
        let uniq = self.counts.iter().filter(|&&c| c > 0).count();
        eprintln!(
            "experts ever used: {uniq} of {} ({:.0}%), max count {} (mean {:.1})",
            self.counts.len(),
            100.0 * uniq as f64 / self.counts.len() as f64,
            all[0],
            total as f64 / self.counts.len() as f64
        );
        if self.expert_bytes > 0 {
            eprintln!("\npinned budget -> route reads served");
            for gib in [2u64, 4, 6, 8, 10, 12, 14, 16] {
                let n = (gib << 30) as usize / self.expert_bytes.max(1) as usize;
                if n == 0 || n >= all.len() {
                    continue;
                }
                eprintln!(
                    "  {gib:2} GiB  ({n:5} experts, {:4.1}% of the bank)  {:5.1}%",
                    100.0 * n as f64 / all.len() as f64,
                    100.0 * cum[n] as f64 / total as f64
                );
            }
        }
        // Temporal stability: how much of this token's set was already set last
        // token. High overlap is what makes a prefetch or a small cache work.
        let pairs: u64 = self.overlap.iter().sum();
        if pairs > 0 {
            let mean: f64 = self
                .overlap
                .iter()
                .enumerate()
                .map(|(i, &c)| i as f64 * c as f64)
                .sum::<f64>()
                / pairs as f64;
            let h: Vec<String> = self
                .overlap
                .iter()
                .enumerate()
                .filter(|(_, &c)| c > 0)
                .map(|(i, &c)| format!("{i}:{:.0}%", 100.0 * c as f64 / pairs as f64))
                .collect();
            eprintln!(
                "\nconsecutive-token set overlap: mean {mean:.2} of 6  [{}]",
                h.join(" ")
            );
        }
    }
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

    fn attn_step(&self, l: &AttnW, x: &mut [f32], pos: usize, kv: &mut FlintKv) -> Result<()> {
        let cfg = &self.cfg;
        let hd = cfg.head_dim;
        let kn = cfg.n_kv * hd;
        let qn = cfg.n_heads * hd;

        let mut xn = vec![0.0f32; cfg.hidden];
        rms_into(x, &l.norm, cfg.eps, &mut xn);

        let mut q = vec![0.0f32; qn];
        let mut k = vec![0.0f32; kn];
        let mut v = vec![0.0f32; kn];
        gemv_par(&l.wq, &xn, &mut q)?;
        gemv_par(&l.wk, &xn, &mut k)?;
        gemv_par(&l.wv, &xn, &mut v)?;

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
        gemv_par(&l.wo, &out, &mut proj)?;
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
        gemv_par(&l.gate, &xn, &mut g)?;
        gemv_par(&l.up, &xn, &mut u)?;
        for (gi, ui) in g.iter_mut().zip(&u) {
            *gi = silu(*gi) * *ui;
        }
        let mut d = vec![0.0f32; cfg.hidden];
        gemv_par(&l.down, &g, &mut d)?;
        for (xv, dv) in x.iter_mut().zip(&d) {
            *xv += dv;
        }
        Ok(())
    }

    fn moe_ffn(
        &self,
        l: &MoeW,
        next: Option<&MoeW>,
        x: &mut [f32],
        trace: Option<&mut Vec<usize>>,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let mut xn = vec![0.0f32; cfg.hidden];
        rms_into(x, &l.norm, cfg.eps, &mut xn);

        // The router and the shared expert's gate/up all read `xn` and do not
        // depend on each other, so they share one parallel region. The experts
        // cannot join it: they need the router's top-k.
        let sh = cfg.shexp_ffn;
        let mut logits = vec![0.0f32; cfg.n_experts];
        let mut sg = vec![0.0f32; sh];
        let mut su = vec![0.0f32; sh];
        let tp = prof::t();
        gemv_par(&l.router, &xn, &mut logits)?;
        gemv_par(&l.sh_gate, &xn, &mut sg)?;
        gemv_par(&l.sh_up, &xn, &mut su)?;

        // Router: raw sigmoid scores, top-k by score, renormalised over the
        // chosen set. (`expert_bias` exists in the training code but the
        // checkpoint ships it as zeros and the GGUF does not carry it at all.)
        let scores: Vec<f32> = logits.iter().map(|&z| sigmoid(z)).collect();
        let k = cfg.n_experts_used;
        let picked = topk_ids(&logits, k);
        // Diagnostics: would the *next* layer's router, run on this layer's
        // input, have predicted this? That is the prediction a prefetch of the
        // next layer has to live on (Maya's PREFETCH_N), and it costs one extra
        // router matvec per layer.
        if prof::predict_on() {
            if let Some(nx) = next {
                let mut pn = vec![0.0f32; cfg.hidden];
                rms_into(x, &nx.norm, cfg.eps, &mut pn);
                let mut pl = vec![0.0f32; cfg.n_experts];
                nx.router.gemv(&pn, &mut pl)?;
                prof::pred_score(&topk_ids(&pl, k), &picked);
            }
        }
        let mut weights: Vec<f32> = picked.iter().map(|&e| scores[e]).collect();
        let sum: f32 = weights.iter().sum();
        let inv = 1.0 / (sum + 1e-20);
        for w in weights.iter_mut() {
            *w *= inv;
        }
        prof::acc(prof::PRE, tp);

        // Shared expert runs on every token, unconditionally, and is added
        // rather than gated.
        let ts = prof::t();
        for (gi, ui) in sg.iter_mut().zip(&su) {
            *gi = silu(*gi) * *ui;
        }
        let mut acc = vec![0.0f32; cfg.hidden];
        gemv_par(&l.sh_down, &sg, &mut acc)?;
        prof::acc(prof::SHARED, ts);

        let te = prof::t();
        // Every chosen expert is one row-block of the flat output, and each
        // expert is itself cut across threads. Rows are independent, so the
        // sum below is still taken in expert order and the result does not
        // depend on the thread count.
        let m = cfg.moe_ffn;
        let mut g = vec![0.0f32; k * m];
        let mut u = vec![0.0f32; k * m];
        gemv_experts(&l.gate_exps, l.gate_stride, &picked, m, &xn, &mut g)?;
        gemv_experts(&l.up_exps, l.gate_stride, &picked, m, &xn, &mut u)?;
        for (gi, ui) in g.iter_mut().zip(&u) {
            *gi = silu(*gi) * *ui;
        }
        let mut d = vec![0.0f32; k * cfg.hidden];
        // The down projection takes each expert's own SwiGLU output, so the
        // input is cut by expert block too.
        for ((xg, o), &e) in g.chunks(m).zip(d.chunks_mut(cfg.hidden)).zip(&picked) {
            gemv_par(&l.down_exps.expert(e, l.down_stride), xg, o)?;
        }
        for (j, w) in weights.iter().enumerate() {
            let part = &d[j * cfg.hidden..(j + 1) * cfg.hidden];
            for (a, dv) in acc.iter_mut().zip(part) {
                *a += w * dv;
            }
        }
        prof::acc(prof::EXPERTS, te);
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
        gemv_par(&self.head, &h, &mut logits)?;
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
            let ta = prof::t();
            self.attn_step(&layer.attn, &mut x, pos, &mut st.kv[i])?;
            prof::acc(prof::ATTN, ta);
            match &layer.mlp {
                Mlp::Dense(l) => {
                    let td = prof::t();
                    self.dense_ffn(l, &mut x)?;
                    prof::acc(prof::DENSE, td);
                }
                Mlp::Moe(l) => {
                    let tm = prof::t();
                    let next = self.layers[i + 1..].iter().find_map(|n| match &n.mlp {
                        Mlp::Moe(m) => Some(m),
                        Mlp::Dense(_) => None,
                    });
                    self.moe_ffn(l, next, &mut x, trace.as_deref_mut())?;
                    prof::acc(prof::MOE, tm);
                }
            }
        }
        let th = prof::t();
        let out = self.logits(&x);
        prof::acc(prof::HEAD, th);
        out
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

    /// Route every token of `prompt` and `n_new` generated ones, accumulating a
    /// per-(layer, expert) count and the overlap between consecutive tokens.
    ///
    /// This is the measurement the expert tier stands or falls on: a hot set is
    /// only worth pinning if the routing is skewed (a few experts take most of
    /// the reads) and if it is *stable* across tokens (yesterday's hot expert is
    /// tomorrow's). Both are reported by [`RouteProfile::summary`].
    pub fn profile_routes(
        &self,
        prompt: &[u32],
        n_new: usize,
        max_ctx: usize,
    ) -> Result<RouteProfile> {
        let mut st = self.new_state(max_ctx);
        let cfg = &self.cfg;
        let moe_layers: Vec<usize> = self
            .layers
            .iter()
            .enumerate()
            .filter(|(_, l)| matches!(l.mlp, Mlp::Moe(_)))
            .map(|(i, _)| i)
            .collect();
        let mut prof = RouteProfile::new(cfg.n_layers, cfg.n_experts);
        if let Some(&li) = moe_layers.first() {
            if let Mlp::Moe(m) = &self.layers[li].mlp {
                // `byte_len` is one expert's payload (the stacked tensor's
                // `n` is a single expert's row count), so this is gate + up +
                // down for one expert, which is the granularity a hot set has.
                prof.expert_bytes =
                    (m.gate_exps.byte_len() + m.up_exps.byte_len() + m.down_exps.byte_len()) as u64;
            }
        }
        prof.moe_layers = moe_layers.len();
        prof.k = cfg.n_experts_used;
        let mut prev: Vec<Vec<usize>> = vec![Vec::new(); cfg.n_layers];
        let mut trace = Vec::new();
        let mut logits = Vec::new();
        let mut traces: Vec<Vec<usize>> = Vec::new();

        for (i, &t) in prompt.iter().enumerate() {
            trace.clear();
            logits = self.forward_traced(t, i, &mut st, Some(&mut trace))?;
            prof.absorb(&trace, &moe_layers, &mut prev, cfg.n_experts_used);
            traces.push(trace.clone());
        }
        for k in 0..n_new {
            let tok = Self::argmax(&logits);
            trace.clear();
            logits = self.forward_traced(tok, prompt.len() + k, &mut st, Some(&mut trace))?;
            prof.absorb(&trace, &moe_layers, &mut prev, cfg.n_experts_used);
            traces.push(trace.clone());
        }
        if std::env::var("KORE_ROUTE_SPLIT").ok().as_deref() == Some("1") && traces.len() >= 8 {
            // A hot set built from the first half of *this* document, scored on
            // the second half. This is the only version of pinning that could
            // still pay: no cross-prompt transfer, just the document's own
            // topic holding still.
            let half = traces.len() / 2;
            let build = |ts: &[Vec<usize>]| {
                let mut out = vec![0u32; cfg.n_layers * cfg.n_experts];
                for t in ts {
                    for (j, &layer) in moe_layers.iter().enumerate() {
                        for &e in &t[j * cfg.n_experts_used..(j + 1) * cfg.n_experts_used] {
                            out[layer * cfg.n_experts + e] += 1;
                        }
                    }
                }
                out
            };
            let (a, b) = traces.split_at(half);
            let train = build(a);
            let test = build(b);
            let tsum: u64 = test.iter().map(|&c| c as u64).sum();
            let mut idx: Vec<u32> = (0..train.len() as u32).collect();
            idx.sort_unstable_by_key(|&i| std::cmp::Reverse(train[i as usize]));
            eprintln!(
                "\nwithin-document hot set: first {} tokens -> next {} tokens",
                a.len(),
                b.len()
            );
            if test.iter().all(|&c| c == 0) || tsum == 0 {
                eprintln!("  (second half has no routes)");
            } else {
                let mut cum = 0u64;
                for (rank, &i) in idx.iter().enumerate() {
                    cum += test[i as usize] as u64;
                    let n = rank + 1;
                    let report = [64usize, 256, 1024, 2048, 4211, 6316, 8422]
                        .iter()
                        .any(|&r| r == n);
                    if report {
                        eprintln!(
                            "  top {:5} pairs ({:4.1}% of bank, {:4.1} GiB)  {:5.1}% of second-half reads  (random {:4.1}%)",
                            n,
                            100.0 * n as f64 / idx.len() as f64,
                            n as f64 * prof.expert_bytes as f64 / (1u64 << 30) as f64,
                            100.0 * cum as f64 / tsum as f64,
                            100.0 * n as f64 / idx.len() as f64
                        );
                    }
                }
            }
        }
        prof::pred_dump();
        Ok(prof)
    }

    /// Phase totals accumulated since `profile_reset` (needs `KORE_PROFILE=1`).
    pub fn profile_dump(steps: usize) {
        prof::dump(steps);
    }

    /// Per-tensor matvec timing, for separating kernel throughput from per-call
    /// cost. `KORE_BENCH=1` calls this once; it is diagnostics only.
    pub fn bench(&self, reps: usize) {
        let x: Vec<f32> = (0..self.cfg.hidden)
            .map(|i| (i % 17) as f32 * 0.01 + 0.3)
            .collect();
        let cases: Vec<(&str, &QW, usize, usize)> = vec![
            (
                "attn_q blk0",
                &self.layers[0].attn.wq,
                self.layers[0].attn.wq.n,
                self.layers[0].attn.wq.m,
            ),
            (
                "attn_k blk0",
                &self.layers[0].attn.wk,
                self.layers[0].attn.wk.n,
                self.layers[0].attn.wk.m,
            ),
            (
                "attn_v blk0",
                &self.layers[0].attn.wv,
                self.layers[0].attn.wv.n,
                self.layers[0].attn.wv.m,
            ),
            (
                "attn_o blk0",
                &self.layers[0].attn.wo,
                self.layers[0].attn.wo.n,
                self.layers[0].attn.wo.m,
            ),
            ("head", &self.head, self.head.n, self.head.m),
        ];
        for (name, w, n, _m) in cases {
            let mut out = vec![0.0f32; n];
            let _ = gemv_par(w, &x, &mut out);
            let t = std::time::Instant::now();
            for _ in 0..reps {
                let _ = gemv_par(w, &x, &mut out);
            }
            let dt = t.elapsed().as_secs_f64();
            let bytes = w.byte_len() as f64;
            eprintln!(
                "  {name:<12} {:>7} rows {:>4.2} MiB  {:7.1} us/call  {:6.1} GB/s  [{:?}]",
                n,
                bytes / 1048576.0,
                dt * 1e6 / reps as f64,
                bytes * reps as f64 / dt / 1e9,
                w.kind
            );
        }
        // The routed experts: same shape, one Q4_K and one Q6_K layer.
        for (label, li) in [("expert blk2", 2usize), ("expert blk6", 6usize)] {
            if let Mlp::Moe(m) = &self.layers[li].mlp {
                let w = m.gate_exps.expert(0, m.gate_stride);
                let mut out = vec![0.0f32; w.n];
                let _ = gemv_par(&w, &x, &mut out);
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    let _ = gemv_par(&w, &x, &mut out);
                }
                let dt = t.elapsed().as_secs_f64();
                let bytes = w.byte_len() as f64;
                eprintln!(
                    "  {label:<12} {:>7} rows {:>4.2} MiB  {:7.1} us/call  {:6.1} GB/s  [{:?}]",
                    w.n,
                    bytes / 1048576.0,
                    dt * 1e6 / reps as f64,
                    bytes * reps as f64 / dt / 1e9,
                    w.kind
                );
            }
        }
    }

    /// Zero the phase accumulators (so prefill is not mixed into decode).
    pub fn profile_reset() {
        prof::reset();
    }
}
