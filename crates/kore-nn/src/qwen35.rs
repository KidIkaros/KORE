//! Qwen3.5 hybrid model (PrismML Ternary Bonsai): gated DeltaNet linear
//! attention layers interleaved with full-attention layers, Q2_0 quantized
//! weights, runtime Hadamard activation transforms.
//!
//! Reference contract (from GGUF metadata + llama.cpp qwen35 model):
//! - Layers: `(i+1) % full_attention_interval == 0` => full attention,
//!   else gated DeltaNet.
//! - All folded weights store W' = H D W on the input axis (D = explicit
//!   signs, H = normalized 512-block WHT). Every matmul against a folded
//!   weight first transforms its activation x' = H D x (signs, then WHT
//!   per block) -- this includes the tied lm head (v2 `tied_output`).
//!   `inverse_weight_names` (token_embd) rows are stored folded; the
//!   embedding lookup restores e = D H e' (WHT, then signs).
//! - DeltaNet step per head: S *= exp(g); d = beta*(v - S^T k);
//!   S += k x d^T; out = S^T q / sqrt(d_state).

use std::path::Path;
use std::sync::Arc;

use kore_core::{KoreError, Result};

use crate::gguf_min::{MiniGguf, MetaVal};
use kore_kernels::q2_0::gemv_q2_0;

fn had_off() -> bool {
    static M: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *M.get_or_init(|| std::env::var("KORE_HAD_OFF").is_ok())
}

fn err(msg: &str) -> KoreError {
    KoreError::StorageError(format!("qwen35: {msg}"))
}

/// Below this many cached keys the merge pass costs more than the sharding saves,
/// so decode attention uses the single-workgroup kernel. Overridable with
/// `KORE_ATTN_SPLIT_MIN` (set it above the context length to disable sharding).
const ATTN_SPLIT_MIN_SEQ_K: usize = 256;

fn attn_split_min_seq_k() -> usize {
    static M: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        std::env::var("KORE_ATTN_SPLIT_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(ATTN_SPLIT_MIN_SEQ_K)
    })
}

/// Keys per shard targeted by [`attn_shard_count`]. Swept on the RTX 2070
/// Max-Q (see `bench_attn`): ~64 keys per shard is the flat part of the curve —
/// below that a shard stops paying for its per-workgroup Q staging, above it the
/// shard count (and so the merge width) grows without buying parallelism.
const ATTN_KEYS_PER_SHARD: usize = 64;

/// Number of KV shards for a decode attention of `seq_k` keys.
///
/// Capped at 64: the merge pass reads `n_splits * head_dim` floats per head, so
/// an unbounded shard count would make the merge the new bottleneck, and the
/// measured benefit flattens past ~64 anyway.
fn attn_shard_count(seq_k: usize) -> usize {
    seq_k.div_ceil(ATTN_KEYS_PER_SHARD).clamp(1, 64)
}

/// Whether the split-KV kernel can run this model's attention shape at this
/// context. Sharding is only an optimisation — whenever it does not apply the
/// caller must fall back to the single-group kernel rather than fail.
fn attn_can_shard(cfg: &Qwen35Config, seq_k: usize) -> bool {
    seq_k >= attn_split_min_seq_k()
        && cfg.head_dim % 32 == 0
        && cfg.head_dim <= 256
        && cfg.n_kv != 0
        && cfg.n_heads % cfg.n_kv == 0
}

#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub n_layers: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub theta: f32,
    pub eps: f32,
    pub full_attn_interval: usize,
    pub ssm_conv: usize,
    pub ssm_state: usize,
    pub ssm_groups: usize,
    pub ssm_dt_rank: usize,
    pub ssm_inner: usize,
    pub vocab: usize,
    pub context: usize,
    pub hadamard_block: usize,
    pub eos: u32,
}

// ---------- Hadamard transform ----------

/// In-place normalized WHT on a power-of-2 block.
fn wht_inplace(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let s = 1.0 / (n as f32).sqrt();
    for v in x.iter_mut() {
        *v *= s;
    }
}

/// D H x: per-block WHT then sign multiply. Inverse of the fold; used to
/// restore rotated embedding rows after lookup.
fn hadamard_t(x: &mut [f32], signs: &[f32], block: usize) {
    debug_assert_eq!(x.len(), signs.len());
    for (i, xs) in x.chunks_exact_mut(block).enumerate() {
        wht_inplace(xs);
        let sb = &signs[i * block..(i + 1) * block];
        for (v, s) in xs.iter_mut().zip(sb) {
            *v *= s;
        }
    }
}

/// H D x: sign multiply then per-block WHT. Same direction as the weight
/// fold; applied to every folded matmul's input (incl. the tied head).
fn hadamard_t_inv(x: &mut [f32], signs: &[f32], block: usize) {
    for (i, xs) in x.chunks_exact_mut(block).enumerate() {
        let sb = &signs[i * block..(i + 1) * block];
        for (v, s) in xs.iter_mut().zip(sb) {
            *v *= s;
        }
        wht_inplace(xs);
    }
}

// ---------- small helpers ----------

fn rms_norm_into(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ss + eps).sqrt();
    for ((o, v), g) in out.iter_mut().zip(x).zip(w) {
        *o = v * inv * g;
    }
}

fn l2_norm_inplace(x: &mut [f32], eps: f32) {
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let inv = 1.0 / (ss + eps * eps).max(1e-30).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// F32 GEMV on GGUF layout [m, n] (m contiguous): out[j] = sum_i w[j*m+i] x[i].
fn gemv_f32(w: &[f32], m: usize, x: &[f32], out: &mut [f32]) {
    for (o, row) in out.iter_mut().zip(w.chunks_exact(m)) {
        *o = row.iter().zip(x).map(|(a, b)| a * b).sum();
    }
}

// ---------- weights ----------

/// Packed Q2_0 weight, mmap-backed; bytes sliced on demand.
#[derive(Clone)]
struct Q2W {
    gg: Arc<MiniGguf>,
    name: String,
    m: usize,
    n: usize,
}

impl Q2W {
    fn bytes(&self) -> &[u8] {
        self.gg.tensor_bytes(&self.name).expect("q2w bytes")
    }
    /// out[n] = W' * x (x must already be Hadamard-transformed).
    fn gemv(&self, x: &[f32], out: &mut [f32]) {
        gemv_q2_0(self.bytes(), self.m, self.n, x, out).expect("q2 gemv");
    }
}

struct DeltaLayer {
    attn_norm: Vec<f32>,
    post_norm: Vec<f32>,
    qkv: Q2W,     // [hidden, qkv_dim]
    z: Q2W,       // [hidden, hidden]
    conv: Vec<f32>, // [conv_k, conv_ch] ne0=conv_k contiguous per channel
    a: Vec<f32>,  // [dt_rank] stored as -exp(A_log)
    alpha_w: Vec<f32>, // [hidden, dt_rank]
    beta_w: Vec<f32>,  // [hidden, dt_rank]
    dt_bias: Vec<f32>, // [dt_rank]
    ssm_norm: Vec<f32>, // [head_v_dim]
    out: Q2W,     // [hidden, hidden]
    ffn_gate: Q2W,
    ffn_up: Q2W,
    ffn_down: Q2W,
}

struct FullLayer {
    attn_norm: Vec<f32>,
    post_norm: Vec<f32>,
    wq: Q2W, // [hidden, 2*n_heads*head_dim] interleaved q|gate per head
    wk: Q2W, // [hidden, n_kv*head_dim]
    wv: Q2W,
    q_norm: Vec<f32>, // [head_dim]
    k_norm: Vec<f32>,
    wo: Q2W,
    ffn_gate: Q2W,
    ffn_up: Q2W,
    ffn_down: Q2W,
}

enum Layer {
    Delta(DeltaLayer),
    Full(FullLayer),
}

// ---------- state ----------

pub struct DeltaState {
    /// conv history: last conv_k-1 inputs per channel, flat [ch*(k-1)].
    conv_hist: Vec<f32>,
    /// per-head state: [h * S*S + j*S + i] = S[i][j]
    state: Vec<f32>,
}

pub struct AttnState {
    k: Vec<f32>, // [pos * kv_dim + c]
    v: Vec<f32>,
    len: usize,
}

pub struct Qwen35Model {
    pub cfg: Qwen35Config,
    emb: Q2W,
    layers: Vec<Layer>,
    out_norm: Vec<f32>,
    /// sign tables: table for width `w` selected by sign_widths order.
    sign_tables: Vec<(usize, Vec<f32>)>,
    head_inv: bool,
}

impl Qwen35Model {
    pub fn from_gguf(path: &Path) -> Result<Self> {
        let gg = MiniGguf::load(path)?;
        let meta = &gg.meta;

        let get_u = |k: &str| -> Result<usize> {
            meta.get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| err(&format!("missing meta '{k}'")))
        };
        let get_f = |k: &str, d: f64| -> f64 {
            meta.get(k).and_then(|v| v.as_f64()).unwrap_or(d)
        };

        let arch = gg.get_str("general.architecture").unwrap_or("");
        if arch != "qwen35" {
            return Err(err(&format!("arch '{arch}' != qwen35")));
        }

        let cfg = Qwen35Config {
            n_layers: get_u("qwen35.block_count")?,
            hidden: get_u("qwen35.embedding_length")?,
            ffn: get_u("qwen35.feed_forward_length")?,
            n_heads: get_u("qwen35.attention.head_count")?,
            n_kv: get_u("qwen35.attention.head_count_kv")?,
            head_dim: get_u("qwen35.attention.key_length")?,
            rope_dim: get_u("qwen35.rope.dimension_count").unwrap_or(0),
            theta: get_f("qwen35.rope.freq_base", 1e7) as f32,
            eps: get_f("qwen35.attention.layer_norm_rms_eps", 1e-6) as f32,
            full_attn_interval: get_u("qwen35.full_attention_interval")?,
            ssm_conv: get_u("qwen35.ssm.conv_kernel")?,
            ssm_state: get_u("qwen35.ssm.state_size")?,
            ssm_groups: get_u("qwen35.ssm.group_count")?,
            ssm_dt_rank: get_u("qwen35.ssm.time_step_rank")?,
            ssm_inner: get_u("qwen35.ssm.inner_size")?,
            vocab: gg.tensor("token_embd.weight")?.dims[1],
            context: get_u("qwen35.context_length").unwrap_or(8192),
            hadamard_block: get_u("prism.hadamard.block_size").unwrap_or(512),
            eos: meta
                .get("tokenizer.ggml.eos_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(248046),
        };

        // sign tables
        let widths: Vec<usize> = match meta.get("prism.hadamard.sign_widths") {
            Some(MetaVal::I64Arr(v)) => v.iter().map(|&x| x as usize).collect(),
            _ => return Err(err("missing prism.hadamard.sign_widths")),
        };
        let values: Vec<f32> = match meta.get("prism.hadamard.sign_values") {
            Some(MetaVal::F32Arr(v)) => v.clone(),
            Some(MetaVal::I64Arr(v)) => v.iter().map(|&x| x as f32).collect(),
            _ => return Err(err("missing prism.hadamard.sign_values")),
        };
        let total: usize = widths.iter().sum();
        if values.len() < total {
            return Err(err("sign_values shorter than sign_widths"));
        }
        let mut sign_tables = Vec::new();
        let mut off = 0usize;
        for &w in &widths {
            sign_tables.push((w, values[off..off + w].to_vec()));
            off += w;
        }

        let q2w = |name: String, gg: &Arc<MiniGguf>| -> Result<Q2W> {
            let t = gg.tensor(&name)?;
            if t.type_id != kore_kernels::q2_0::Q2_0_TYPE_ID {
                return Err(err(&format!("tensor '{name}' type {} != Q2_0", t.type_id)));
            }
            Ok(Q2W {
                gg: gg.clone(),
                name,
                m: t.dims[0],
                n: t.dims[1],
            })
        };
        let f32w = |name: &str| -> Result<Vec<f32>> { gg.tensor_f32(name) };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}");
            let attn_norm = f32w(&p("attn_norm.weight"))?;
            let post_norm = f32w(&p("post_attention_norm.weight"))?;
            let fg = q2w(p("ffn_gate.weight"), &gg)?;
            let fu = q2w(p("ffn_up.weight"), &gg)?;
            let fd = q2w(p("ffn_down.weight"), &gg)?;

            if (i + 1) % cfg.full_attn_interval == 0 {
                layers.push(Layer::Full(FullLayer {
                    attn_norm,
                    post_norm,
                    wq: q2w(p("attn_q.weight"), &gg)?,
                    wk: q2w(p("attn_k.weight"), &gg)?,
                    wv: q2w(p("attn_v.weight"), &gg)?,
                    q_norm: f32w(&p("attn_q_norm.weight"))?,
                    k_norm: f32w(&p("attn_k_norm.weight"))?,
                    wo: q2w(p("attn_output.weight"), &gg)?,
                    ffn_gate: fg,
                    ffn_up: fu,
                    ffn_down: fd,
                }));
            } else {
                layers.push(Layer::Delta(DeltaLayer {
                    attn_norm,
                    post_norm,
                    qkv: q2w(p("attn_qkv.weight"), &gg)?,
                    z: q2w(p("attn_gate.weight"), &gg)?,
                    conv: f32w(&p("ssm_conv1d.weight"))?,
                    a: f32w(&p("ssm_a"))?,
                    alpha_w: f32w(&p("ssm_alpha.weight"))?,
                    beta_w: f32w(&p("ssm_beta.weight"))?,
                    dt_bias: f32w(&p("ssm_dt.bias"))?,
                    ssm_norm: f32w(&p("ssm_norm.weight"))?,
                    out: q2w(p("ssm_out.weight"), &gg)?,
                    ffn_gate: fg,
                    ffn_up: fu,
                    ffn_down: fd,
                }));
            }
        }

        let emb = q2w("token_embd.weight".to_string(), &gg)?;
        let out_norm = f32w("output_norm.weight")?;

        Ok(Self {
            cfg,
            emb,
            layers,
            out_norm,
            sign_tables,
            head_inv: true,
        })
    }

    fn signs_for(&self, width: usize) -> Result<&[f32]> {
        self.sign_tables
            .iter()
            .find(|(w, _)| *w == width)
            .map(|(_, t)| t.as_slice())
            .ok_or_else(|| err(&format!("no sign table for width {width}")))
    }

    /// D H x in place (embedding restore direction).
    fn apply_t(&self, x: &mut [f32]) -> Result<()> {
        if had_off() {
            return Ok(());
        }
        let signs = self.signs_for(x.len())?;
        hadamard_t(x, signs, self.cfg.hadamard_block);
        Ok(())
    }

    /// H D x in place (folded-matmul input direction, incl. tied head).
    fn apply_t_inv(&self, x: &mut [f32]) -> Result<()> {
        if had_off() {
            return Ok(());
        }
        let signs = self.signs_for(x.len())?;
        hadamard_t_inv(x, signs, self.cfg.hadamard_block);
        Ok(())
    }

    /// Embedding lookup + inverse-rotation restore.
    fn embed(&self, token: u32) -> Result<Vec<f32>> {
        let m = self.emb.m;
        let row_bytes = (m / 64) * 18;
        let bytes = self.emb.bytes();
        let row = &bytes[token as usize * row_bytes..(token as usize + 1) * row_bytes];
        let mut x = kore_kernels::q2_0::dequantize_q2_0(row, m)?;
        // inverse-lookup restore: D H x (WHT then signs)
        self.apply_t(&mut x)?;
        Ok(x)
    }

    /// RoPE over first rope_dim elements. Qwen3.5 uses IMROPE which pairs
    /// (j, j + n_rot/2) (neox-style half-split), text positions equal across
    /// mrope sections so it collapses to standard rope.
    fn rope(&self, x: &mut [f32], pos: usize) {
        let n_rot = self.cfg.rope_dim;
        for j in 0..n_rot / 2 {
            let freq = self.cfg.theta.powf(-2.0 * j as f32 / n_rot as f32);
            let ang = pos as f32 * freq;
            let (c, s) = (ang.cos(), ang.sin());
            let (a, b) = (x[j], x[j + n_rot / 2]);
            x[j] = a * c - b * s;
            x[j + n_rot / 2] = a * s + b * c;
        }
    }

    fn ffn(&self, x: &mut [f32], post_norm: &[f32], gate: &Q2W, up: &Q2W, down: &Q2W) -> Result<()> {
        let mut h = vec![0.0f32; self.cfg.hidden];
        rms_norm_into(x, post_norm, self.cfg.eps, &mut h);
        self.apply_t_inv(&mut h)?;
        let mut g = vec![0.0f32; self.cfg.ffn];
        let mut u = vec![0.0f32; self.cfg.ffn];
        gate.gemv(&h, &mut g);
        up.gemv(&h, &mut u);
        let mut act: Vec<f32> = g.iter().zip(&u).map(|(a, b)| silu(*a) * b).collect();
        self.apply_t_inv(&mut act)?;
        let mut o = vec![0.0f32; self.cfg.hidden];
        down.gemv(&act, &mut o);
        for (xv, ov) in x.iter_mut().zip(&o) {
            *xv += ov;
        }
        Ok(())
    }

    fn delta_step(&self, l: &DeltaLayer, st: &mut DeltaState, x: &mut [f32]) -> Result<()> {
        let cfg = &self.cfg;
        let hidden = cfg.hidden;
        let n_k = cfg.ssm_groups;
        let n_v = cfg.ssm_dt_rank;
        let d_k = cfg.ssm_state;
        let d_v = cfg.ssm_inner / n_v;
        let qkv_dim = d_k * n_k * 2 + d_v * n_v;
        let conv_k = cfg.ssm_conv;

        let mut h = vec![0.0f32; hidden];
        rms_norm_into(x, &l.attn_norm, cfg.eps, &mut h);

        let mut ht = h.clone();
        self.apply_t_inv(&mut ht)?;

        let mut qkv = vec![0.0f32; qkv_dim];
        l.qkv.gemv(&ht, &mut qkv);
        let mut z = vec![0.0f32; hidden];
        l.z.gemv(&ht, &mut z);

        let mut beta = vec![0.0f32; n_v];
        gemv_f32(&l.beta_w, hidden, &h, &mut beta);
        for b in beta.iter_mut() {
            *b = sigmoid(*b);
        }
        let mut alpha = vec![0.0f32; n_v];
        gemv_f32(&l.alpha_w, hidden, &h, &mut alpha);
        let mut gate = vec![0.0f32; n_v];
        for i in 0..n_v {
            gate[i] = softplus(alpha[i] + l.dt_bias[i]) * l.a[i];
        }

        // causal depthwise conv over qkv channels, then silu
        let conv_dim = qkv_dim;
        let hist_w = conv_k - 1;
        // st.conv_hist layout: [c * hist_w + slot] where slot 0 = oldest
        for c in 0..conv_dim {
            let mut acc = 0.0f32;
            for k in 0..hist_w {
                acc += l.conv[c * conv_k + k] * st.conv_hist[c * hist_w + k];
            }
            acc += l.conv[c * conv_k + conv_k - 1] * qkv[c];
            // shift history
            for k in 0..hist_w - 1 {
                st.conv_hist[c * hist_w + k] = st.conv_hist[c * hist_w + k + 1];
            }
            st.conv_hist[c * hist_w + hist_w - 1] = qkv[c];
            qkv[c] = silu(acc);
        }

        // split conv output: [q | k | v], heads of d_k/d_k/d_v
        let mut q = qkv[..d_k * n_k].to_vec();
        let mut k = qkv[d_k * n_k..d_k * n_k * 2].to_vec();
        let v = &qkv[d_k * n_k * 2..];

        // l2-norm q,k per head
        for i in 0..n_k {
            l2_norm_inplace(&mut q[i * d_k..(i + 1) * d_k], cfg.eps);
            l2_norm_inplace(&mut k[i * d_k..(i + 1) * d_k], cfg.eps);
        }

        // gated delta rule per v-head (n_v == n_k here)
        let scale = 1.0 / (d_v as f32).sqrt();
        let mut out = vec![0.0f32; n_v * d_v];
        for hv in 0..n_v {
            let s = &mut st.state[hv * d_v * d_v..(hv + 1) * d_v * d_v];
            let ki = &k[hv * d_k..(hv + 1) * d_k];
            let qi = &q[hv * d_k..(hv + 1) * d_k];
            let vi = &v[hv * d_v..(hv + 1) * d_v];
            let decay = gate[hv].exp();
            for val in s.iter_mut() {
                *val *= decay;
            }
            // delta[j] = (v[j] - sum_i S[i][j] k[i]) * beta ; S[i][j] += k[i] delta[j]
            // s layout: s[j * d_v + i] = S[i][j]
            let mut delta = vec![0.0f32; d_v];
            for j in 0..d_v {
                let row = &s[j * d_v..(j + 1) * d_v];
                let dot: f32 = row.iter().zip(ki).map(|(a, b)| a * b).sum();
                delta[j] = (vi[j] - dot) * beta[hv];
            }
            for j in 0..d_v {
                let row = &mut s[j * d_v..(j + 1) * d_v];
                for (i, sv) in row.iter_mut().enumerate() {
                    *sv += ki[i] * delta[j];
                }
            }
            for j in 0..d_v {
                let row = &s[j * d_v..(j + 1) * d_v];
                let dot: f32 = row.iter().zip(qi).map(|(a, b)| a * b).sum();
                out[hv * d_v + j] = dot * scale;
            }
        }

        // gated rmsnorm per head then concat
        let mut o = vec![0.0f32; n_v * d_v];
        for hv in 0..n_v {
            let mut tmp = out[hv * d_v..(hv + 1) * d_v].to_vec();
            let mut normed = vec![0.0f32; d_v];
            rms_norm_into(&tmp, &l.ssm_norm, cfg.eps, &mut normed);
            tmp.copy_from_slice(&normed);
            let zv = &z[hv * d_v..(hv + 1) * d_v];
            for j in 0..d_v {
                o[hv * d_v + j] = tmp[j] * silu(zv[j]);
            }
        }

        self.apply_t_inv(&mut o)?;
        let mut proj = vec![0.0f32; hidden];
        l.out.gemv(&o, &mut proj);
        for (xv, pv) in x.iter_mut().zip(&proj) {
            *xv += pv;
        }
        Ok(())
    }

    fn full_attn_step(&self, l: &FullLayer, st: &mut AttnState, x: &mut [f32], pos: usize) -> Result<()> {
        let cfg = &self.cfg;
        let hidden = cfg.hidden;
        let n_h = cfg.n_heads;
        let n_kv = cfg.n_kv;
        let hd = cfg.head_dim;
        let kv_dim = n_kv * hd;

        let mut h = vec![0.0f32; hidden];
        rms_norm_into(x, &l.attn_norm, cfg.eps, &mut h);
        self.apply_t_inv(&mut h)?;

        let mut qg = vec![0.0f32; n_h * hd * 2];
        l.wq.gemv(&h, &mut qg);
        let mut kk = vec![0.0f32; kv_dim];
        l.wk.gemv(&h, &mut kk);
        let mut vv = vec![0.0f32; kv_dim];
        l.wv.gemv(&h, &mut vv);

        // per-head q rmsnorm + gate extraction, k rmsnorm
        let mut q = vec![0.0f32; n_h * hd];
        let mut g = vec![0.0f32; n_h * hd];
        for i in 0..n_h {
            let qsrc = &qg[i * 2 * hd..i * 2 * hd + hd];
            let gsrc = &qg[i * 2 * hd + hd..(i + 1) * 2 * hd];
            rms_norm_into(qsrc, &l.q_norm, cfg.eps, &mut q[i * hd..(i + 1) * hd]);
            g[i * hd..(i + 1) * hd].copy_from_slice(gsrc);
        }
        for i in 0..n_kv {
            let mut tmp = kk[i * hd..(i + 1) * hd].to_vec();
            let mut normed = vec![0.0f32; hd];
            rms_norm_into(&tmp, &l.k_norm, cfg.eps, &mut normed);
            tmp.copy_from_slice(&normed);
            kk[i * hd..(i + 1) * hd].copy_from_slice(&tmp);
        }

        // rope on first rope_dim elements (interleaved)
        for i in 0..n_h {
            self.rope(&mut q[i * hd..(i + 1) * hd], pos);
        }
        for i in 0..n_kv {
            self.rope(&mut kk[i * hd..(i + 1) * hd], pos);
        }

        // append to cache
        st.k.extend_from_slice(&kk);
        st.v.extend_from_slice(&vv);
        st.len += 1;

        // attention per q-head; kv head = i / (n_h / n_kv)
        let group = n_h / n_kv;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0.0f32; n_h * hd];
        for i in 0..n_h {
            let kvh = i / group;
            let qi = &q[i * hd..(i + 1) * hd];
            let mut scores = vec![0.0f32; st.len];
            for p in 0..st.len {
                let kp = &st.k[p * kv_dim + kvh * hd..p * kv_dim + (kvh + 1) * hd];
                scores[p] = qi.iter().zip(kp).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                sum += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum;
            }
            for j in 0..hd {
                let mut acc = 0.0f32;
                for p in 0..st.len {
                    acc += scores[p] * st.v[p * kv_dim + kvh * hd + j];
                }
                out[i * hd + j] = acc * sigmoid(g[i * hd + j]);
            }
        }

        self.apply_t_inv(&mut out)?;
        let mut proj = vec![0.0f32; hidden];
        l.wo.gemv(&out, &mut proj);
        for (xv, pv) in x.iter_mut().zip(&proj) {
            *xv += pv;
        }
        Ok(())
    }

    /// Logits for a hidden state (tied embedding head, inverse transform).
    pub fn logits(&self, x: &[f32]) -> Result<Vec<f32>> {
        let mut h = vec![0.0f32; self.cfg.hidden];
        rms_norm_into(x, &self.out_norm, self.cfg.eps, &mut h);
        if self.head_inv {
            self.apply_t_inv(&mut h)?;
        }
        let mut logits = vec![0.0f32; self.cfg.vocab];
        gemv_q2_0(self.emb.bytes(), self.emb.m, self.cfg.vocab, &h, &mut logits)?;
        Ok(logits)
    }

    /// Forward one token; returns logits.
    pub fn forward(
        &self,
        token: u32,
        pos: usize,
        dstate: &mut [DeltaState],
        astate: &mut [AttnState],
    ) -> Result<Vec<f32>> {
        let mut x = self.embed(token)?;
        let mut di = 0;
        let mut ai = 0;
        for layer in &self.layers {
            match layer {
                Layer::Delta(l) => {
                    self.delta_step(l, &mut dstate[di], &mut x)?;
                    di += 1;
                    self.ffn(&mut x, &l.post_norm, &l.ffn_gate, &l.ffn_up, &l.ffn_down)?;
                }
                Layer::Full(l) => {
                    self.full_attn_step(l, &mut astate[ai], &mut x, pos)?;
                    ai += 1;
                    self.ffn(&mut x, &l.post_norm, &l.ffn_gate, &l.ffn_up, &l.ffn_down)?;
                }
            }
        }
        self.logits(&x)
    }

    /// Allocate fresh states for a sequence.
    pub fn new_state(&self) -> (Vec<DeltaState>, Vec<AttnState>) {
        let cfg = &self.cfg;
        let conv_dim = cfg.ssm_inner + 2 * cfg.ssm_groups * cfg.ssm_state;
        let mut d = Vec::new();
        let mut a = Vec::new();
        for layer in &self.layers {
            match layer {
                Layer::Delta(_) => d.push(DeltaState {
                    conv_hist: vec![0.0; conv_dim * (cfg.ssm_conv - 1)],
                    state: vec![0.0; cfg.ssm_dt_rank * cfg.ssm_state * cfg.ssm_state],
                }),
                Layer::Full(_) => a.push(AttnState {
                    k: Vec::new(),
                    v: Vec::new(),
                    len: 0,
                }),
            }
        }
        (d, a)
    }

    /// Greedy argmax over logits.
    pub fn argmax(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    /// Greedy generation loop (CPU reference path).
    pub fn generate_greedy(&self, tokens: &[u32], n_new: usize, eos: u32) -> Result<Vec<u32>> {
        let (mut dstate, mut astate) = self.new_state();
        let mut out = Vec::with_capacity(n_new);
        let mut logits = Vec::new();
        for (i, &t) in tokens.iter().enumerate() {
            logits = self.forward(t, i, &mut dstate, &mut astate)?;
        }
        for k in 0..n_new {
            let tok = Self::argmax(&logits);
            if tok == eos {
                break;
            }
            out.push(tok);
            logits = self.forward(tok, tokens.len() + k, &mut dstate, &mut astate)?;
        }
        Ok(out)
    }
}

// ---------- Vulkan GPU path ----------

use kore_core::device::{get_backend, DeviceBackend};
use kore_core::{Device, DType, Storage, Tensor};

struct GpuDelta {
    attn_norm: Tensor,
    post_norm: Tensor,
    conv: Tensor,
    a: Tensor,
    alpha_w: Tensor,
    beta_w: Tensor,
    dt_bias: Tensor,
    ssm_norm: Tensor,
    qkv: Tensor,
    z: Tensor,
    out: Tensor,
    ffn_gate: Tensor,
    ffn_up: Tensor,
    ffn_down: Tensor,
}

struct GpuFull {
    attn_norm: Tensor,
    post_norm: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    ffn_gate: Tensor,
    ffn_up: Tensor,
    ffn_down: Tensor,
}

enum GpuLayer {
    Delta(GpuDelta),
    Full(GpuFull),
}

/// Mutable per-layer device state.
pub enum GpuState {
    Delta { hist: Tensor, state: Tensor },
    Full { kc: Tensor, vc: Tensor, k_scales: Tensor, v_scales: Tensor, len: usize },
}

/// Qwen3.5 with all weights device-resident; single-token decode via the
/// fused Vulkan kernels. Requires `n_k == n_v` and `d_k == d_v` (true for
/// Bonsai-2B) plus head_dim == 256.
pub struct Qwen35Gpu {
    cfg: Qwen35Config,
    device: Device,
    backend: Arc<dyn DeviceBackend>,
    emb: Tensor, // U8 packed Q2_0 rows
    emb_m: usize,
    layers: Vec<GpuLayer>,
    out_norm: Tensor,
    signs: Vec<(usize, Tensor)>,
}

impl Qwen35Gpu {
    /// Materialize a device copy of `cpu` on `device`.
    pub fn from_cpu(cpu: &Qwen35Model, device: Device) -> Result<Self> {
        let backend = get_backend(device).ok_or_else(|| err("no backend for device"))?;
        let cfg = cpu.cfg.clone();
        let n_k = cfg.ssm_groups;
        let n_v = cfg.ssm_dt_rank;
        let d_k = cfg.ssm_state;
        let d_v = cfg.ssm_inner / n_v;
        if n_k != n_v || d_k != d_v {
            return Err(err("gpu path requires n_k == n_v and d_k == d_v"));
        }
        if cfg.head_dim != 256 {
            return Err(err("gpu path requires head_dim 256"));
        }

        let up_f32 = |v: &[f32]| -> Result<Tensor> {
            Tensor::from_f32(v, &[v.len()])
                .to(device)
                .map_err(|e| err(&format!("upload: {e}")))
        };
        let up_q2 = |w: &Q2W| -> Result<Tensor> {
            let bytes = kore_kernels::q2_0::repack_q2_0(w.bytes(), w.m, w.n)?;
            let n = bytes.len();
            let s = Storage::from_bytes(DType::U8, n, bytes)?;
            Tensor::from_storage(s, &[n])
                .to(device)
                .map_err(|e| err(&format!("upload q2: {e}")))
        };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for layer in &cpu.layers {
            match layer {
                Layer::Delta(l) => layers.push(GpuLayer::Delta(GpuDelta {
                    attn_norm: up_f32(&l.attn_norm)?,
                    post_norm: up_f32(&l.post_norm)?,
                    conv: up_f32(&l.conv)?,
                    a: up_f32(&l.a)?,
                    alpha_w: up_f32(&l.alpha_w)?,
                    beta_w: up_f32(&l.beta_w)?,
                    dt_bias: up_f32(&l.dt_bias)?,
                    ssm_norm: up_f32(&l.ssm_norm)?,
                    qkv: up_q2(&l.qkv)?,
                    z: up_q2(&l.z)?,
                    out: up_q2(&l.out)?,
                    ffn_gate: up_q2(&l.ffn_gate)?,
                    ffn_up: up_q2(&l.ffn_up)?,
                    ffn_down: up_q2(&l.ffn_down)?,
                })),
                Layer::Full(l) => layers.push(GpuLayer::Full(GpuFull {
                    attn_norm: up_f32(&l.attn_norm)?,
                    post_norm: up_f32(&l.post_norm)?,
                    q_norm: up_f32(&l.q_norm)?,
                    k_norm: up_f32(&l.k_norm)?,
                    wq: up_q2(&l.wq)?,
                    wk: up_q2(&l.wk)?,
                    wv: up_q2(&l.wv)?,
                    wo: up_q2(&l.wo)?,
                    ffn_gate: up_q2(&l.ffn_gate)?,
                    ffn_up: up_q2(&l.ffn_up)?,
                    ffn_down: up_q2(&l.ffn_down)?,
                })),
            }
        }

        let emb = {
            let bytes = kore_kernels::q2_0::repack_q2_0(cpu.emb.bytes(), cpu.emb.m, cpu.emb.n)?;
            let n = bytes.len();
            let s = Storage::from_bytes(DType::U8, n, bytes)?;
            Tensor::from_storage(s, &[n])
                .to(device)
                .map_err(|e| err(&format!("upload emb: {e}")))?
        };

        let mut signs = Vec::new();
        for (w, t) in &cpu.sign_tables {
            signs.push((*w, up_f32(t)?));
        }

        Ok(Self {
            emb_m: cpu.emb.m,
            cfg,
            device,
            backend,
            emb,
            layers,
            out_norm: up_f32(&cpu.out_norm)?,
            signs,
        })
    }

    /// Fresh per-layer states; `max_ctx` bounds the full-attention KV caches.
    pub fn new_state(&self, max_ctx: usize) -> Result<Vec<GpuState>> {
        let cfg = &self.cfg;
        let conv_dim = cfg.ssm_inner + 2 * cfg.ssm_groups * cfg.ssm_state;
        let kv_dim = cfg.n_kv * cfg.head_dim;
        let mut out = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            match layer {
                GpuLayer::Delta(_) => out.push(GpuState::Delta {
                    hist: Tensor::zeros(&[conv_dim * (cfg.ssm_conv - 1)], DType::F32)
                        .to(self.device)
                        .map_err(|e| err(&format!("hist: {e}")))?,
                    state: Tensor::zeros(
                        &[cfg.ssm_dt_rank * cfg.ssm_state * cfg.ssm_state],
                        DType::F32,
                    )
                    .to(self.device)
                    .map_err(|e| err(&format!("state: {e}")))?,
                }),
                GpuLayer::Full(_) => out.push(GpuState::Full {
                    kc: Tensor::zeros(&[max_ctx, kv_dim], DType::U8)
                        .to(self.device)
                        .map_err(|e| err(&format!("kc: {e}")))?,
                    vc: Tensor::zeros(&[max_ctx, kv_dim], DType::U8)
                        .to(self.device)
                        .map_err(|e| err(&format!("vc: {e}")))?,
                    k_scales: Tensor::zeros(&[max_ctx], DType::F32)
                        .to(self.device)
                        .map_err(|e| err(&format!("k_scales: {e}")))?,
                    v_scales: Tensor::zeros(&[max_ctx], DType::F32)
                        .to(self.device)
                        .map_err(|e| err(&format!("v_scales: {e}")))?,
                    len: 0,
                }),
            }
        }
        Ok(out)
    }

    fn signs_t(&self, width: usize) -> Result<&Tensor> {
        self.signs
            .iter()
            .find(|(w, _)| *w == width)
            .map(|(_, t)| t)
            .ok_or_else(|| err(&format!("no gpu sign table width {width}")))
    }

    fn had_fwd(&self, x: &Tensor) -> Result<Tensor> {
        // The signed Hadamard is per-row and the sign table covers one row, so a
        // multi-column batch reuses the single-row table. Keying on `numel` would
        // demand an `m * cols` table that the model does not have; keying on the
        // row width is identical at `m = 1`.
        let width = *x.shape().dims().last().unwrap_or(&x.numel());
        self.backend.hadamard_signed(x, self.signs_t(width)?, 512, 0)
    }

    /// Shared FFN block with the *incoming* residual folded into its post-norm.
    ///
    /// Returns `(x_r, down)` where `x_r = x + res` is the residual stream *before*
    /// the FFN output. The caller carries `down` forward as the pending residual,
    /// so the trailing `x += down` add is absorbed into the next layer's norm
    /// instead of costing its own dispatch.
    #[allow(clippy::too_many_arguments)]
    fn ffn_gpu(
        &self,
        x: &Tensor,
        res: &Tensor,
        post_norm: &Tensor,
        g: &Tensor,
        u: &Tensor,
        d: &Tensor,
        ffn: usize,
        hidden: usize,
    ) -> Result<(Tensor, Tensor)> {
        let b = &self.backend;
        // Fused residual-add + RMSNorm + Hadamard: one dispatch instead of two.
        // The sign table is one row wide (the batch shares it across rows).
        let (x_r, _h, ht) = b.had_pre(0, x, res, post_norm, self.signs_t(hidden)?, self.cfg.eps)?;
        let gv = b.gemv_q2_0(&ht, g, ffn, hidden)?;
        let uv = b.gemv_q2_0(&ht, u, ffn, hidden)?;
        // Fused SwiGLU + Hadamard (the second half of the FFN's activation pair).
        let signs_ffn = self.signs_t(ffn)?;
        let (_gv, _uv, actt) = b.had_pre(2, &gv, &uv, signs_ffn, signs_ffn, self.cfg.eps)?;
        let down = b.gemv_q2_0(&actt, d, hidden, ffn)?;
        Ok((x_r, down))
    }

    /// One decode step; returns the next greedy token.
    pub fn forward(&self, cpu: &Qwen35Model, token: u32, pos: usize, states: &mut [GpuState]) -> Result<u32> {
        let b = &self.backend;
        let cfg = &self.cfg;
        let hidden = cfg.hidden;
        let ffn = cfg.ffn;
        let n_v = cfg.ssm_dt_rank;
        let d_k = cfg.ssm_state;
        let d_v = cfg.ssm_inner / n_v;
        let conv_dim = cfg.ssm_inner + 2 * cfg.ssm_groups * cfg.ssm_state;
        let kv_dim = cfg.n_kv * cfg.head_dim;
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();

        // embed on CPU (row dequant + restore), upload
        let e = cpu.embed(token)?;
        let x0 = Tensor::from_f32(&e, &[1, hidden])
            .to(self.device)
            .map_err(|e2| err(&format!("embed upload: {e2}")))?;

        // The residual stream and the projection still to be added to it. Starting
        // with a zero pending lets every block use the same fused add+rms_norm
        // (`x + 0` is exact), avoiding a special case for layer 0.
        let mut x = x0;
        let mut pending = Tensor::from_f32(&vec![0.0f32; hidden], &[1, hidden])
            .to(self.device)
            .map_err(|e2| err(&format!("zero residual upload: {e2}")))?;

        for (layer, st) in self.layers.iter().zip(states.iter_mut()) {
            match (layer, st) {
                (GpuLayer::Delta(l), GpuState::Delta { hist, state }) => {
                    // Fold the pending residual in, normalize and rotate in one dispatch.
                    let (x_r, h, ht) = b.had_pre(0, &x, &pending, &l.attn_norm, self.signs_t(hidden)?, cfg.eps)?;
                    let qkv = b.gemv_q2_0(&ht, &l.qkv, conv_dim, hidden)?;
                    let z = b.gemv_q2_0(&ht, &l.z, hidden, hidden)?;
                    let (beta, gate) = b.gdn_gates(&h, &l.alpha_w, &l.beta_w, &l.dt_bias, &l.a, hidden, n_v)?;
                    let conv_out = b.gdn_conv(&qkv, &l.conv, hist, conv_dim, cfg.ssm_conv)?;
                    let o = b.gdn_step(&conv_out, &z, state, &beta, &gate, &l.ssm_norm, n_v, d_k, d_v, cfg.eps)?;
                    let ot = self.had_fwd(&o)?;
                    let proj = b.gemv_q2_0(&ot, &l.out, hidden, hidden)?;
                    let (x_next, down) = self.ffn_gpu(&x_r, &proj, &l.post_norm, &l.ffn_gate, &l.ffn_up, &l.ffn_down, ffn, hidden)?;
                    x = x_next;
                    pending = down;
                }
                (GpuLayer::Full(l), GpuState::Full { kc, vc, k_scales, v_scales, len }) => {
                    let (x_r, _h, ht) = b.had_pre(0, &x, &pending, &l.attn_norm, self.signs_t(hidden)?, cfg.eps)?;
                    let qg = b.gemv_q2_0(&ht, &l.wq, 2 * cfg.n_heads * cfg.head_dim, hidden)?;
                    let kk = b.gemv_q2_0(&ht, &l.wk, kv_dim, hidden)?;
                    let vv = b.gemv_q2_0(&ht, &l.wv, kv_dim, hidden)?;
                    let (q, k, g) = b.qknorm_rope(
                        &qg, &kk, &l.q_norm, &l.k_norm,
                        cfg.n_heads, cfg.n_kv, cfg.head_dim, cfg.rope_dim,
                        pos, cfg.theta, cfg.eps,
                    )?;

                    // ---- Q8 (INT8) KV-cache path ----
                    // Quantize K/V **on the GPU** (no host round-trip, unlike the
                    // previous per-token CPU quantization which cost two device
                    // syncs per full-attention layer) and append to the INT8 cache.
                    // Decode then reads 4x less VRAM than F32.
                    let (k_q8_t, k_scale_new) = b.quantize_q8_rows(&k)?;
                    let (v_q8_t, v_scale_new) = b.quantize_q8_rows(&vv)?;

                    let (kv_q8, vv_q8) = b.kv_cache_append_q8(
                        kc, vc, k_scales, v_scales,
                        &k_q8_t, &v_q8_t, &k_scale_new, &v_scale_new, *len,
                    )?;
                    *len += 1;

                    // Decode attention is O(seq_k) KV reads and, without sharding,
                    // runs in only `n_heads` workgroups -- not enough parallel work
                    // to hide memory latency, so it decays to ~1.4 GB/s and dominates
                    // the long-context cliff. Shard the window across n_splits
                    // workgroups per head once the window is big enough to pay for
                    // the merge pass.
                    let seq_k = *len;
                    let attn = if attn_can_shard(cfg, seq_k) {
                        let n_splits = attn_shard_count(seq_k);
                        b.fused_attention_decode_gqa_q8_split(
                            &q, &kv_q8, &vv_q8, &k_scales, &v_scales,
                            cfg.n_heads, cfg.n_kv, cfg.head_dim, scale,
                            n_splits,
                        )?
                    } else {
                        b.fused_attention_decode_gqa_q8(
                            &q, &kv_q8, &vv_q8, &k_scales, &v_scales,
                            cfg.n_heads, cfg.n_kv, cfg.head_dim, scale,
                        )?
                    };
                    // ---- end Q8 path ----------------------------------------------

                    let (_a, _bv, ot) = b.had_pre(3, &attn, &g, self.signs_t(hidden)?, self.signs_t(hidden)?, cfg.eps)?;
                    let proj = b.gemv_q2_0(&ot, &l.wo, hidden, hidden)?;
                    let (x_next, down) = self.ffn_gpu(&x_r, &proj, &l.post_norm, &l.ffn_gate, &l.ffn_up, &l.ffn_down, ffn, hidden)?;
                    x = x_next;
                    pending = down;
                }
                _ => return Err(err("gpu state/layer mismatch")),
            }
        }

        let (_r, _h, ht) = b.had_pre(0, &x, &pending, &self.out_norm, self.signs_t(hidden)?, cfg.eps)?;
        let logits = b.gemv_q2_0(&ht, &self.emb, cfg.vocab, self.emb_m)?;
        let toks = b.argmax_last_dim(&logits)?;
        Ok(toks[0])
    }

    /// Greedy generation on GPU; returns generated tokens.
    pub fn generate_greedy(
        &self,
        cpu: &Qwen35Model,
        tokens: &[u32],
        n_new: usize,
        max_ctx: usize,
    ) -> Result<Vec<u32>> {
        let mut states = self.new_state(max_ctx)?;
        let mut out = Vec::with_capacity(n_new);
        let mut next = 0u32;
        for (i, &t) in tokens.iter().enumerate() {
            next = self.forward(cpu, t, i, &mut states)?;
        }
        for k in 0..n_new {
            let tok = next;
            if tok == cpu.cfg.eos {
                break;
            }
            out.push(tok);
            next = self.forward(cpu, tok, tokens.len() + k, &mut states)?;
        }
        Ok(out)
    }
}
