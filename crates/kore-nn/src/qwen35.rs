//! Qwen3.5 hybrid model: gated DeltaNet linear attention layers interleaved
//! with full-attention layers, packed or unpacked weights, runtime Hadamard
//! activation transforms.
//!
//! Two GGUF arch keys are served, and the arch key is also the metadata prefix
//! (`qwen3_5` in a HuggingFace filename is *not* what the file says):
//!
//! - `qwen35`   -- one dense SwiGLU per layer.
//! - `qwen35moe` -- a router over `expert_count` stacked experts, top
//!   `expert_used_count` of them evaluated, plus a sigmoid-gated shared expert.
//!   Files of this arch may also carry an MTP/`nextn` block; it is a draft head,
//!   not part of the stack, and is skipped.
//!
//! Weight formats: F32, F16, Q2_0, Q4_K, Q5_K, Q6_K. A single file may mix
//! them -- `qwen35moe` files keep the router in F32 and quantize only the
//! experts.
//!
//! Reference contract (from GGUF metadata + llama.cpp `qwen35`/`qwen35moe`):
//! - Layers: `(i+1) % full_attention_interval == 0` => full attention,
//!   else gated DeltaNet.
//! - All folded weights store W' = H D W on the input axis (D = explicit
//!   signs, H = normalized 512-block WHT). Every matmul against a folded
//!   weight first transforms its activation x' = H D x (signs, then WHT
//!   per block) -- this includes the tied lm head (v2 `tied_output`).
//!   `inverse_weight_names` (token_embd) rows are stored folded; the
//!   embedding lookup restores e = D H e' (WHT, then signs). A file without
//!   `prism.hadamard.*` metadata has no transform at all, and folding is
//!   decided per file, never assumed.
//! - DeltaNet step per head: S *= exp(g); d = beta*(v - S^T k);
//!   S += k x d^T; out = S^T q / sqrt(d_state).
//! - DeltaNet q/k are broadcast to the value heads by **tiling** (`hk = hv %
//!   n_k`), not by interleaving: that is what every GGUF of this family is
//!   built for, even where the HF reference does the opposite.
//! - MoE routing is llama.cpp's `build_moe_ffn` with `LLM_FFN_SILU`, softmax
//!   gating and `norm_w = true`: softmax over *all* experts, top-k, then
//!   renormalise. The shared expert sits outside the top-k and is scaled by
//!   `sigmoid(gate_inp_shexp . h)`.
//!
//! End-to-end parity for `qwen35moe` is checked against llama.cpp on a 9 MB
//! conversion of `tiny-random/qwen3.5-moe` (128 experts, top-10, n_k != n_v):
//!
//! ```text
//! python convert_hf_to_gguf.py <hf dir> --outtype f16 --outfile tiny.gguf
//! ref_ids tiny.gguf "The capital of France is" 8      # gen_ids: 132029 192124 633 ...
//! bonsai  tiny.gguf <tokenizer.json> "The capital of France is" 8 cpu
//! ```

use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use kore_core::{KoreError, Result};

use crate::gguf_min::{MetaVal, MiniGguf};
use crate::moe::{CacheOutcome, ExpertCache};
use kore_kernels::k_quant;
use kore_kernels::q2_0::gemv_q2_0;
use oxillama_gguf::{ByteOwner, GgufTensorType, SharedBytes};
use oxillama_quant::{global_dispatcher, QuantTensor};

fn had_off() -> bool {
    static M: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *M.get_or_init(|| std::env::var("KORE_HAD_OFF").is_ok())
}

fn err(msg: &str) -> KoreError {
    KoreError::StorageError(format!("qwen35: {msg}"))
}

/// `KORE_STEP_PROFILE=1`: split one decode step into the regions the no-sync
/// census cannot see -- the per-token activation upload, the layer chain (host
/// encode time), and the argmax readback that ends the step.
///
/// The third number is the interesting one: because nothing syncs inside the
/// chain, the readback *is* the queue drain, so it absorbs every dispatch the
/// GPU had not finished. When `argmax+readback` dominates, the step is
/// GPU-bound and the host round trip costs nothing; when it does not, the host
/// is the limit.
fn step_profile() -> Option<&'static Mutex<[f64; 4]>> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("KORE_STEP_PROFILE").is_some()) {
        return None;
    }
    static P: Mutex<[f64; 4]> = Mutex::new([0.0; 4]);
    Some(&P)
}

fn step_profile_note(us: [f64; 3]) {
    if let Some(p) = step_profile() {
        let mut p = p.lock().expect("step profile");
        p[0] += us[0];
        p[1] += us[1];
        p[2] += us[2];
        p[3] += 1.0;
    }
}

fn step_profile_report(label: &str) {
    let Some(p) = step_profile() else { return };
    let p = p.lock().expect("step profile");
    let n = p[3].max(1.0);
    eprintln!(
        "{label} step profile over {} steps: embed {:.2} ms, layers {:.2} ms (host encode), \
argmax+readback {:.2} ms (queue drain), total {:.2} ms",
        p[3] as usize,
        p[0] / n / 1000.0,
        p[1] / n / 1000.0,
        p[2] / n / 1000.0,
        (p[0] + p[1] + p[2]) / n / 1000.0
    );
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
        && cfg.head_dim.is_multiple_of(32)
        && cfg.head_dim <= 256
        && cfg.n_kv != 0
        && cfg.n_heads.is_multiple_of(cfg.n_kv)
}

/// How much of the vocabulary projection a forward pass needs.
///
/// The head is a `vocab x hidden` matvec and the single largest weight read in a
/// step, but the head is not allowed to touch the recurrent state: it reads the
/// last position's normalized hidden and nothing else. A prefill therefore only
/// ever reads the prediction at its **last** position, and none at all for the
/// chunks that precede it, which makes `Skip` and `Last` pure savings.
#[derive(Clone, Copy)]
enum Head {
    /// Advance state only; produce no token.
    Skip,
    /// Project the last column only (one prefill chunk's prediction).
    Last,
    /// Project every column (a verify pass wants all of them).
    Every,
}

/// The widest batch any pass can feed the ternary GEMV. A workgroup's
/// accumulator keeps four columns live (`grid.y = ceil(m / 4)`), and the
/// backend accepts up to two such row groups.
pub const MAX_BATCH_COLS: usize = 8;

/// Prefill chunk width: the ternary GEMV's column width.
const PREFILL_CHUNK: usize = MAX_BATCH_COLS;

#[derive(Debug, Clone)]
pub struct Qwen35Config {
    /// The GGUF arch key, which is also the metadata prefix: `qwen35` or
    /// `qwen35moe`.
    pub prefix: String,
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
    /// Routed-expert geometry. All zero for a dense `qwen35` file.
    pub n_experts: usize,
    pub n_experts_used: usize,
    pub moe_ffn: usize,
    pub shexp_ffn: usize,
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

/// In-place softmax over a slice, max-subtracted so a large router logit cannot
/// overflow `exp`.
fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
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

/// The packed weight formats the fast path can actually compute.
///
/// `Q2_0` is the ternary format this engine started with. The three K-quants
/// are what real GGUFs ship: a file labelled `Q4_K_M` is a *mix*, and a single
/// one can carry Q4_K, Q5_K and Q6_K tensors all at once (the 9B staging fixture
/// does: Q5_K holds its 24 `ssm_out.weight` matrices).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum QKind {
    /// GGML type 0: plain f32. Routers keep this format even inside an
    /// otherwise K-quant file, so it is not a corner case.
    F32,
    /// GGML type 1: plain f16. An unquantised export is a perfectly ordinary
    /// file, and it is the only way to get an exact f32 parity harness.
    F16,
    /// GGML type 42, 64 weights per 18-byte block.
    Q2_0,
    /// GGML type 2, 32 weights per 18-byte block. The single most common
    /// legacy format on the Hub, and the fallback `Q2_K` reaches for when a row
    /// is too short for a super-block.
    Q4_0,
    /// GGML type 6, 32 weights per 22-byte block. Not a boutique format: a
    /// K-quant file falls back to it whenever a tensor's row width is not a
    /// multiple of 256, which is exactly what a routed expert's `down`
    /// projection looks like (`ne[0] = expert_ffn`, e.g. 192).
    Q5_0,
    /// GGML type 8, 32 weights per 34-byte block. The other half of the same
    /// fallback: `Q4_K_M` picks Q8_0 for the tensors Q5_0 is too lossy for.
    Q8_0,
    /// K-quants, 256 weights per super-block. Q2_K and Q3_K are here because
    /// the checkpoint this engine is actually sized for -- a 35 B-A3B MoE whose
    /// resident tier is 1.27 GB and whose experts are 11.66 GB -- ships them.
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl QKind {
    fn from_type_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            kore_kernels::q2_0::Q2_0_TYPE_ID => Some(Self::Q2_0),
            2 => Some(Self::Q4_0),
            6 => Some(Self::Q5_0),
            8 => Some(Self::Q8_0),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            k_quant::Q4_K_TYPE_ID => Some(Self::Q4K),
            k_quant::Q5_K_TYPE_ID => Some(Self::Q5K),
            k_quant::Q6_K_TYPE_ID => Some(Self::Q6K),
            _ => None,
        }
    }

    fn type_id(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q2_0 => kore_kernels::q2_0::Q2_0_TYPE_ID,
            Self::Q4_0 => 2,
            Self::Q5_0 => 6,
            Self::Q8_0 => 8,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => k_quant::Q4_K_TYPE_ID,
            Self::Q5K => k_quant::Q5_K_TYPE_ID,
            Self::Q6K => k_quant::Q6_K_TYPE_ID,
        }
    }

    /// Weights per packed block.
    const fn block(self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q2_0 => kore_kernels::q2_0::Q2_0_BLOCK,
            Self::Q4_0 | Self::Q5_0 | Self::Q8_0 => 32,
            _ => k_quant::QK_K,
        }
    }

    /// Bytes per packed block.
    const fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q2_0 => kore_kernels::q2_0::Q2_0_BLOCK_BYTES,
            // Legacy 32-weight blocks: f16 scale plus 32 quantized values, and
            // above them 4 bytes of high bits for Q5_0.
            // f16 scale plus 16 bytes of nibbles.
            Self::Q4_0 => 2 + 16,
            Self::Q5_0 => 2 + 4 + 16,
            Self::Q8_0 => 2 + 32,
            // 16 packed scales + 64 quants + f16 d + f16 dmin.
            Self::Q2K => 84,
            // 32 high-bit masks + 64 two-bit quants + 12 scales + f16 d.
            Self::Q3K => 110,
            Self::Q4K => k_quant::Q4_K_BLOCK_BYTES,
            Self::Q5K => k_quant::Q5_K_BLOCK_BYTES,
            Self::Q6K => k_quant::Q6_K_BLOCK_BYTES,
        }
    }

    /// The `oxillama` type tag, which is how the CPU lane names a quantized
    /// kernel.
    ///
    /// The K-quants and the legacy pair both go through `oxillama`; Q2_0 and the
    /// unpacked formats have their own loops and never ask for a tag, so their
    /// value here is a placeholder.
    fn gguf(self) -> GgufTensorType {
        match self {
            Self::Q4_0 => GgufTensorType::Q4_0,
            Self::Q5_0 => GgufTensorType::Q5_0,
            Self::Q8_0 => GgufTensorType::Q8_0,
            Self::Q2K => GgufTensorType::Q2K,
            Self::Q3K => GgufTensorType::Q3K,
            Self::Q4K => GgufTensorType::Q4K,
            Self::Q5K => GgufTensorType::Q5K,
            Self::Q6K => GgufTensorType::Q6K,
            Self::F32 | Self::F16 | Self::Q2_0 => GgufTensorType::Q8_0,
        }
    }
}

/// `oxillama`'s zero-copy byte view over one tensor of a [`MiniGguf`]'s map.
///
/// The K-quant CPU kernels take a `SharedBytes` rather than a slice, so that a
/// weight can be used straight out of the file instead of being copied into
/// RAM first. That is the whole point of the mmap on a box where the model does
/// not fit: a 5.7 GB file must stay 5.7 GB of page cache, not 5.7 GB of RSS.
struct TensorBytes {
    gg: Arc<MiniGguf>,
    name: String,
}

// SAFETY: the payload is a slice of the file's `memmap2::Mmap`, owned by `gg`,
// never written through, and unable to move while `gg` is alive. `tensor_bytes`
// resolves the same range on every call, so the pointer `SharedBytes` caches is
// the one it will dereference for the rest of its life. The view is read-only.
unsafe impl ByteOwner for TensorBytes {
    fn owned_bytes(&self) -> &[u8] {
        self.gg.tensor_bytes(&self.name).unwrap_or(&[])
    }
}

/// A packed weight matrix, mmap-backed and format-agnostic.
///
/// One type covers every quant the engine can run: the byte geometry and the
/// arithmetic are chosen from `kind` inside [`QW::gemv`], so no layer struct has
/// to know — or care — which format a particular file happens to use.
#[derive(Clone)]
struct QW {
    name: String,
    /// Row length (GGUF `ne[0]`): the matvec's input width.
    m: usize,
    /// Row count (GGUF `ne[1]`): the matvec's output width.
    n: usize,
    kind: QKind,
    /// Byte offset of this matrix inside the tensor's payload. Zero for a plain
    /// 2-D weight; `e * stride` for one expert of a stacked `[.., .., n_experts]`
    /// tensor, which is how a `qwen35moe` FFN addresses its experts without
    /// copying (or even touching) the ones it is not using.
    off: usize,
    /// Zero-copy view over the packed bytes, resolved once at load. Keeps the
    /// file's mapping alive for as long as the model is.
    shared: SharedBytes,
    /// The whole-tensor owner, kept so an expert window can be re-sliced
    /// without copying.
    owner: Arc<dyn ByteOwner>,
}

/// Fused GEMV for the K-quant formats unpacked by [`k_quant`].
///
/// One 256-element scratch buffer is reused for the whole matrix, so a row costs
/// a block unpack plus a dot, not an allocation. Only Q2_K/Q3_K take this path:
/// everything else with a kernel is served by `oxillama`, whose AVX2 code is
/// faster and whose layouts are validated against the same fixtures.
fn gemv_k_quant(w: &QW, x: &[f32], out: &mut [f32]) -> Result<()> {
    let bb = w.kind.block_bytes();
    let rb = w.row_bytes();
    let mut scratch = [0.0f32; k_quant::QK_K];
    for (o, row) in out.iter_mut().zip(w.bytes().chunks_exact(rb)) {
        let mut acc = 0.0f32;
        for (b, blk) in row.chunks_exact(bb).enumerate() {
            k_quant::dequant_block(w.kind.type_id(), blk, &mut scratch)
                .map_err(|e| err(&format!("'{}': dequant block: {e}", w.name)))?;
            let base = b * k_quant::QK_K;
            let take = (w.m - base).min(k_quant::QK_K);
            acc += scratch[..take]
                .iter()
                .zip(&x[base..base + take])
                .map(|(a, b)| a * b)
                .sum::<f32>();
        }
        *o = acc;
    }
    Ok(())
}

impl QW {
    /// The packed payload, `n * (m / block) * block_bytes` bytes.
    fn bytes(&self) -> &[u8] {
        let all = self.shared.as_slice();
        let end = self.off.saturating_add(self.byte_len()).min(all.len());
        &all[self.off.min(all.len())..end]
    }

    /// Payload size of one matrix: `n * (m / block) * block_bytes`.
    fn byte_len(&self) -> usize {
        self.n * self.row_bytes()
    }

    /// The view the K-quant kernels need. Whole-tensor weights reuse the view
    /// built at load; an expert gets a cheap re-slice of the same mapping.
    fn view(&self) -> Result<SharedBytes> {
        if self.off == 0 && self.shared.as_slice().len() == self.byte_len() {
            return Ok(self.shared.clone());
        }
        SharedBytes::from_owner(self.owner.clone(), self.off, self.byte_len())
            .ok_or_else(|| err(&format!("'{}': expert window is out of range", self.name)))
    }

    /// Expert `e` of a stacked tensor whose slices are `stride` bytes apart.
    fn expert(&self, e: usize, stride: usize) -> Self {
        Self {
            off: self.off + e * stride,
            ..self.clone()
        }
    }

    /// Bytes per matrix row.
    fn row_bytes(&self) -> usize {
        (self.m / self.kind.block()) * self.kind.block_bytes()
    }

    /// `out[n] = W x`.
    ///
    /// `x` must already be in whatever basis the weights are stored in — i.e.
    /// Hadamard-transformed exactly when the file is folded. That choice lives
    /// with the caller, because it is a property of the file, not of the format.
    fn gemv(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        if x.len() != self.m || out.len() != self.n {
            return Err(err(&format!(
                "'{}': gemv wants {}x{} but got x={} out={}",
                self.name,
                self.n,
                self.m,
                x.len(),
                out.len()
            )));
        }
        match self.kind {
            QKind::F32 => gemv_bytes_f32(self.bytes(), self.m, self.n, x, out),
            QKind::F16 => gemv_bytes_f16(self.bytes(), self.m, self.n, x, out),
            QKind::Q2_0 => gemv_q2_0(self.bytes(), self.m, self.n, x, out),
            QKind::Q2K | QKind::Q3K => gemv_k_quant(self, x, out),
            kind => {
                // The K-quants go through `oxillama`'s dispatcher. It already
                // carries validated AVX2/AVX-512 kernels for all of them, and
                // driving one implementation is what keeps the packed layouts
                // honest: the same code that KORE's `k_quant` reference was
                // cross-checked against is the code that runs here.
                let qt = QuantTensor::from_shared(self.view()?, vec![self.n, self.m], kind.gguf());
                let kernel = global_dispatcher()
                    .get_kernel(qt.tensor_type)
                    .map_err(|e| err(&format!("'{}': no kernel: {e}", self.name)))?;
                kernel
                    .gemv(&qt, x, out)
                    .map_err(|e| err(&format!("'{}': gemv: {e}", self.name)))
            }
        }
    }

    /// One row, dequantized to f32.
    fn row_f32(&self, index: usize) -> Result<Vec<f32>> {
        let rb = self.row_bytes();
        let start = index
            .checked_mul(rb)
            .ok_or_else(|| err(&format!("'{}': row {index} overflows", self.name)))?;
        let row = self
            .bytes()
            .get(start..start + rb)
            .ok_or_else(|| err(&format!("'{}': row {index} is out of range", self.name)))?;
        crate::dequant::tensor_to_f32(self.kind.type_id(), row, self.m)
            .map_err(|e| err(&format!("'{}': dequantize: {e}", self.name)))
    }
}

/// `out[j] = dot(W row j, x)` for an unpacked f32 matrix.
///
/// Only the unpacked formats use this; they are what a router ships as even
/// inside a K-quant file, and what an unquantised export is made of. The loop
/// is written so the vectoriser can see the reduction.
fn gemv_bytes_f32(w: &[u8], m: usize, n: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
    let need = n * m * 4;
    if w.len() < need {
        return Err(err(&format!("f32 gemv: {} bytes < {need}", w.len())));
    }
    if n * m >= PAR_MIN_ELEMS {
        use rayon::prelude::*;
        // Rows are independent and a row's sum is one sequential chain, so
        // *which* thread computes a row cannot enter its arithmetic. Blocking
        // the rows is a scheduling change, not a reassociation -- which is the
        // only reason this is allowed to touch the router, whose output is a
        // discrete top-k and not a number that could be averaged over.
        let rows_per = n.div_ceil(parallel_row_blocks()).max(8);
        out.par_chunks_mut(rows_per)
            .enumerate()
            .for_each(|(b, o)| f32_gemv_rows(w, m, x, b * rows_per, o));
        return Ok(());
    }
    f32_gemv_rows(w, m, x, 0, out);
    Ok(())
}

/// How many row blocks to cut an unpacked matrix into.
///
/// The region is flat from 4 blocks to 48 (measured in situ on the router:
/// 62.5, 62.6, 67.8, 68.3, 66.4 ms per 8 tokens at 4/8/12/24/48, against 79.9
/// serial), so this is sized to *stop* adding threads once the work per block
/// stops paying for the fork/join and the parked-thread wake-up -- the block
/// count is not where the remaining time is. `KORE_GEMV_BLOCKS` overrides it,
/// which is how the serial path is reached for an A/B (`1`).
fn parallel_row_blocks() -> usize {
    let blocks = std::env::var("KORE_GEMV_BLOCKS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    if blocks > 0 {
        return blocks;
    }
    (rayon::current_num_threads() / 3).clamp(2, 8)
}

/// Below this many weights the unpacked matvecs run serially: a fork/join costs
/// more than the work, and the shapes that reach here are a router (2048x256) or
/// an untied output projection, both far above it.
const PAR_MIN_ELEMS: usize = 1 << 15;

/// `out[j - start] = dot(W row j, x)` for rows `[start, start + out.len())`.
///
/// Several rows at a time. Each row keeps one sequential accumulator, so every
/// row's sum is the same sequence of adds as the naive loop -- what changes is
/// that the lanes' add chains overlap, which is the difference between one MAC
/// per four cycles (f32 add latency) and eight. Eight is past the point where
/// the loop is latency-bound and into memory: measured 246 us at four lanes,
/// 212 us at eight, on a 2048x256 matrix -- and 12 GB/s, i.e. one core's worth
/// of memory bandwidth, which is what the row blocking above is for.
fn f32_gemv_rows(w: &[u8], m: usize, x: &[f32], start: usize, out: &mut [f32]) {
    const R: usize = 8;
    let rb = m * 4;
    let x = &x[..m];
    let end = start + out.len();
    let mut j = start;
    while j + R <= end {
        let mut lanes: [std::slice::ChunksExact<'_, u8>; R] =
            std::array::from_fn(|r| w[(j + r) * rb..(j + r + 1) * rb].chunks_exact(4));
        let mut acc = [0.0f32; R];
        for &xi in x {
            for (r, lane) in lanes.iter_mut().enumerate() {
                let c = lane.next().expect("a row holds exactly m chunks");
                acc[r] += f32::from_le_bytes([c[0], c[1], c[2], c[3]]) * xi;
            }
        }
        out[j - start..j - start + R].copy_from_slice(&acc);
        j += R;
    }
    for (idx, o) in out.iter_mut().enumerate() {
        let row = start + idx;
        if row < j {
            continue;
        }
        let rw = &w[row * rb..(row + 1) * rb];
        let mut acc = 0.0f32;
        for (c, &xi) in rw.chunks_exact(4).zip(x) {
            acc += f32::from_le_bytes([c[0], c[1], c[2], c[3]]) * xi;
        }
        *o = acc;
    }
}

/// `out[j] = dot(W row j, x)` for an unpacked f16 matrix.
fn gemv_bytes_f16(w: &[u8], m: usize, n: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
    let need = n * m * 2;
    if w.len() < need {
        return Err(err(&format!("f16 gemv: {} bytes < {need}", w.len())));
    }
    let x = &x[..m];
    for (j, o) in out.iter_mut().enumerate().take(n) {
        let row = &w[j * m * 2..(j + 1) * m * 2];
        let mut acc = 0.0f32;
        for (c, &xi) in row.chunks_exact(2).zip(x) {
            acc += half::f16::from_le_bytes([c[0], c[1]]).to_f32() * xi;
        }
        *o = acc;
    }
    Ok(())
}

struct DeltaLayer {
    attn_norm: Vec<f32>,
    post_norm: Vec<f32>,
    qkv: QW,            // [hidden, qkv_dim]
    z: QW,              // [hidden, hidden]
    conv: Vec<f32>,     // [conv_k, conv_ch] ne0=conv_k contiguous per channel
    a: Vec<f32>,        // [dt_rank] stored as -exp(A_log)
    alpha_w: Vec<f32>,  // [hidden, dt_rank]
    beta_w: Vec<f32>,   // [hidden, dt_rank]
    dt_bias: Vec<f32>,  // [dt_rank]
    ssm_norm: Vec<f32>, // [head_v_dim]
    out: QW,            // [hidden, hidden]
    ffn: FfnW,
}

struct FullLayer {
    attn_norm: Vec<f32>,
    post_norm: Vec<f32>,
    wq: QW, // [hidden, 2*n_heads*head_dim] interleaved q|gate per head
    wk: QW, // [hidden, n_kv*head_dim]
    wv: QW,
    q_norm: Vec<f32>, // [head_dim]
    k_norm: Vec<f32>,
    wo: QW,
    ffn: FfnW,
}

enum Layer {
    Delta(DeltaLayer),
    Full(FullLayer),
}

/// The feed-forward block.
///
/// `qwen35` has one dense SwiGLU. `qwen35moe` replaces it with a router over
/// `n_experts` stacked experts, of which the top `n_experts_used` are evaluated,
/// plus one always-on shared expert behind a sigmoid gate.
// One per layer, and the dense arm is the common one; boxing either arm
// would put an indirection in the decode loop for nothing.
#[allow(clippy::large_enum_variant)]
enum FfnW {
    Dense { gate: QW, up: QW, down: QW },
    Moe(MoeW),
}

/// A `qwen35moe` FFN.
///
/// The routed experts are each one 3-D tensor -- `[hidden, moe_ffn, n_experts]`
/// for gate/up, `[moe_ffn, hidden, n_experts]` for down -- so expert `e` is a
/// byte window into that same mapping. Nothing is copied, and an expert outside
/// the top-k is never touched at all, which is what lets the same struct work
/// for an all-resident file and for one whose experts stream off a disk.
struct MoeW {
    /// `ffn_gate_inp.weight`, `[hidden, n_experts]`.
    router: QW,
    gate: QW,
    up: QW,
    down: QW,
    /// Bytes between consecutive experts of each stacked tensor.
    stride_gate: usize,
    stride_up: usize,
    stride_down: usize,
    n_experts: usize,
    /// `ffn_{gate,up,down}_shexp.weight`: one dense expert, for every token.
    shexp_gate: QW,
    shexp_up: QW,
    shexp_down: QW,
    /// `ffn_gate_inp_shexp.weight`, `[hidden, 1]`: a sigmoid gate on the above.
    shexp_router: QW,
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

impl AttnState {
    /// The appended keys and values, `[len * kv_dim]` each.
    ///
    /// `k` is post-RoPE and `v` is raw, i.e. exactly what a KV quantizer is
    /// handed. That is why the KV probe (plan §14.22, P6.6a) reads the cache
    /// instead of recomputing the projections and hoping they agree.
    pub fn kv(&self) -> (&[f32], &[f32]) {
        (&self.k, &self.v)
    }

    /// Rows appended so far.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether nothing has been appended yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

pub struct Qwen35Model {
    pub cfg: Qwen35Config,
    emb: QW,
    /// The vocabulary projection. Equals `emb` when the file ties the two.
    head: QW,
    layers: Vec<Layer>,
    out_norm: Vec<f32>,
    /// sign tables: table for width `w` selected by sign_widths order.
    sign_tables: Vec<(usize, Vec<f32>)>,
    /// Whether the weights are stored folded through the Hadamard transform.
    ///
    /// `true` for a file that carries `prism.hadamard.*`: every folded weight is
    /// `W' = H D W`, so every activation has to be rotated into that basis
    /// before it meets one, and the embedding rows have to be rotated back
    /// after lookup. `false` for a file with plain weights, where the transform
    /// does not exist and must not be applied — a 9B K-quant file has no
    /// `prism.hadamard.*` at all, and rotating it would silently turn every
    /// matvec into noise.
    folded: bool,
}

/// Router probabilities: one score per expert, softmax over *all* of them.
///
/// llama.cpp takes the top-k after a full softmax and renormalises afterwards,
/// which is not the same as taking a softmax over the top-k -- so this is shared
/// by both lanes rather than written twice.
fn router_probs(router: &QW, h: &[f32], n_experts: usize) -> Result<Vec<f32>> {
    let mut probs = vec![0.0f32; n_experts];
    router.gemv(h, &mut probs)?;
    softmax_inplace(&mut probs);
    Ok(probs)
}

/// The top `k` experts, renormalised, in descending probability order.
fn top_k_experts(probs: &[f32], k: usize) -> Vec<(usize, f32)> {
    // `total_cmp` so a NaN neither panics nor silently wins the sort.
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    let k = k.min(order.len());
    let sum: f32 = order[..k].iter().map(|&e| probs[e]).sum();
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    order[..k].iter().map(|&e| (e, probs[e] * inv)).collect()
}

/// The router weights in their native `[n, k]` row-major layout.
///
/// The GGUF router is already `[n_experts, hidden]` with each expert's row
/// contiguous; that is exactly what [`DeviceBackend::router_matvec`] wants.
/// Transposing it (as the generic GEMV requires) made every weight read stride
/// a whole expert row and cost more than the rest of the layer.
fn router_rows(w: &QW) -> Result<Vec<f32>> {
    if !matches!(w.kind, QKind::F32) {
        return Err(err(&format!(
            "device routing needs an F32 router, got {:?} ('{}')",
            w.kind, w.name
        )));
    }
    let bytes = w.bytes();
    let (m, n) = (w.m, w.n);
    if bytes.len() != m * n * 4 {
        return Err(err(&format!(
            "'{}': router payload is {} B, not {m}x{n} F32",
            w.name,
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(m * n);
    for i in 0..n {
        for k in 0..m {
            let off = (i * m + k) * 4;
            out.push(f32::from_le_bytes([
                bytes[off],
                bytes[off + 1],
                bytes[off + 2],
                bytes[off + 3],
            ]));
        }
    }
    Ok(out)
}

/// One contiguous device copy of a whole stacked expert bank.
///
/// The repack is the per-expert one [`DeviceBank::acquire`] performs, so the
/// bytes in the big buffer are exactly the bytes a miss would have uploaded.
/// That is what keeps a device-routed run comparable with a streamed one, and
/// it is why this only needs one allocation: thirty thousand small buffers cost
/// far more driver memory than their payload, which is enough to push a bank
/// that fits into system memory and destroy the very locality it was for.
fn upload_bank(
    host: &QW,
    n_experts: usize,
    b: &dyn DeviceBackend,
    device: Device,
) -> Result<(Tensor, usize)> {
    let per = host.byte_len();
    let mut stride = 0usize;
    let mut all = Vec::with_capacity(per.saturating_mul(n_experts));
    for e in 0..n_experts {
        let w = host.expert(e, per);
        let packed: Vec<u8> = match w.kind {
            QKind::Q2_0 => kore_kernels::q2_0::repack_q2_0(w.bytes(), w.m, w.n)?,
            QKind::Q3K => kore_kernels::k_quant::repack_q3_k(w.bytes())?,
            QKind::Q8_0 => kore_kernels::q8_0::repack_q8_0(w.bytes())?,
            _ => w.bytes().to_vec(),
        };
        if e == 0 {
            stride = packed.len();
        } else if packed.len() != stride {
            return Err(err(&format!(
                "'{}': expert {e} repacks to {} B, expert 0 to {stride} B",
                host.name,
                packed.len()
            )));
        }
        all.extend_from_slice(&packed);
    }
    let len = all.len();
    let t = Tensor::from_storage(b.upload_u8(&all, device)?, &[len]);
    b.flush_uploads();
    Ok((t, stride))
}

/// Measurement hook: `KORE_MOE_ROUTE_DUMP=<path>` appends one line of picked
/// expert ids per routed-FFN call, in call order, and is otherwise free.
///
/// The layer index is not written down, because it does not have to be: a token
/// calls the routed FFN exactly once per routed layer, in a fixed order, so the
/// position inside each block of `calls / tokens` lines *is* the layer. What the
/// dump exists for is the union of picks across a hypothetical **chunk** of
/// tokens -- the quantity that decides whether prefill chunks wider than one
/// token would amortise expert traffic -- and that needs the order and the
/// layer, and nothing else.
///
/// The handle is opened once per process. Reopening per call would dominate the
/// step it is measuring, which is the classic way instrumentation lies.
fn dump_route(picks: &[(usize, f32)]) -> Result<()> {
    static PATH: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    static FILE: Mutex<Option<std::fs::File>> = Mutex::new(None);

    let Some(path) = PATH.get_or_init(|| std::env::var("KORE_MOE_ROUTE_DUMP").ok()) else {
        return Ok(());
    };
    let mut guard = FILE.lock().expect("route dump handle");
    if guard.is_none() {
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| err(&format!("route dump '{path}': {e}")))?;
        *guard = Some(f);
    }
    let mut line = String::with_capacity(4 * picks.len());
    for (e, _) in picks {
        line.push_str(&e.to_string());
        line.push(' ');
    }
    line.push('\n');
    guard
        .as_mut()
        .expect("route dump handle just opened")
        .write_all(line.as_bytes())
        .map_err(|e| err(&format!("route dump '{path}': {e}")))
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
        // The metadata prefix *is* the arch key, and it is not the HuggingFace
        // tag: the family ships as `qwen35` (dense FFN) and `qwen35moe` (routed
        // FFN), while HF calls both `qwen3_5*`. Every `qwen35.*` key below is
        // therefore spelled under whichever prefix the file declares.
        let arch = gg.get_str("general.architecture").unwrap_or("");
        let moe = match arch {
            "qwen35" => false,
            "qwen35moe" => true,
            other => {
                return Err(err(&format!(
                    "arch '{other}' is neither qwen35 nor qwen35moe"
                )))
            }
        };

        let get_uk = |s: &str| -> Result<usize> {
            meta.get(&format!("{arch}.{s}"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| err(&format!("missing meta '{arch}.{s}'")))
        };
        let get_fk = |s: &str, d: f64| -> f64 {
            meta.get(&format!("{arch}.{s}"))
                .and_then(|v| v.as_f64())
                .unwrap_or(d)
        };

        // A file with an MTP/nextn head declares it as extra blocks. That head
        // is a speculative drafter, not part of the main stack, and KORE's
        // drafter is the prompt-lookup one, so those blocks are not loaded.
        let nextn = get_uk("nextn_predict_layers").unwrap_or(0);
        let blocks = get_uk("block_count")?;
        let n_layers = blocks.checked_sub(nextn).ok_or_else(|| {
            err(&format!(
                "block_count {blocks} < nextn_predict_layers {nextn}"
            ))
        })?;

        let cfg = Qwen35Config {
            prefix: arch.to_string(),
            n_layers,
            hidden: get_uk("embedding_length")?,
            // A pure-MoE file has no dense feed-forward at all, so the key is
            // simply absent (POCKET-35B is one: every layer routes, and the
            // only dense FFN is the shared expert's). Only the dense path reads
            // this, so zero is the honest value rather than a magic number.
            ffn: match get_uk("feed_forward_length") {
                Ok(v) => v,
                Err(_) if moe => 0,
                Err(e) => return Err(e),
            },
            n_heads: get_uk("attention.head_count")?,
            n_kv: get_uk("attention.head_count_kv")?,
            head_dim: get_uk("attention.key_length")?,
            rope_dim: get_uk("rope.dimension_count").unwrap_or(0),
            theta: get_fk("rope.freq_base", 1e7) as f32,
            // llama.cpp spells this `layer_norm_rms_epsilon`; `_eps` is accepted
            // too because Bonsai's file was written with the short spelling.
            eps: meta
                .get(&format!("{arch}.attention.layer_norm_rms_epsilon"))
                .or_else(|| meta.get(&format!("{arch}.attention.layer_norm_rms_eps")))
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-6) as f32,
            full_attn_interval: get_uk("full_attention_interval")?,
            ssm_conv: get_uk("ssm.conv_kernel")?,
            ssm_state: get_uk("ssm.state_size")?,
            ssm_groups: get_uk("ssm.group_count")?,
            ssm_dt_rank: get_uk("ssm.time_step_rank")?,
            ssm_inner: get_uk("ssm.inner_size")?,
            n_experts: if moe { get_uk("expert_count")? } else { 0 },
            n_experts_used: if moe { get_uk("expert_used_count")? } else { 0 },
            moe_ffn: if moe {
                get_uk("expert_feed_forward_length")?
            } else {
                0
            },
            shexp_ffn: if moe {
                get_uk("expert_shared_feed_forward_length")?
            } else {
                0
            },
            vocab: gg.tensor("token_embd.weight")?.dims[1],
            context: get_uk("context_length").unwrap_or(8192),
            hadamard_block: get_u("prism.hadamard.block_size").unwrap_or(512),
            eos: meta
                .get("tokenizer.ggml.eos_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(248046),
        };

        // Geometry the recurrence depends on. State is per value head and is
        // kept as a `[d_k, d_v]` matrix, so a file whose key and value head
        // widths differ would index the two apart from each other; and the
        // value heads borrow key heads, which only partitions cleanly when one
        // count divides the other. Qwen3.5 ships 128/128 heads and n_v a
        // multiple of n_k, so this is a guard against a foreign file rather
        // than a limitation to work around.
        if cfg.ssm_state * cfg.ssm_dt_rank != cfg.ssm_inner {
            return Err(err(&format!(
                "delta-net head widths disagree: {} key heads x {} state \
                 != {} value dim",
                cfg.ssm_groups, cfg.ssm_state, cfg.ssm_inner
            )));
        }
        if !cfg.ssm_dt_rank.is_multiple_of(cfg.ssm_groups) {
            return Err(err(&format!(
                "{} value heads is not a multiple of {} key heads",
                cfg.ssm_dt_rank, cfg.ssm_groups
            )));
        }

        // Sign tables. Their presence *is* the fold: a file that carries them
        // stores `W' = H D W` on the input axis of every listed weight, and one
        // that does not stores plain weights. Absence is not an error -- it is
        // the other, equally valid layout.
        let folded = meta.contains_key("prism.hadamard.sign_widths");
        let mut sign_tables = Vec::new();
        if folded {
            let widths: Vec<usize> = match meta.get("prism.hadamard.sign_widths") {
                Some(MetaVal::I64Arr(v)) => v.iter().map(|&x| x as usize).collect(),
                _ => return Err(err("prism.hadamard.sign_widths is not an integer array")),
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
            let mut off = 0usize;
            for &w in &widths {
                sign_tables.push((w, values[off..off + w].to_vec()));
                off += w;
            }
        }

        // Any packed matrix the engine can compute, whatever the format.
        let qw = |name: String, gg: &Arc<MiniGguf>| -> Result<QW> {
            let t = gg.tensor(&name)?;
            let kind = QKind::from_type_id(t.type_id).ok_or_else(|| {
                err(&format!(
                    "tensor '{name}' has GGML type {}, which this loader has no \
                     kernel for (supported: F32=0, F16=1, Q4_0=2, Q5_0=6, Q8_0=8, Q2_K=10, \
                     Q3_K=11, Q4_K=12, Q5_K=13, Q6_K=14, Q2_0=42)",
                    t.type_id
                ))
            })?;
            // A 1-D weight is an `[n, 1]` matrix. The GGUF writer trims trailing
            // unit dimensions, so a single-output projection -- the shared-expert
            // router `ffn_gate_inp_shexp` is one -- comes back as `[hidden]`
            // rather than `[hidden, 1]`. Rejecting it would refuse every file
            // that has been through llama-quantize, including the real MoE.
            let (m, n) = match t.dims.as_slice() {
                [m] => (*m, 1),
                [m, n, ..] => (*m, *n),
                _ => return Err(err(&format!("tensor '{name}' has no dimensions"))),
            };
            if m == 0 || n == 0 || m % kind.block() != 0 {
                return Err(err(&format!(
                    "tensor '{name}' has row length {m}, not a multiple of the {} \
                     weights per block of its format",
                    kind.block()
                )));
            }
            // Rows are packed back to back, each a whole number of blocks --
            // the layout every packed kernel assumes.
            let row = (m / kind.block()) * kind.block_bytes();
            let want = n * row;
            // A stacked `[.., .., n_experts]` tensor is `n_experts` copies of
            // that layout, one after another; the view has to span all of them
            // even though only one expert is ever a matrix.
            let n_experts = t.dims.get(2).copied().unwrap_or(1).max(1);
            let total = want
                .checked_mul(n_experts)
                .ok_or_else(|| err(&format!("tensor '{name}': payload size overflows")))?;
            if t.nbytes < total {
                return Err(err(&format!(
                    "tensor '{name}' carries {} bytes but {total} are required \
                     ({n} rows of {m} in {} expert slices)",
                    t.nbytes, n_experts
                )));
            }
            let owner: Arc<dyn ByteOwner> = Arc::new(TensorBytes {
                gg: gg.clone(),
                name: name.clone(),
            });
            let shared = SharedBytes::from_owner(owner.clone(), 0, total)
                .ok_or_else(|| err(&format!("tensor '{name}': {total} bytes out of range")))?;
            Ok(QW {
                name,
                m,
                n,
                kind,
                off: 0,
                shared,
                owner,
            })
        };
        let f32w = |name: &str| -> Result<Vec<f32>> { gg.tensor_f32(name) };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}");
            let attn_norm = f32w(&p("attn_norm.weight"))?;
            let post_norm = f32w(&p("post_attention_norm.weight"))?;

            let ffn = if moe {
                let name = p("ffn_gate_exps.weight");
                let t = gg.tensor(&name)?;
                if t.dims.len() != 3 {
                    return Err(err(&format!(
                        "'{name}' is not a stacked expert tensor \
                         ({} dims, expected 3)",
                        t.dims.len()
                    )));
                }
                let n_experts = t.dims[2];
                if n_experts != cfg.n_experts {
                    return Err(err(&format!(
                        "'{name}' stacks {n_experts} experts but the header \
                         says {}",
                        cfg.n_experts
                    )));
                }
                let router = qw(p("ffn_gate_inp.weight"), &gg)?;
                if router.m != cfg.hidden || router.n != cfg.n_experts {
                    return Err(err(&format!(
                        "router is {}x{} but the model is hidden {} with {} \
                         experts",
                        router.n, router.m, cfg.hidden, cfg.n_experts
                    )));
                }
                let gate = qw(name.clone(), &gg)?;
                let up = qw(p("ffn_up_exps.weight"), &gg)?;
                let down = qw(p("ffn_down_exps.weight"), &gg)?;
                // One expert's payload, which is also the distance to the next
                // one; the three tensors differ because gate/up are
                // `[hidden, moe_ffn]` per expert and down is `[moe_ffn, hidden]`.
                let (sg, su, sd) = (gate.byte_len(), up.byte_len(), down.byte_len());
                let span = sg
                    .checked_mul(n_experts)
                    .ok_or_else(|| err("expert tensor size overflows"))?;
                if t.nbytes < span {
                    return Err(err(&format!(
                        "'{name}' carries {} bytes for {n_experts} experts of \
                         {sg}",
                        t.nbytes
                    )));
                }
                FfnW::Moe(MoeW {
                    router,
                    gate,
                    up,
                    down,
                    stride_gate: sg,
                    stride_up: su,
                    stride_down: sd,
                    n_experts,
                    shexp_gate: qw(p("ffn_gate_shexp.weight"), &gg)?,
                    shexp_up: qw(p("ffn_up_shexp.weight"), &gg)?,
                    shexp_down: qw(p("ffn_down_shexp.weight"), &gg)?,
                    shexp_router: qw(p("ffn_gate_inp_shexp.weight"), &gg)?,
                })
            } else {
                FfnW::Dense {
                    gate: qw(p("ffn_gate.weight"), &gg)?,
                    up: qw(p("ffn_up.weight"), &gg)?,
                    down: qw(p("ffn_down.weight"), &gg)?,
                }
            };

            if (i + 1) % cfg.full_attn_interval == 0 {
                layers.push(Layer::Full(FullLayer {
                    attn_norm,
                    post_norm,
                    wq: qw(p("attn_q.weight"), &gg)?,
                    wk: qw(p("attn_k.weight"), &gg)?,
                    wv: qw(p("attn_v.weight"), &gg)?,
                    q_norm: f32w(&p("attn_q_norm.weight"))?,
                    k_norm: f32w(&p("attn_k_norm.weight"))?,
                    wo: qw(p("attn_output.weight"), &gg)?,
                    ffn,
                }));
            } else {
                layers.push(Layer::Delta(DeltaLayer {
                    attn_norm,
                    post_norm,
                    qkv: qw(p("attn_qkv.weight"), &gg)?,
                    z: qw(p("attn_gate.weight"), &gg)?,
                    conv: f32w(&p("ssm_conv1d.weight"))?,
                    a: f32w(&p("ssm_a"))?,
                    alpha_w: f32w(&p("ssm_alpha.weight"))?,
                    beta_w: f32w(&p("ssm_beta.weight"))?,
                    dt_bias: f32w(&p("ssm_dt.bias"))?,
                    ssm_norm: f32w(&p("ssm_norm.weight"))?,
                    out: qw(p("ssm_out.weight"), &gg)?,
                    ffn,
                }));
            }
        }

        let emb = qw("token_embd.weight".to_string(), &gg)?;
        // Untied head: `output.weight` is the vocabulary projection in its own
        // right, not a copy of the embedding matrix. Ternary Bonsai ties them;
        // the K-quant files do not, and at `[hidden, vocab]` the head is the
        // single largest weight read in a decode step, so silently using the
        // embedding here would produce plausible-looking garbage.
        let head = if gg.tensor("output.weight").is_ok() {
            qw("output.weight".to_string(), &gg)?
        } else {
            emb.clone()
        };
        if head.m != cfg.hidden || head.n != cfg.vocab {
            return Err(err(&format!(
                "head is {}x{} but the model is hidden {} vocab {}",
                head.n, head.m, cfg.hidden, cfg.vocab
            )));
        }
        let out_norm = f32w("output_norm.weight")?;

        Ok(Self {
            cfg,
            emb,
            head,
            layers,
            out_norm,
            sign_tables,
            folded,
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
    ///
    /// A no-op for an unfolded file: there is no transform to invert. Note this
    /// is not the same as `KORE_HAD_OFF`, which is an ablation of a *folded*
    /// model's transform.
    fn apply_t(&self, x: &mut [f32]) -> Result<()> {
        if !self.folded || had_off() {
            return Ok(());
        }
        let signs = self.signs_for(x.len())?;
        hadamard_t(x, signs, self.cfg.hadamard_block);
        Ok(())
    }

    /// H D x in place (folded-matmul input direction, incl. a folded head).
    fn apply_t_inv(&self, x: &mut [f32]) -> Result<()> {
        if !self.folded || had_off() {
            return Ok(());
        }
        let signs = self.signs_for(x.len())?;
        hadamard_t_inv(x, signs, self.cfg.hadamard_block);
        Ok(())
    }

    /// Embedding lookup + inverse-rotation restore.
    fn embed(&self, token: u32) -> Result<Vec<f32>> {
        let mut x = self.emb.row_f32(token as usize)?;
        // inverse-lookup restore: D H x (WHT then signs); identity when unfolded
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

    fn ffn(&self, x: &mut [f32], post_norm: &[f32], w: &FfnW) -> Result<()> {
        let mut h = vec![0.0f32; self.cfg.hidden];
        rms_norm_into(x, post_norm, self.cfg.eps, &mut h);
        self.apply_t_inv(&mut h)?;
        let o = match w {
            FfnW::Dense { gate, up, down } => self.dense_ffn(&h, gate, up, down)?,
            FfnW::Moe(m) => self.moe_ffn(&h, m)?,
        };
        for (xv, ov) in x.iter_mut().zip(&o) {
            *xv += ov;
        }
        Ok(())
    }

    /// SwiGLU: `down( silu(gate h) * (up h) )`.
    fn dense_ffn(&self, h: &[f32], gate: &QW, up: &QW, down: &QW) -> Result<Vec<f32>> {
        let mut g = vec![0.0f32; self.cfg.ffn];
        let mut u = vec![0.0f32; self.cfg.ffn];
        gate.gemv(h, &mut g)?;
        up.gemv(h, &mut u)?;
        let mut act: Vec<f32> = g.iter().zip(&u).map(|(a, b)| silu(*a) * b).collect();
        self.apply_t_inv(&mut act)?;
        let mut o = vec![0.0f32; self.cfg.hidden];
        down.gemv(&act, &mut o)?;
        Ok(o)
    }

    /// `qwen35moe` FFN.
    ///
    /// Routing follows llama.cpp's `build_moe_ffn` with `LLM_FFN_SILU`, softmax
    /// gating and `norm_w = true`: softmax over *all* experts, take the top
    /// `n_experts_used` by probability, then renormalise those to sum to 1. The
    /// shared expert is added afterwards, scaled by
    /// `sigmoid(gate_inp_shexp . h)`; it is not a candidate in the top-k.
    fn moe_ffn(&self, h: &[f32], m: &MoeW) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let ne = m.n_experts;
        let k = cfg.n_experts_used.min(ne);
        if k == 0 {
            return Err(err("qwen35moe: expert_used_count is zero"));
        }

        let probs = router_probs(&m.router, h, ne)?;
        let picks = top_k_experts(&probs, k);
        dump_route(&picks)?;

        let mut out = vec![0.0f32; cfg.hidden];
        let mut g = vec![0.0f32; cfg.moe_ffn];
        let mut u = vec![0.0f32; cfg.moe_ffn];
        let mut act = vec![0.0f32; cfg.moe_ffn];
        let mut d = vec![0.0f32; cfg.hidden];
        for (e, w) in picks {
            m.gate.expert(e, m.stride_gate).gemv(h, &mut g)?;
            m.up.expert(e, m.stride_up).gemv(h, &mut u)?;
            for i in 0..cfg.moe_ffn {
                act[i] = silu(g[i]) * u[i];
            }
            self.apply_t_inv(&mut act)?;
            m.down.expert(e, m.stride_down).gemv(&act, &mut d)?;
            for i in 0..cfg.hidden {
                out[i] += w * d[i];
            }
        }

        // Shared expert: always evaluated, sigmoid-gated by its own router row.
        let mut sg = vec![0.0f32; m.shexp_router.n];
        m.shexp_router.gemv(h, &mut sg)?;
        let gs = 1.0 / (1.0 + (-sg[0]).exp());
        let mut sgate = vec![0.0f32; cfg.shexp_ffn];
        let mut sup = vec![0.0f32; cfg.shexp_ffn];
        m.shexp_gate.gemv(h, &mut sgate)?;
        m.shexp_up.gemv(h, &mut sup)?;
        let mut sact: Vec<f32> = sgate.iter().zip(&sup).map(|(a, b)| silu(*a) * b).collect();
        self.apply_t_inv(&mut sact)?;
        m.shexp_down.gemv(&sact, &mut d)?;
        for i in 0..cfg.hidden {
            out[i] += gs * d[i];
        }
        Ok(out)
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
        l.qkv.gemv(&ht, &mut qkv)?;
        // The output gate is one scalar per value channel, not one per hidden
        // unit: `attn_gate` is `[hidden, n_v * d_v]`. For Bonsai and Ornith
        // those two widths are equal, which is exactly why a `hidden`-sized
        // buffer looked right there and is wrong here.
        let mut z = vec![0.0f32; n_v * d_v];
        l.z.gemv(&ht, &mut z)?;

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
        // `c` indexes three parallel arrays (weight, history, activation), so
        // an iterator over any one of them would not describe the loop.
        #[allow(clippy::needless_range_loop)]
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

        // gated delta rule, one state per *value* head.
        //
        // The q/k projections carry `n_k` heads and v carries `n_v`; when the
        // two differ, each value head borrows a key head by *tiling* the key
        // heads up to the value head count -- value head `hv` reads key head
        // `hv % n_k`. That is what llama.cpp's `ggml_gated_delta_net` does
        // (`iq1 = iv1 % neq1`, and the unfused path's `ggml_repeat`), and it is
        // what the GGUF layout is built for. Bonsai-2B has n_v == n_k, so the
        // mapping is the identity there and its output cannot move.
        let scale = 1.0 / (d_k as f32).sqrt();
        let mut out = vec![0.0f32; n_v * d_v];
        for hv in 0..n_v {
            let s = &mut st.state[hv * d_v * d_v..(hv + 1) * d_v * d_v];
            let hk = hv % n_k;
            let ki = &k[hk * d_k..(hk + 1) * d_k];
            let qi = &q[hk * d_k..(hk + 1) * d_k];
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
        l.out.gemv(&o, &mut proj)?;
        for (xv, pv) in x.iter_mut().zip(&proj) {
            *xv += pv;
        }
        Ok(())
    }

    fn full_attn_step(
        &self,
        l: &FullLayer,
        st: &mut AttnState,
        x: &mut [f32],
        pos: usize,
    ) -> Result<()> {
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
        l.wq.gemv(&h, &mut qg)?;
        let mut kk = vec![0.0f32; kv_dim];
        l.wk.gemv(&h, &mut kk)?;
        let mut vv = vec![0.0f32; kv_dim];
        l.wv.gemv(&h, &mut vv)?;

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
            // `p` walks the cache; it indexes K and is also the score's own
            // position, which no single iterator describes.
            #[allow(clippy::needless_range_loop)]
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
                #[allow(clippy::needless_range_loop)]
                for p in 0..st.len {
                    acc += scores[p] * st.v[p * kv_dim + kvh * hd + j];
                }
                out[i * hd + j] = acc * sigmoid(g[i * hd + j]);
            }
        }

        self.apply_t_inv(&mut out)?;
        let mut proj = vec![0.0f32; hidden];
        l.wo.gemv(&out, &mut proj)?;
        for (xv, pv) in x.iter_mut().zip(&proj) {
            *xv += pv;
        }
        Ok(())
    }

    /// Logits for a hidden state (vocabulary projection, inverse transform).
    pub fn logits(&self, x: &[f32]) -> Result<Vec<f32>> {
        let mut h = vec![0.0f32; self.cfg.hidden];
        rms_norm_into(x, &self.out_norm, self.cfg.eps, &mut h);
        // Folded files store the head rotated like every other folded weight, so
        // its input needs the same rotation. An unfolded head reads `h` as-is.
        if self.folded {
            self.apply_t_inv(&mut h)?;
        }
        let mut logits = vec![0.0f32; self.cfg.vocab];
        self.head.gemv(&h, &mut logits)?;
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
                    self.ffn(&mut x, &l.post_norm, &l.ffn)?;
                }
                Layer::Full(l) => {
                    self.full_attn_step(l, &mut astate[ai], &mut x, pos)?;
                    ai += 1;
                    self.ffn(&mut x, &l.post_norm, &l.ffn)?;
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
use kore_core::{DType, Device, Storage, Tensor};
use std::sync::Mutex;

struct GpuDelta {
    attn_norm: Tensor,
    post_norm: Tensor,
    conv: Tensor,
    a: Tensor,
    alpha_w: Tensor,
    beta_w: Tensor,
    dt_bias: Tensor,
    ssm_norm: Tensor,
    qkv: GpuW,
    z: GpuW,
    out: GpuW,
    ffn: GpuFfn,
}

struct GpuFull {
    attn_norm: Tensor,
    post_norm: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    wq: GpuW,
    wk: GpuW,
    wv: GpuW,
    wo: GpuW,
    ffn: GpuFfn,
}

// One per layer; see `FfnW`.
#[allow(clippy::large_enum_variant)]
enum GpuLayer {
    Delta(GpuDelta),
    Full(GpuFull),
}

/// Mutable per-layer device state.
// One per layer; see `FfnW`.
#[allow(clippy::large_enum_variant)]
pub enum GpuState {
    Delta {
        hist: Tensor,
        state: Tensor,
    },
    Full {
        kc: Tensor,
        vc: Tensor,
        k_scales: Tensor,
        v_scales: Tensor,
        len: usize,
    },
}

/// The recurrent state a batched pass passed through, one slot per column.
///
/// A verify batch advances `m` positions in one go, and it cannot know until
/// its last projection how many of the tokens it was fed were the ones the
/// model would have chosen. Cutting that batch short needs two different
/// things:
///
/// * The KV cache rewinds with an integer. It is append-only and every read
///   stays inside `[0, len)`, so the bytes above the restored `len` are stale
///   and are simply overwritten by the next append.
/// * The DeltaNet state cannot be rewound at all: `gdn_step` updates it *in
///   place*, and a recurrent state that has consumed a token cannot un-consume
///   it. It has to have been copied.
///
/// So this is a copy of `(state, hist)` after each column of the last batched
/// pass. Slot `r` holds the state after exactly `r + 1` columns, for
/// `r = 0 .. cols - 2`; the state after all `cols` columns is the live one, so
/// it is not duplicated. [`Qwen35Gpu::forward_verify`] fills it and
/// [`Qwen35Gpu::rewind`] consumes it.
pub struct Rewind {
    /// `rows[layer][r]`, in layer order; empty for full-attention layers.
    rows: Vec<Vec<(Tensor, Tensor)>>,
    /// Position of column 0 of the pass these slots came from.
    pos0: usize,
    /// How many columns that pass advanced.
    cols: usize,
}

impl Rewind {
    /// Keep the state as it is after column `row` of a `cols`-column pass.
    #[allow(clippy::too_many_arguments)]
    fn capture(
        &mut self,
        b: &Arc<dyn DeviceBackend>,
        layer: usize,
        row: usize,
        hist: &Tensor,
        state: &Tensor,
        pos0: usize,
        cols: usize,
    ) -> Result<()> {
        self.pos0 = pos0;
        self.cols = cols;
        if row + 1 >= cols {
            // The last column: the live state already *is* the post-state.
            return Ok(());
        }
        let pair = self
            .rows
            .get_mut(layer)
            .and_then(|r| r.get_mut(row))
            .ok_or_else(|| err("rewind arena is too small for this batch"))?;
        b.copy_buffer(&pair.0, state)
            .map_err(|e| err(&format!("rewind state copy: {e}")))?;
        b.copy_buffer(&pair.1, hist)
            .map_err(|e| err(&format!("rewind hist copy: {e}")))?;
        Ok(())
    }
}

/// A device-resident packed weight plus what it takes to compute with it.
#[derive(Clone)]
struct GpuW {
    t: Tensor,
    kind: QKind,
    /// Row length (matvec input width).
    m: usize,
    /// Row count (matvec output width).
    n: usize,
}

/// The feed-forward block on the device: a resident dense SwiGLU, or a routed
/// `qwen35moe` FFN whose experts stream through a slot pool.
// One per layer; a box would add a pointer chase to every FFN for 900 bytes.
#[allow(clippy::large_enum_variant)]
enum GpuFfn {
    /// Gate and up share one packed matrix: rows `0..gate_n` are gate, the
    /// rest are up, so the pair costs a single GEMV.
    Dense { gate_up: GpuW, gate_n: usize, down: GpuW },
    Moe(GpuMoe),
}

/// The device-routed expert tier: one contiguous device buffer per projection,
/// plus the router weights in the layout the device matvec reads.
///
/// Built only for `KORE_MOE_DEVICE_ROUTE`. The streaming banks stay in
/// [`GpuMoe`] for the host-routed path, which remains the default.
struct ResidentMoe {
    gate: Tensor,
    up: Tensor,
    down: Tensor,
    gate_per: usize,
    up_per: usize,
    down_per: usize,
    gate_kind: QKind,
    up_kind: QKind,
    down_kind: QKind,
    /// Routed router rows followed by the shared expert's row, so one matvec
    /// feeds both the top-k and the shared gate.
    router_w: Tensor,
    n_routed: usize,
}

/// The routed FFN on the device.
///
/// Without `KORE_MOE_DEVICE_ROUTE` the router stays on the host: the expert
/// windows have to be named before anything can be uploaded, so a device-side
/// router would only add a readback on top of the one that is already
/// unavoidable. With it, the whole routing decision -- matvec, softmax, top-k,
/// gather -- runs on the device and the step has no per-layer round trip.
///
/// The banks are behind a `Mutex` because `forward` takes `&self`: the slot pool
/// is mutable state that lives with the weights, not with the sequence, and a
/// lock is one uncontended acquire per expert against a ~1 MB upload.
struct GpuMoe {
    router: QW,
    shexp_router: QW,
    gate: Mutex<DeviceBank>,
    up: Mutex<DeviceBank>,
    down: Mutex<DeviceBank>,
    shexp_gate: GpuW,
    shexp_up: GpuW,
    shexp_down: GpuW,
    n_experts: usize,
    n_used: usize,
    /// Gather target for the token's `n_used` expert `gate` matrices.
    ///
    /// Expert columns live in independent LRU slots, so they are not adjacent in
    /// memory and cannot feed one matvec. Collecting them here -- one
    /// device-to-device copy per expert, inside the same batch -- makes them a
    /// single `[n_used * n, blocks]` matrix, and the matvec that follows is one
    /// dispatch at `n_used` times the row count instead of `n_used` dispatches.
    /// The win is dispatch count, which is the host's binding cost: measured on
    /// POCKET the step's `submit` region drops 297 ms -> 88 ms per 8 tokens,
    /// while the queue drain (device time) is unchanged. The number of bytes
    /// moved is identical either way -- this is not a bandwidth optimisation.
    gate_stage: Tensor,
    up_stage: Tensor,
    /// One activation row per expert, split out of the batched SwiGLU output so
    /// each `down` matvec sees exactly its own 512-wide input.
    ///
    /// `down` is deliberately *not* batched: one expert's `down` is already 2048
    /// rows, which is past the point where the kernel reaches its bandwidth
    /// ceiling, so stacking them buys dispatch count and nothing else.
    act_rows: Vec<Tensor>,
    /// Row count / row width of an `moe_ffn` expert matrix.
    ffn_rows: usize,
    ffn_cols: usize,
    /// Bytes of one expert matrix, as uploaded.
    ffn_bytes: usize,
    /// Whether the batched expert path may run (every expert uses a K-quant the
    /// concatenation trick is valid for).
    can_batch: bool,
    /// The routing state a *prefetcher* would have to work from: this layer's
    /// picks on the previous token, plus how well they would have covered this
    /// token's misses.
    ///
    /// Prefetch is the only lever left on the upload region (see the plan), and
    /// its whole value depends on routing locality. This measures that directly
    /// rather than inferring it from the LRU's hit rate, which reflects a
    /// *policy* (recency over the last 32) rather than a *predictor*.
    prefetch: Mutex<PrefetchStats>,
    /// Host-side wall clock accumulated by the routed FFN, split by region.
    ///
    /// Only the *host* half of each region shows up here: a `gemv` costs one
    /// command-buffer append and returns, so `compute_us` is submission cost,
    /// not device cost. `download_us` is the opposite case -- it drains the
    /// queue, so it carries the device time of everything queued before it.
    /// Read together they answer the question this exists for: how much of a
    /// step is host encoding, how much is waiting, and how much is bytes.
    times: Mutex<MoeTimes>,
    /// Device-routed path: the whole expert tier in one buffer per projection
    /// and the router weights on the device. `Some` means routing never leaves
    /// the device.
    res: Option<ResidentMoe>,
}

/// How well the previous token's routing at the same layer predicts this one's.
#[derive(Default, Clone)]
pub struct PrefetchStats {
    /// This layer's picks on the previous token.
    pub prev: Vec<usize>,
    /// Steps observed (one per layer per token).
    pub steps: u64,
    /// Experts this token picked at all, summed over steps.
    pub picks: u64,
    /// Of those, how many were also picked on the previous token.
    pub overlap: u64,
    /// Experts that were *not* resident, i.e. the uploads the region pays for.
    pub misses: u64,
    /// Of those misses, how many the previous token's set would have named, and
    /// so could have been uploaded early instead of on the critical path.
    pub covered: u64,
}

/// Microseconds per region of one routed FFN, summed since construction.
#[derive(Default, Clone, Copy)]
pub struct MoeTimes {
    /// `ht` download to the host. This drains the queue, so it contains every
    /// dispatch the layer queued before the router needed its input.
    pub download_us: u64,
    /// Host router matvec plus top-k, with the device idle.
    pub route_us: u64,
    /// `DeviceBank::acquire` over every expert of every top-k, misses included.
    pub upload_us: u64,
    /// Submission of everything else: the three matvecs, the fused SwiGLU, the
    /// combine. Host encoding only.
    pub compute_us: u64,
    /// Expert bytes actually moved host -> device, misses only.
    ///
    /// `upload_us / upload_bytes` is the number that matters: allocation and
    /// per-call overhead show up as a rate far below the raw memcpy ceiling.
    pub upload_bytes: u64,
}

/// Expert bytes moved into the three banks since construction.
fn expert_bytes_moved(m: &GpuMoe) -> u64 {
    let sum = |b: &Mutex<DeviceBank>| b.lock().map(|b| b.uploaded()).unwrap_or(0);
    sum(&m.gate) + sum(&m.up) + sum(&m.down)
}

/// One expert's device copy, uploaded first on a miss.
fn bank_expert(
    bank: &Mutex<DeviceBank>,
    e: usize,
    b: &dyn DeviceBackend,
    device: Device,
) -> Result<GpuW> {
    let mut bank = bank
        .lock()
        .map_err(|_| err("qwen35moe: an expert bank was poisoned by an earlier panic"))?;
    Ok(bank.acquire(e, b, device)?.clone())
}

/// A stacked expert matrix whose device copies stream through a small slot pool.
///
/// The host side is the file's own mapping: `host.expert(e, ..)` is a byte
/// window, so a miss costs exactly one upload of the window that was asked for,
/// and an expert outside the top-k is never touched at all. Replacement is
/// [`ExpertCache`]'s LRU -- the policy is shared with the CPU-side prototype in
/// `moe.rs` rather than reimplemented here.
///
/// Streaming is a *residency* concern only. A slot is fed to the same
/// `gemv_q*` kernel that an all-resident bank would use, over the same bytes, so
/// an offloaded run is bit-identical to a resident one by construction. The
/// interesting failures are therefore all address arithmetic -- which window,
/// which slot, which expert -- and `streamed_experts_match_resident_by_bit`
/// below pins exactly that, evictions included.
///
/// The routed FFN's expert loop is the one consumer: [`GpuMoe`] asks each bank
/// for the experts a token selected, in the order the router ranked them.
struct DeviceBank {
    /// The whole stacked tensor; one expert every `host.byte_len()` bytes.
    host: QW,
    /// One entry per device slot: which expert it holds, and its device copy.
    slots: Vec<Option<(usize, GpuW)>>,
    cache: ExpertCache,
    /// Experts in the stacked tensor (`cache` caps capacity, not this).
    n_experts: usize,
    /// Bytes moved host -> device since construction.
    uploaded: u64,
}

#[allow(dead_code)]
impl DeviceBank {
    /// A bank over `host`, with room for `n_slots` experts on the device.
    ///
    /// The slot count is raised to `n_expert_used` (never above `n_experts`):
    /// one token needs its whole top-k, so a smaller pool would evict an expert
    /// it is about to need again.
    fn new(host: QW, n_experts: usize, n_slots: usize, n_expert_used: usize) -> Result<Self> {
        let block = host.kind.block();
        if host.m == 0 || !host.m.is_multiple_of(block) {
            return Err(err(&format!(
                "'{}': row width {} is not a whole number of {block}-weight blocks",
                host.name, host.m
            )));
        }
        if n_experts == 0 {
            return Err(err(&format!("'{}': expert bank is empty", host.name)));
        }
        let per = host.byte_len();
        let payload = host.owner.owned_bytes().len();
        let end = per
            .checked_mul(n_experts)
            .and_then(|need| need.checked_add(host.off));
        match end {
            Some(end) if end <= payload => {}
            _ => {
                return Err(err(&format!(
                    "'{}': {n_experts} experts at {per} B do not fit the {payload} B payload",
                    host.name
                )))
            }
        }
        let n_slots = n_slots.max(n_expert_used).clamp(1, n_experts);
        Ok(Self {
            host,
            slots: (0..n_slots).map(|_| None).collect(),
            cache: ExpertCache::new(n_experts, n_slots),
            n_experts,
            uploaded: 0,
        })
    }

    /// Upload every expert once, so every later [`Self::acquire`] is a hit.
    ///
    /// This is the resident mode: it only makes sense when the whole bank fits
    /// in VRAM, and the caller decides that (the streaming pool stays the
    /// fallback for a card that cannot hold it). The bytes each slot receives
    /// are exactly the bytes a miss would have uploaded, so a resident run is
    /// bit-identical to a streamed one -- only the timing moves.
    fn preload(&mut self, b: &dyn DeviceBackend, device: Device) -> Result<()> {
        self.preload_n(self.n_experts, b, device)
    }

    /// Preload the first `n` experts, leaving the rest to stream.
    ///
    /// `n` smaller than the bank is the honest fit test: a card with room for
    /// some but not all of a routed tier should keep the resident experts it can
    /// hold and stream the tail, rather than spilling the whole thing into
    /// system memory and paying PCIe on every access.
    fn preload_n(&mut self, n: usize, b: &dyn DeviceBackend, device: Device) -> Result<()> {
        for e in 0..n.min(self.n_experts) {
            self.acquire(e, b, device)?;
        }
        Ok(())
    }

    fn n_slots(&self) -> usize {
        self.slots.len()
    }

    fn hits(&self) -> u64 {
        self.cache.hits()
    }

    fn misses(&self) -> u64 {
        self.cache.misses()
    }

    fn uploaded(&self) -> u64 {
        self.uploaded
    }

    /// The slot holding expert `e`, uploading it first if it is not resident.
    ///
    /// The returned `GpuW` is shaped like the host window, so callers hand it to
    /// the same `wgemv` a resident weight uses; nothing about the matvec changes.
    fn acquire(&mut self, e: usize, b: &dyn DeviceBackend, device: Device) -> Result<&GpuW> {
        let slot = match self.cache.acquire(e) {
            CacheOutcome::Hit => self.slot_holding(e).ok_or_else(|| {
                err(&format!(
                    "'{}': expert {e} is resident but owns no slot",
                    self.host.name
                ))
            })?,
            CacheOutcome::Miss { evicted } => {
                let slot = match evicted {
                    Some(v) => self.slot_holding(v).ok_or_else(|| {
                        err(&format!(
                            "'{}': evicted expert {v} owns no slot",
                            self.host.name
                        ))
                    })?,
                    None => self.slots.iter().position(|s| s.is_none()).ok_or_else(|| {
                        err(&format!(
                            "'{}': no free slot for expert {e}",
                            self.host.name
                        ))
                    })?,
                };
                let w = self.host.expert(e, self.host.byte_len());
                // The ternary kernel reads a padded, row-reordered copy. Every
                // other format's payload goes up exactly as the file packs it,
                // so a miss is one upload and no repacking.
                let t = match w.kind {
                    QKind::Q2_0 => {
                        let packed = kore_kernels::q2_0::repack_q2_0(w.bytes(), w.m, w.n)?;
                        let len = packed.len();
                        self.uploaded += len as u64;
                        Tensor::from_storage(b.upload_u8(&packed, device)?, &[len])
                    }
                    QKind::Q3K => {
                        let packed = kore_kernels::k_quant::repack_q3_k(w.bytes())?;
                        let len = packed.len();
                        self.uploaded += len as u64;
                        Tensor::from_storage(b.upload_u8(&packed, device)?, &[len])
                    }
                    QKind::Q8_0 => {
                        let packed = kore_kernels::q8_0::repack_q8_0(w.bytes())?;
                        let len = packed.len();
                        self.uploaded += len as u64;
                        Tensor::from_storage(b.upload_u8(&packed, device)?, &[len])
                    }
                    _ => {
                        let bytes = w.bytes();
                        self.uploaded += bytes.len() as u64;
                        Tensor::from_storage(b.upload_u8(bytes, device)?, &[bytes.len()])
                    }
                };
                self.slots[slot] = Some((
                    e,
                    GpuW {
                        t,
                        kind: w.kind,
                        m: w.m,
                        n: w.n,
                    },
                ));
                slot
            }
        };
        Ok(&self
            .slots
            .get(slot)
            .and_then(|s| s.as_ref())
            .expect("acquire fills the slot it returns")
            .1)
    }

    /// Whether expert `e` is resident and would be a hit rather than a miss.
    fn resident(&self, e: usize) -> bool {
        self.slot_holding(e).is_some()
    }

    fn slot_holding(&self, e: usize) -> Option<usize> {
        self.slots
            .iter()
            .position(|s| matches!(s, Some((held, _)) if *held == e))
    }
}

/// Qwen3.5 with all weights device-resident; single-token decode via the
/// fused Vulkan kernels. Requires `n_k == n_v` and `d_k == d_v` (true for
/// Bonsai-2B) plus head_dim == 256.
pub struct Qwen35Gpu {
    cfg: Qwen35Config,
    device: Device,
    backend: Arc<dyn DeviceBackend>,
    /// The vocabulary projection (== `emb` for a tied file).
    head: GpuW,
    layers: Vec<GpuLayer>,
    out_norm: Tensor,
    signs: Vec<(usize, Tensor)>,
    /// Whether the weights are folded through the Hadamard transform; see
    /// [`Qwen35Model::folded`]. Drives the `had` flag of every fused pre-op and
    /// whether `had_fwd` rotates at all.
    folded: bool,
    /// Stand-in bound where a sign table would go when `folded` is false. Never
    /// read: the fused kernels only touch `signs` when the Hadamard stage runs.
    no_signs: Tensor,
}

impl Qwen35Gpu {
    /// Materialize a device copy of `cpu` on `device`.
    pub fn from_cpu(cpu: &Qwen35Model, device: Device) -> Result<Self> {
        let backend = get_backend(device).ok_or_else(|| err("no backend for device"))?;
        let cfg = cpu.cfg.clone();
        let n_v = cfg.ssm_dt_rank;
        let d_k = cfg.ssm_state;
        let d_v = cfg.ssm_inner / n_v;
        // `gdn_step` reads one key head per value head and stages a single head
        // of k, q and v per workgroup, so the two head dims have to agree with
        // each other and with the workgroup size. Head *counts* are free to
        // differ: the kernel groups them itself.
        if d_k != d_v || d_k != 128 {
            return Err(err(&format!(
                "gpu path requires d_k == d_v == 128, got {d_k} and {d_v}"
            )));
        }
        if cfg.head_dim != 256 {
            return Err(err("gpu path requires head_dim 256"));
        }

        let up_f32 = |v: &[f32]| -> Result<Tensor> {
            Tensor::from_f32(v, &[v.len()])
                .to(device)
                .map_err(|e| err(&format!("upload: {e}")))
        };
        let pack_w = |w: &QW| -> Result<Vec<u8>> {
            // Refuse at load rather than on the first token. A quant with no
            // device kernel is a property of the lane, not of the file, and
            // discovering it mid-generation reads like a corrupt model.
            if matches!(w.kind, QKind::F32 | QKind::F16 | QKind::Q4_0 | QKind::Q5_0) {
                return Err(err(&format!(
                    "the gpu lane has no kernel for {:?} weights (tensor '{}'); run the CPU lane",
                    w.kind, w.name
                )));
            }
            Ok(match w.kind {
                // The ternary kernel reads a padded, reordered copy (four rows
                // per workgroup). The K-quant kernels read the file's own
                // packing, so their payload goes up untouched.
                QKind::Q2_0 => kore_kernels::q2_0::repack_q2_0(w.bytes(), w.m, w.n)?,
                QKind::Q3K => kore_kernels::k_quant::repack_q3_k(w.bytes())?,
                // Q8_0 blocks are 34 bytes, not a whole number of words, so the
                // kernel reads a repacked 36-byte copy (see `kore_kernels::q8_0`).
                QKind::Q8_0 => kore_kernels::q8_0::repack_q8_0(w.bytes())?,
                _ => w.bytes().to_vec(),
            })
        };
        let up_bytes = |bytes: Vec<u8>, name: &str| -> Result<Tensor> {
            let len = bytes.len();
            let s = Storage::from_bytes(DType::U8, len, bytes)?;
            Tensor::from_storage(s, &[len])
                .to(device)
                .map_err(|e| err(&format!("upload '{name}': {e}")))
        };
        let up_w = |w: &QW| -> Result<GpuW> {
            Ok(GpuW {
                t: up_bytes(pack_w(w)?, &w.name)?,
                kind: w.kind,
                m: w.m,
                n: w.n,
            })
        };

        // A routed FFN has no resident device form: its experts are the whole
        // reason for the streaming engine, which is not wired to the device yet.
        // Failing here is the honest outcome -- silently building a dense FFN
        // would run and produce wrong tokens.
        let gpu_ffn = |w: &FfnW| -> Result<GpuFfn> {
            match w {
                FfnW::Dense { gate, up, down } => {
                    if gate.kind != up.kind || gate.m != up.m {
                        return Err(err(&format!(
                            "dense gate/up must share a quant and input width to fuse \
                             (got {:?}/{:?}, {} vs {})",
                            gate.kind, up.kind, gate.m, up.m
                        )));
                    }
                    let gate_n = gate.n;
                    let mut packed = pack_w(gate)?;
                    packed.extend_from_slice(&pack_w(up)?);
                    Ok(GpuFfn::Dense {
                        gate_up: GpuW {
                            t: up_bytes(packed, "gate_up")?,
                            kind: gate.kind,
                            m: gate.m,
                            n: gate.n + up.n,
                        },
                        gate_n,
                        down: up_w(down)?,
                    })
                }
                FfnW::Moe(m) => {
                    // One token needs its whole top-k at once, and slots are
                    // filled lazily, so the cap costs nothing until experts are
                    // actually touched. Measured on the real 14 GB bartowski
                    // file: 32 -> 64 slots cuts uploaded bytes 1.46x for
                    // +6-8% tok/s, and 64 -> 96 changes *six* fetches over a
                    // whole run, so the pool saturates almost immediately and
                    // the misses left are cold experts, not evictions. It is
                    // therefore not what decides the rate: a 2.1x byte cut buys
                    // ~6%, because expert upload is under half the routed step
                    // and the per-layer routing readback is the other half. This
                    // stays at the floor and the VRAM goes to context instead.
                    // `KORE_MOE_SLOTS` overrides it for measurements.
                    // `KORE_MOE_RESIDENT` keeps a bank in VRAM instead of
                    // streaming it: the value is how many experts *per layer*
                    // to upload once at load, or `all`. That is only a win when
                    // the bank fits, which is exactly the case this mode exists
                    // for (a 16 GB card against an 11 GiB routed tier). A count
                    // below `n_experts` keeps what the card can hold and streams
                    // the tail -- spilling the whole bank into system memory is
                    // far worse than streaming, because every access then pays
                    // PCIe. Unset keeps the streaming pool, so an 8 GB card is
                    // unchanged.
                    // `KORE_MOE_DEVICE_ROUTE` moves the whole routing decision
                    // onto the device: router matvec, softmax, top-k and the
                    // expert gather. It needs the routed tier resident in one
                    // buffer per projection, so it is opt-in and mutually
                    // exclusive with the host-routed residency modes.
                    let device_route = std::env::var_os("KORE_MOE_DEVICE_ROUTE").is_some();
                    let resident = if device_route {
                        None
                    } else {
                        std::env::var("KORE_MOE_RESIDENT").ok().and_then(|v| {
                            if v == "all" {
                                Some(m.n_experts)
                            } else {
                                v.parse::<usize>()
                                    .ok()
                                    .filter(|n| *n > 0)
                                    .map(|n| n.min(m.n_experts))
                            }
                        })
                    };
                    let n_slots = if device_route {
                        cfg.n_experts_used.max(1)
                    } else {
                        match resident {
                            Some(n) => n,
                            None => std::env::var("KORE_MOE_SLOTS")
                                .ok()
                                .and_then(|v| v.parse::<usize>().ok())
                                .unwrap_or(cfg.n_experts_used.max(1) * 4),
                        }
                    };
                    let bank = |host: &QW| -> Result<Mutex<DeviceBank>> {
                        let mut bank = DeviceBank::new(
                            host.clone(),
                            m.n_experts,
                            n_slots,
                            cfg.n_experts_used,
                        )?;
                        if let Some(n) = resident {
                            bank.preload_n(n, &*backend, device)?;
                        }
                        Ok(Mutex::new(bank))
                    };
                    let n_used = cfg.n_experts_used.min(m.n_experts);
                    // A zeroed device buffer of `n_used` expert matrices. Wasteful
                    // to write 220 MB of zeros at load, but it is the only way to
                    // get one buffer the batched matvec can read without adding an
                    // allocation call to the backend.
                    let stage = |host: &QW| -> Result<Tensor> {
                        let bytes = host.byte_len().checked_mul(n_used).ok_or_else(|| {
                            err(&format!("'{}': expert stage overflows usize", host.name))
                        })?;
                        let s = Storage::from_bytes(DType::U8, bytes, vec![0u8; bytes])?;
                        Tensor::from_storage(s, &[bytes])
                            .to(device)
                            .map_err(|e| err(&format!("expert stage upload: {e}")))
                    };
                    // The concatenation is only a valid `[n_used * n, blocks]`
                    // matrix when every expert's payload is a whole number of
                    // rows laid end to end -- true for the K-quant families,
                    // whose blocks are uniform. Q2_0's repack pads each expert to
                    // a multiple of four rows, which would only line up by luck.
                    let can_batch = matches!(
                        m.gate.kind,
                        QKind::Q2K | QKind::Q3K | QKind::Q4K | QKind::Q5K | QKind::Q6K
                    ) && n_used > 1;
                    let act_rows = (0..n_used)
                        .map(|_| {
                            Tensor::from_f32(&vec![0.0f32; m.gate.n], &[m.gate.n])
                                .to(device)
                                .map_err(|e| err(&format!("expert act row upload: {e}")))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let res = if device_route {
                        let (gate, gate_per) =
                            upload_bank(&m.gate, m.n_experts, &*backend, device)?;
                        let (up, up_per) = upload_bank(&m.up, m.n_experts, &*backend, device)?;
                        let (down, down_per) =
                            upload_bank(&m.down, m.n_experts, &*backend, device)?;
                        let mut router = router_rows(&m.router)?;
                        let n_routed = m.router.n;
                        if m.shexp_router.m != m.router.m {
                            return Err(err("shared router width differs from the routed router"));
                        }
                        // Append the shared expert's router row: `moe_topk_gate`
                        // reads the top-k from the first `n_routed` and its gate
                        // from the row after them, so the layer's two router
                        // matvecs become one.
                        router.extend(router_rows(&m.shexp_router)?);
                        let router_w =
                            Tensor::from_f32(&router, &[n_routed + m.shexp_router.n, m.router.m])
                                .to(device)
                                .map_err(|e| err(&format!("router upload: {e}")))?;
                        Some(ResidentMoe {
                            gate,
                            up,
                            down,
                            gate_per,
                            up_per,
                            down_per,
                            gate_kind: m.gate.kind,
                            up_kind: m.up.kind,
                            down_kind: m.down.kind,
                            router_w,
                            n_routed,
                        })
                    } else {
                        None
                    };
                    Ok(GpuFfn::Moe(GpuMoe {
                        router: m.router.clone(),
                        shexp_router: m.shexp_router.clone(),
                        gate: bank(&m.gate)?,
                        up: bank(&m.up)?,
                        down: bank(&m.down)?,
                        shexp_gate: up_w(&m.shexp_gate)?,
                        shexp_up: up_w(&m.shexp_up)?,
                        shexp_down: up_w(&m.shexp_down)?,
                        n_experts: m.n_experts,
                        n_used,
                        gate_stage: stage(&m.gate)?,
                        up_stage: stage(&m.up)?,
                        act_rows,
                        ffn_rows: m.gate.n,
                        ffn_cols: m.gate.m,
                        ffn_bytes: m.gate.byte_len(),
                        can_batch,
                        prefetch: Mutex::new(PrefetchStats::default()),
                        times: Mutex::new(MoeTimes::default()),
                        res,
                    }))
                }
            }
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
                    qkv: up_w(&l.qkv)?,
                    z: up_w(&l.z)?,
                    out: up_w(&l.out)?,
                    ffn: gpu_ffn(&l.ffn)?,
                })),
                Layer::Full(l) => layers.push(GpuLayer::Full(GpuFull {
                    attn_norm: up_f32(&l.attn_norm)?,
                    post_norm: up_f32(&l.post_norm)?,
                    q_norm: up_f32(&l.q_norm)?,
                    k_norm: up_f32(&l.k_norm)?,
                    wq: up_w(&l.wq)?,
                    wk: up_w(&l.wk)?,
                    wv: up_w(&l.wv)?,
                    wo: up_w(&l.wo)?,
                    ffn: gpu_ffn(&l.ffn)?,
                })),
            }
        }

        let head = up_w(&cpu.head)?;

        let mut signs = Vec::new();
        for (w, t) in &cpu.sign_tables {
            signs.push((*w, up_f32(t)?));
        }

        Ok(Self {
            folded: cpu.folded,
            // One float is enough: the fused kernels bind this slot on every
            // pre-op but only index it when the Hadamard stage runs.
            no_signs: up_f32(&[0.0])?,
            cfg,
            device,
            backend,
            head,
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
        // An unfolded model's weights are not rotated, so neither is its
        // activation: the readout feeds `ssm_out` directly.
        if !self.folded {
            return Ok(x.clone());
        }
        // The signed Hadamard is per-row and the sign table covers one row, so a
        // multi-column batch reuses the single-row table. Keying on `numel` would
        // demand an `m * cols` table that the model does not have; keying on the
        // row width is identical at `m = 1`.
        let width = *x.shape().dims().last().unwrap_or(&x.numel());
        self.backend
            .hadamard_signed(x, self.signs_t(width)?, 512, 0)
    }

    /// Shared FFN block with the *incoming* residual folded into its post-norm.
    ///
    /// Returns `(x_r, down)` where `x_r = x + res` is the residual stream *before*
    /// the FFN output. The caller carries `down` forward as the pending residual,
    /// so the trailing `x += down` add is absorbed into the next layer's norm
    /// `out = W x` for whichever quant the file used.
    /// `(hits, misses, bytes uploaded)` summed over every expert bank.
    pub fn expert_stats(&self) -> (u64, u64, u64) {
        let (mut hits, mut misses, mut bytes) = (0, 0, 0);
        for layer in &self.layers {
            let ffn = match layer {
                GpuLayer::Delta(l) => &l.ffn,
                GpuLayer::Full(l) => &l.ffn,
            };
            if let GpuFfn::Moe(m) = ffn {
                for bank in [&m.gate, &m.up, &m.down] {
                    let bank = bank.lock().expect("expert bank");
                    hits += bank.hits();
                    misses += bank.misses();
                    bytes += bank.uploaded();
                }
            }
        }
        (hits, misses, bytes)
    }

    /// Host microseconds per region of the routed FFN, summed over layers.
    ///
    /// Populated only when `KORE_MOE_PROFILE` is set: the timers are two
    /// `Instant::now()` calls per region per layer, which is noise against a
    /// 1 MB upload but pointless to pay in a normal run.
    pub fn moe_times(&self) -> MoeTimes {
        let mut total = MoeTimes::default();
        for layer in &self.layers {
            let ffn = match layer {
                GpuLayer::Delta(l) => &l.ffn,
                GpuLayer::Full(l) => &l.ffn,
            };
            if let GpuFfn::Moe(m) = ffn {
                let t = *m.times.lock().expect("moe profile");
                total.download_us += t.download_us;
                total.route_us += t.route_us;
                total.upload_us += t.upload_us;
                total.compute_us += t.compute_us;
                total.upload_bytes += t.upload_bytes;
            }
        }
        total
    }

    /// How well the previous token's routing predicts this one's, over layers.
    ///
    /// `covered / misses` is the fraction of uploads a perfect prefetcher drawn
    /// from the previous token could have moved off the critical path; only the
    /// counters are summed, the per-layer `prev` sets are left alone.
    pub fn prefetch_stats(&self) -> PrefetchStats {
        let mut total = PrefetchStats::default();
        for layer in &self.layers {
            let ffn = match layer {
                GpuLayer::Delta(l) => &l.ffn,
                GpuLayer::Full(l) => &l.ffn,
            };
            if let GpuFfn::Moe(m) = ffn {
                let p = m.prefetch.lock().expect("prefetch stats").clone();
                total.steps += p.steps;
                total.picks += p.picks;
                total.overlap += p.overlap;
                total.misses += p.misses;
                total.covered += p.covered;
            }
        }
        total
    }

    /// Zero the [`MoeTimes`] counters, e.g. after a warmup token.
    ///
    /// The prefetch counters reset with them (they are the same "decode steps
    /// only" statistic), but the per-layer `prev` sets deliberately do **not**:
    /// the last prompt token is the correct predictor for the first generated
    /// one, so clearing it would throw away a real observation.
    pub fn reset_moe_times(&self) {
        for layer in &self.layers {
            let ffn = match layer {
                GpuLayer::Delta(l) => &l.ffn,
                GpuLayer::Full(l) => &l.ffn,
            };
            if let GpuFfn::Moe(m) = ffn {
                *m.times.lock().expect("moe profile") = MoeTimes::default();
                let mut p = m.prefetch.lock().expect("prefetch stats");
                p.steps = 0;
                p.picks = 0;
                p.overlap = 0;
                p.misses = 0;
                p.covered = 0;
            }
        }
    }

    fn wgemv(&self, w: &GpuW, x: &Tensor) -> Result<Tensor> {
        self.wgemv_at(w.kind, &w.t, w.n, w.m, x)
    }

    /// Matvec against an explicitly-sized weight buffer.
    ///
    /// The routed FFN gathers several experts' matrices into one contiguous
    /// buffer, which is a `[k * rows, cols]` weight the `GpuW` it came from does
    /// not describe. Everything else about the dispatch is unchanged.
    fn wgemv_at(&self, kind: QKind, t: &Tensor, n: usize, k: usize, x: &Tensor) -> Result<Tensor> {
        let b = &self.backend;
        match kind {
            QKind::Q2_0 => b.gemv_q2_0(x, t, n, k),
            QKind::Q2K => b.gemv_q2_k(x, t, n, k),
            QKind::Q3K => b.gemv_q3_k(x, t, n, k),
            QKind::Q4K => b.gemv_q4_k(x, t, n, k),
            QKind::Q5K => b.gemv_q5_k(x, t, n, k),
            QKind::Q6K => b.gemv_q6_k(x, t, n, k),
            QKind::Q8_0 => b.gemv_q8_0(x, t, n, k),
            QKind::F32 | QKind::F16 => Err(err(
                "unpacked F32/F16 weights have no device kernel; run the CPU lane",
            )),
            QKind::Q4_0 | QKind::Q5_0 => Err(err(
                "Q4_0/Q5_0 weights have no device kernel yet; run the CPU lane",
            )),
        }
    }

    /// A matvec against expert `e` of a stacked bank, without copying it out.
    ///
    /// Only the Q3_K down projection needs this today: the routed gate/up pair
    /// is batched into one stage, but `down` has a different activation row per
    /// expert, so it stays one matvec per expert and reads straight from the
    /// gathered block.
    fn wgemv_at_off(
        &self,
        kind: QKind,
        t: &Tensor,
        n: usize,
        k: usize,
        x: &Tensor,
        byte_off: usize,
    ) -> Result<Tensor> {
        match kind {
            QKind::Q2_0 => self.backend.gemv_q2_0_at(x, t, n, k, byte_off / 4),
            QKind::Q2K => self.backend.gemv_q2_k_at(x, t, n, k, byte_off / 4),
            QKind::Q3K => self.backend.gemv_q3_k_at(x, t, n, k, byte_off / 4),
            _ => Err(err(&format!(
                "a stacked-bank offset is only wired for the Q2_0/Q2_K/Q3_K projections; got {kind:?}"
            ))),
        }
    }

    /// One expert-strided matvec over a stacked bank: every selected expert's
    /// `down` projection in a single dispatch.
    ///
    /// `x` is `[n_experts, k]`, one activation row per expert; output row `r`
    /// reads expert `r / rows_per_expert`. `expert_words` is the per-expert
    /// size of the bank in u32 words.
    fn wgemv_id(
        &self,
        kind: QKind,
        t: &Tensor,
        n: usize,
        k: usize,
        x: &Tensor,
        expert_words: usize,
        rows_per_expert: usize,
        idx: &Tensor,
        idx_off: usize,
        x_stride: usize,
    ) -> Result<Tensor> {
        match kind {
            QKind::Q2K => self.backend.gemv_q2_k_id(
                x,
                t,
                n,
                k,
                expert_words,
                rows_per_expert,
                idx,
                idx_off,
                x_stride,
            ),
            QKind::Q3K => self.backend.gemv_q3_k_id(
                x,
                t,
                n,
                k,
                expert_words,
                rows_per_expert,
                idx,
                idx_off,
                x_stride,
            ),
            _ => Err(err(&format!(
                "expert-strided dispatch is only wired for the Q2_K/Q3_K down projection; got {kind:?}"
            ))),
        }
    }

    /// The sign table for `width`, or the placeholder when nothing is folded.
    ///
    /// An unfolded model has no sign tables at all -- `signs_t` would fail, and
    /// the fused kernels must still be handed *something* for a binding they
    /// will not read.
    fn signs_or_placeholder(&self, width: usize) -> Result<&Tensor> {
        if self.folded {
            self.signs_t(width)
        } else {
            Ok(&self.no_signs)
        }
    }

    /// instead of costing its own dispatch.
    #[allow(clippy::too_many_arguments)]
    fn ffn_gpu(
        &self,
        x: &Tensor,
        res: &Tensor,
        post_norm: &Tensor,
        w: &GpuFfn,
        ffn: usize,
        hidden: usize,
    ) -> Result<(Tensor, Tensor)> {
        let b = &self.backend;
        // Fused residual-add + RMSNorm [+ Hadamard]: one dispatch instead of
        // two. The sign table is one row wide (the batch shares it across rows).
        let (x_r, _h, ht) = b.had_pre(
            0,
            self.folded,
            x,
            res,
            post_norm,
            self.signs_or_placeholder(hidden)?,
            self.cfg.eps,
        )?;
        let down = match w {
            GpuFfn::Dense { gate_up, gate_n, down } => {
                let n_total = gate_up.n;
                let signs_ffn = self.signs_or_placeholder(ffn)?;
                if std::env::var_os("KORE_DENSE_SPLIT_GATE_UP").is_some()
                    && gate_up.kind == QKind::Q2_0
                {
                    let row_words = (gate_up.m / 64) * 18 / 4;
                    let g =
                        self.wgemv_at_off(QKind::Q2_0, &gate_up.t, *gate_n, gate_up.m, &ht, 0)?;
                    let u = self.wgemv_at_off(
                        QKind::Q2_0,
                        &gate_up.t,
                        *gate_n,
                        gate_up.m,
                        &ht,
                        gate_n * row_words * 4,
                    )?;
                    let (_gv, _uv, actt) =
                        b.had_pre(2, self.folded, &g, &u, signs_ffn, signs_ffn, self.cfg.eps)?;
                    self.wgemv(down, &actt)?
                } else {
                    let gu = self.wgemv(gate_up, &ht)?;
                    let dims = gu.shape().dims();
                    let rows = if dims.len() == 2 { dims[0] } else { 1 };
                    // Fused SwiGLU [+ Hadamard] over the two strided halves.
                    let (_gv, _uv, actt) = b.had_pre_at(
                        2,
                        self.folded,
                        &gu,
                        &gu,
                        rows * *gate_n,
                        *gate_n,
                        n_total,
                        *gate_n,
                        signs_ffn,
                        signs_ffn,
                        self.cfg.eps,
                    )?;
                    self.wgemv(down, &actt)?
                }
            }
            GpuFfn::Moe(m) => self.moe_ffn_gpu(&ht, m)?,
        };
        Ok((x_r, down))
    }

    /// The routed FFN: routing on the host, matvecs on the device.
    ///
    /// A step costs one 8 KB readback per layer and no per-expert round trip.
    /// The router weight is never uploaded -- it is a 2 MB F32 matrix per layer,
    /// and its result is wanted on the host anyway.
    ///
    /// One column at a time. Widening the pass is not the traffic multiplication
    /// this comment used to claim: a chunk deduplicates the *union* of its
    /// columns' picks, so per-token traffic falls and never rises. But it falls
    /// only once the chunk is wider than the slot pool -- with the pool at its
    /// floor, a chunk of 16 uploads no fewer experts than the pool already holds,
    /// so it evicts what it just fetched. Past that it is worth ~1.9x of bytes at
    /// m=64, and 2.1x of bytes on this arch is worth ~6% tok/s, because expert
    /// upload is under half the routed step and the per-layer routing readback is
    /// the other half (`kore-tools/moe-route-locality.py` measures the union).
    /// A grouped GEMM and m-wide recurrent state is not a trade for single
    /// digits.
    fn moe_ffn_gpu(&self, ht: &Tensor, m: &GpuMoe) -> Result<Tensor> {
        if let Some(r) = &m.res {
            return self.moe_ffn_gpu_resident(ht, m, r);
        }
        self.moe_ffn_gpu_host(ht, m)
    }

    /// The host-routed lane: download the column, route on the CPU, upload the
    /// picked experts. The default, and the only lane that works when the tier
    /// does not fit in VRAM.
    fn moe_ffn_gpu_host(&self, ht: &Tensor, m: &GpuMoe) -> Result<Tensor> {
        let b = &self.backend;
        let cfg = &self.cfg;
        let dims = ht.shape().dims();
        let rows = if dims.len() == 2 { dims[0] } else { 1 };
        if rows != 1 {
            return Err(err(
                "qwen35moe: the routed FFN runs one column at a time (each column \
                 picks its own experts); prefill chunks on this arch are one token wide",
            ));
        }

        let t_route = Instant::now();
        let h_cpu = ht
            .to(Device::Cpu)
            .map_err(|e| err(&format!("expert routing download: {e}")))?;
        let download_us = t_route.elapsed().as_micros() as u64;
        let h = h_cpu
            .as_f32_slice()
            .ok_or_else(|| err("expert routing download did not come back as f32"))?;
        let t_router = Instant::now();
        let mut picks = top_k_experts(&router_probs(&m.router, h, m.n_experts)?, m.n_used);
        // Records the *real* routing, before any ablation rewrites it.
        dump_route(&picks)?;
        let route_us = t_router.elapsed().as_micros() as u64;

        // Ablation: pin every slot to one expert so the banks go resident after
        // the first layer and the step measures the routed *compute* path with
        // uploads removed. The output is nonsense; only the timing is read.
        if let Some(forced) = std::env::var("KORE_MOE_FORCE_EXPERT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            for p in &mut picks {
                p.0 = forced.min(m.n_experts.saturating_sub(1));
            }
        }

        if std::env::var_os("KORE_MOE_PROFILE").is_some() {
            // Locality probe. `prev` is last token's pick set at this layer --
            // the only thing a prefetcher can know before the router runs.
            let mut pf = m.prefetch.lock().expect("prefetch stats");
            let prev: Vec<usize> = pf.prev.clone();
            pf.steps += 1;
            pf.picks += picks.len() as u64;
            pf.overlap += picks.iter().filter(|(e, _)| prev.contains(e)).count() as u64;
            let g = m.gate.lock().expect("gate bank");
            for (e, _) in &picks {
                if !g.resident(*e) {
                    pf.misses += 1;
                    if prev.contains(e) {
                        pf.covered += 1;
                    }
                }
            }
            drop(g);
            pf.prev = picks.iter().map(|(e, _)| *e).collect();
        }

        let signs = self.signs_or_placeholder(cfg.moe_ffn)?;
        let mut acc: Option<Tensor> = None;
        let (mut upload_us, mut compute_us) = (0u64, 0u64);
        let bytes_before = expert_bytes_moved(m);

        // Ablation switch: `KORE_MOE_BATCH=0` forces the one-expert-at-a-time
        // path so the two can be timed against the same input.
        let batch = m.can_batch
            && std::env::var("KORE_MOE_BATCH")
                .map(|v| v != "0")
                .unwrap_or(true);
        if batch {
            let t_up = Instant::now();
            let mut gs = Vec::with_capacity(picks.len());
            let mut us = Vec::with_capacity(picks.len());
            let mut ds = Vec::with_capacity(picks.len());
            for (e, _) in &picks {
                gs.push(bank_expert(&m.gate, *e, &**b, self.device)?);
                us.push(bank_expert(&m.up, *e, &**b, self.device)?);
                ds.push(bank_expert(&m.down, *e, &**b, self.device)?);
            }
            upload_us += t_up.elapsed().as_micros() as u64;

            let t_c = Instant::now();
            // Gather: one copy per expert, in the same batch as the dispatch
            // that reads the result.
            for (i, (g, u)) in gs.iter().zip(us.iter()).enumerate() {
                b.copy_buffer_range(&m.gate_stage, i * m.ffn_bytes, &g.t, 0, m.ffn_bytes)?;
                b.copy_buffer_range(&m.up_stage, i * m.ffn_bytes, &u.t, 0, m.ffn_bytes)?;
            }
            let rows = picks.len() * m.ffn_rows;
            let gv = self.wgemv_at(gs[0].kind, &m.gate_stage, rows, m.ffn_cols, ht)?;
            let uv = self.wgemv_at(us[0].kind, &m.up_stage, rows, m.ffn_cols, ht)?;
            // Same elementwise SwiGLU, one row per expert: the result is
            // bit-identical to running it per expert, since nothing crosses a
            // row boundary.
            let gv = gv.reshape(&[picks.len() as isize, m.ffn_rows as isize])?;
            let uv = uv.reshape(&[picks.len() as isize, m.ffn_rows as isize])?;
            // The router weight is a per-expert scalar. SwiGLU folds it in on
            // the way out, one dispatch instead of two; `down` is linear, so
            // this is the same arithmetic as scaling its 2048-wide output.
            let scales = Tensor::from_f32(
                &picks.iter().map(|(_, w)| *w).collect::<Vec<f32>>(),
                &[picks.len()],
            )
            .to(self.device)
            .map_err(|e| err(&format!("expert weight upload: {e}")))?;
            let act = b.swiglu_rows(
                &gv,
                &uv,
                picks.len(),
                m.ffn_rows,
                &scales,
                signs,
                self.folded,
                cfg.eps,
            )?;
            // `down` per expert: its input is that expert's 512-wide row.
            for (i, a) in m.act_rows.iter().enumerate().take(picks.len()) {
                b.copy_buffer_range(a, 0, &act, i * m.ffn_rows * 4, m.ffn_rows * 4)?;
            }
            // `down` stays one dispatch per expert, and so does the fold: staging
            // the `down` outputs and summing them in one kernel was measured
            // against this add chain and lost (the extra copies cost more than
            // the seven adds they replace).
            for (i, d) in ds.iter().enumerate().take(picks.len()) {
                let dv = self.wgemv(d, &m.act_rows[i])?;
                acc = Some(match acc {
                    Some(a) => b.add(&a, &dv)?,
                    None => dv,
                });
            }
            compute_us += t_c.elapsed().as_micros() as u64;
        } else {
            for (e, weight) in picks {
                let t_up = Instant::now();
                let gate = bank_expert(&m.gate, e, &**b, self.device)?;
                let up = bank_expert(&m.up, e, &**b, self.device)?;
                let down = bank_expert(&m.down, e, &**b, self.device)?;
                upload_us += t_up.elapsed().as_micros() as u64;

                let t_c = Instant::now();
                let gv = self.wgemv(&gate, ht)?;
                let uv = self.wgemv(&up, ht)?;
                // `down` is linear, so scaling the activation is the same arithmetic
                // as scaling its 2048-wide output, for a quarter of the dispatch.
                let act = b.had_pre(2, self.folded, &gv, &uv, signs, signs, cfg.eps)?.2;
                let act = b.mul_scalar(&act, weight)?;
                let dv = self.wgemv(&down, &act)?;
                acc = Some(match acc {
                    Some(a) => b.add(&a, &dv)?,
                    None => dv,
                });
                compute_us += t_c.elapsed().as_micros() as u64;
            }
        }

        let t_c = Instant::now();
        // Shared expert: always evaluated, gated by `sigmoid` of its own row.
        let mut sg = vec![0.0f32; m.shexp_router.n];
        m.shexp_router.gemv(h, &mut sg)?;
        let gs = 1.0 / (1.0 + (-sg[0]).exp());
        let sgate = self.wgemv(&m.shexp_gate, ht)?;
        let sup = self.wgemv(&m.shexp_up, ht)?;
        let signs = self.signs_or_placeholder(cfg.shexp_ffn)?;
        let sact = b.had_pre(2, self.folded, &sgate, &sup, signs, signs, cfg.eps)?.2;
        let sact = b.mul_scalar(&sact, gs)?;
        let sd = self.wgemv(&m.shexp_down, &sact)?;
        compute_us += t_c.elapsed().as_micros() as u64;
        let out = match acc {
            Some(a) => b.add(&a, &sd)?,
            None => sd,
        };

        if std::env::var_os("KORE_MOE_PROFILE").is_some() {
            let mut t = m.times.lock().expect("moe profile");
            t.download_us += download_us;
            t.route_us += route_us;
            t.upload_us += upload_us;
            t.compute_us += compute_us;
            t.upload_bytes += expert_bytes_moved(m) - bytes_before;
        }
        Ok(out)
    }

    /// The routed FFN with the routing decision entirely on the device.
    ///
    /// Nothing here reads a tensor back to the host: the router matvec, softmax,
    /// top-k and expert gather are all dispatches in the same command stream as
    /// the matvecs that consume them. That is the difference that matters on a
    /// card where the expert tier fits -- the streaming path's per-layer
    /// readback drains the queue forty times a token.
    fn moe_ffn_gpu_resident(&self, ht: &Tensor, m: &GpuMoe, r: &ResidentMoe) -> Result<Tensor> {
        let b = &self.backend;
        let cfg = &self.cfg;
        let n_used = m.n_used;

        let logits = b.router_matvec(ht, &r.router_w, r.n_routed + 1)?;
        let (idx, weights, gs) = b.moe_topk_gate(&logits, r.n_routed, n_used, r.n_routed)?;

        if std::env::var_os("KORE_MOE_DEBUG_ROUTE").is_some() {
            use std::sync::Once;
            static ONCE: Once = Once::new();
            ONCE.call_once(|| {
                let h_cpu = ht.to(Device::Cpu).expect("debug route download");
                let h = h_cpu.as_f32_slice().expect("debug route f32");
                let picks =
                    top_k_experts(&router_probs(&m.router, h, m.n_experts).unwrap(), n_used);
                eprintln!("host picks:  {picks:?}");
                let idx_cpu = idx.to(Device::Cpu).expect("debug idx");
                let w_cpu = weights.to(Device::Cpu).expect("debug weights");
                eprintln!(
                    "device idx:  {:?}",
                    idx_cpu.as_f32_slice().unwrap_or(&[])
                );
                eprintln!(
                    "device w:    {:?}",
                    w_cpu.as_f32_slice().unwrap_or(&[])
                );
            });
        }

        // The resident bank holds every expert, so the expert-strided GEMV
        // reads the selected experts in place through `idx`: no gather, and
        // the ids never leave the device.
        let rows = n_used * m.ffn_rows;
        let gv = self.wgemv_id(
            r.gate_kind,
            &r.gate,
            rows,
            m.ffn_cols,
            ht,
            r.gate_per / 4,
            m.ffn_rows,
            &idx,
            0,
            0,
        )?;
        let uv = self.wgemv_id(
            r.up_kind,
            &r.up,
            rows,
            m.ffn_cols,
            ht,
            r.up_per / 4,
            m.ffn_rows,
            &idx,
            0,
            0,
        )?;
        let gv = gv.reshape(&[n_used as isize, m.ffn_rows as isize])?;
        let uv = uv.reshape(&[n_used as isize, m.ffn_rows as isize])?;
        let signs = self.signs_or_placeholder(cfg.moe_ffn)?;
        // SwiGLU and the router weight in one dispatch; the weight multiplies
        // the whole 2048-wide row block, and `down` is linear, so scaling the
        // activation is the same arithmetic as scaling its output.
        let act = b.swiglu_rows(
            &gv,
            &uv,
            n_used,
            m.ffn_rows,
            &weights,
            signs,
            self.folded,
            cfg.eps,
        )?;

        // All selected experts' `down` projections in one dispatch, read in
        // place from the bank. `act` is the matching `[n_used, k]` row block
        // and `idx` names each position's expert.
        let down_all = self.wgemv_id(
            r.down_kind,
            &r.down,
            n_used * m.ffn_cols,
            m.ffn_rows,
            &act,
            r.down_per / 4,
            m.ffn_cols,
            &idx,
            0,
            m.ffn_rows / 4,
        )?;
        // Shared expert: always evaluated, gated by `sigmoid` of its own row.
        // It does not depend on the routed experts, so it is computed first and
        // its output folded into the `down` reduction below.
        let sgate = self.wgemv(&m.shexp_gate, ht)?;
        let sup = self.wgemv(&m.shexp_up, ht)?;
        let ssigns = self.signs_or_placeholder(cfg.shexp_ffn)?;
        let sact = b.swiglu_rows(
            &sgate,
            &sup,
            1,
            cfg.shexp_ffn,
            &gs,
            ssigns,
            self.folded,
            cfg.eps,
        )?;
        let sd = self.wgemv(&m.shexp_down, &sact)?;
        let out = b.sum_rows_add(&down_all, n_used, m.ffn_cols, &sd)?;
        if std::env::var_os("KORE_MOE_DEBUG_ROUTE").is_some() {
            use std::sync::atomic::{AtomicBool, Ordering};
            static DONE: AtomicBool = AtomicBool::new(false);
            if !DONE.swap(true, Ordering::Relaxed) {
                let host = self.moe_ffn_gpu_host(ht, m)?;
                let dc = out
                    .to(Device::Cpu)
                    .map_err(|e| err(&format!("debug out: {e}")))?;
                let hc = host
                    .to(Device::Cpu)
                    .map_err(|e| err(&format!("debug host: {e}")))?;
                let da = dc.as_f32_slice().unwrap_or(&[]);
                let ha = hc.as_f32_slice().unwrap_or(&[]);
                let mut maxd = 0.0f32;
                let mut first = 0usize;
                for i in 0..da.len().min(ha.len()) {
                    let d = (da[i] - ha[i]).abs();
                    if d > maxd {
                        maxd = d;
                        first = i;
                    }
                }
                eprintln!(
                    "moe host vs device: n={} max|d|={maxd:e} at {first} (device={:?} host={:?})",
                    da.len().min(ha.len()),
                    da.get(first),
                    ha.get(first)
                );
            }
        }
        Ok(out)
    }

    /// One decode step; returns the next greedy token.
    pub fn forward(
        &self,
        cpu: &Qwen35Model,
        token: u32,
        pos: usize,
        states: &mut [GpuState],
    ) -> Result<u32> {
        let b = &self.backend;
        let cfg = &self.cfg;
        let hidden = cfg.hidden;
        let ffn = cfg.ffn;
        let n_k = cfg.ssm_groups;
        let n_v = cfg.ssm_dt_rank;
        let d_k = cfg.ssm_state;
        let d_v = cfg.ssm_inner / n_v;
        let conv_dim = cfg.ssm_inner + 2 * cfg.ssm_groups * cfg.ssm_state;
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();

        let t_embed = Instant::now();
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
        let embed_us = t_embed.elapsed().as_micros() as f64;

        for (layer, st) in self.layers.iter().zip(states.iter_mut()) {
            match (layer, st) {
                (GpuLayer::Delta(l), GpuState::Delta { hist, state }) => {
                    // Fold the pending residual in, normalize and rotate in one dispatch.
                    let (x_r, h, ht) = b.had_pre(
                        0,
                        self.folded,
                        &x,
                        &pending,
                        &l.attn_norm,
                        self.signs_or_placeholder(hidden)?,
                        cfg.eps,
                    )?;
                    let qkv = self.wgemv(&l.qkv, &ht)?;
                    let z = self.wgemv(&l.z, &ht)?;
                    let (beta, gate) =
                        b.gdn_gates(&h, &l.alpha_w, &l.beta_w, &l.dt_bias, &l.a, hidden, n_v)?;
                    let conv_out = b.gdn_conv(&qkv, &l.conv, hist, conv_dim, cfg.ssm_conv)?;
                    let o = b.gdn_step(
                        &conv_out,
                        &z,
                        state,
                        &beta,
                        &gate,
                        &l.ssm_norm,
                        n_v,
                        n_k,
                        d_k,
                        d_v,
                        cfg.eps,
                    )?;
                    let ot = self.had_fwd(&o)?;
                    let proj = self.wgemv(&l.out, &ot)?;
                    let (x_next, down) =
                        self.ffn_gpu(&x_r, &proj, &l.post_norm, &l.ffn, ffn, hidden)?;
                    x = x_next;
                    pending = down;
                }
                (
                    GpuLayer::Full(l),
                    GpuState::Full {
                        kc,
                        vc,
                        k_scales,
                        v_scales,
                        len,
                    },
                ) => {
                    let (x_r, _h, ht) = b.had_pre(
                        0,
                        self.folded,
                        &x,
                        &pending,
                        &l.attn_norm,
                        self.signs_or_placeholder(hidden)?,
                        cfg.eps,
                    )?;
                    let qg = self.wgemv(&l.wq, &ht)?;
                    let kk = self.wgemv(&l.wk, &ht)?;
                    let vv = self.wgemv(&l.wv, &ht)?;
                    let (q, k, g) = b.qknorm_rope(
                        &qg,
                        &kk,
                        &l.q_norm,
                        &l.k_norm,
                        cfg.n_heads,
                        cfg.n_kv,
                        cfg.head_dim,
                        cfg.rope_dim,
                        pos,
                        cfg.theta,
                        cfg.eps,
                    )?;

                    // ---- Q8 (INT8) KV-cache path ----
                    // Quantize K/V **on the GPU** (no host round-trip, unlike the
                    // previous per-token CPU quantization which cost two device
                    // syncs per full-attention layer) and append to the INT8 cache.
                    // Decode then reads 4x less VRAM than F32.
                    let (k_q8_t, k_scale_new) = b.quantize_q8_rows(&k)?;
                    let (v_q8_t, v_scale_new) = b.quantize_q8_rows(&vv)?;

                    let (kv_q8, vv_q8) = b.kv_cache_append_q8(
                        kc,
                        vc,
                        k_scales,
                        v_scales,
                        &k_q8_t,
                        &v_q8_t,
                        &k_scale_new,
                        &v_scale_new,
                        *len,
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
                            &q,
                            &kv_q8,
                            &vv_q8,
                            k_scales,
                            v_scales,
                            cfg.n_heads,
                            cfg.n_kv,
                            cfg.head_dim,
                            scale,
                            n_splits,
                            seq_k,
                        )?
                    } else {
                        b.fused_attention_decode_gqa_q8(
                            &q,
                            &kv_q8,
                            &vv_q8,
                            k_scales,
                            v_scales,
                            cfg.n_heads,
                            cfg.n_kv,
                            cfg.head_dim,
                            scale,
                            seq_k,
                        )?
                    };
                    // ---- end Q8 path ----------------------------------------------

                    let (_a, _bv, ot) = b.had_pre(
                        3,
                        self.folded,
                        &attn,
                        &g,
                        self.signs_or_placeholder(hidden)?,
                        self.signs_or_placeholder(hidden)?,
                        cfg.eps,
                    )?;
                    let proj = self.wgemv(&l.wo, &ot)?;
                    let (x_next, down) =
                        self.ffn_gpu(&x_r, &proj, &l.post_norm, &l.ffn, ffn, hidden)?;
                    x = x_next;
                    pending = down;
                }
                _ => return Err(err("gpu state/layer mismatch")),
            }
        }

        let (_r, _h, ht) = b.had_pre(
            0,
            self.folded,
            &x,
            &pending,
            &self.out_norm,
            self.signs_or_placeholder(hidden)?,
            cfg.eps,
        )?;
        let logits = self.wgemv(&self.head, &ht)?;
        let t_argmax = Instant::now();
        let layers_us = t_embed.elapsed().as_micros() as f64 - embed_us;
        let toks = b.argmax_last_dim(&logits)?;
        step_profile_note([
            embed_us,
            layers_us,
            t_argmax.elapsed().as_micros() as f64,
        ]);
        Ok(toks[0])
    }

    /// Multi-column decode: advance `tokens.len()` positions in one pass and
    /// return one greedy token per column.
    ///
    /// Everything whose cost is the *weight stream* — both Hadamards, every
    /// projection matvec, the FFN, attention — runs once over all columns, which
    /// amortises the read that dominates a decode step. The two ops whose state
    /// carries across columns (the DeltaNet conv history and recurrent state) run
    /// one column at a time; that is bit-identical to that many single-token
    /// steps by construction, so a verification pass can accept a draft on
    /// exactly the logits a sequential decode would have produced.
    ///
    /// `pos0` is the position of column 0. Columns form a causal chunk: column
    /// `r` attends to every KV row up to and including its own.
    ///
    /// `m` is capped at 4 by the ternary GEMV kernel, whose accumulator keeps
    /// four columns live per workgroup; beyond that the weight matrix is re-read
    /// per group of four instead of being amortised further.
    pub fn forward_batch(
        &self,
        cpu: &Qwen35Model,
        tokens: &[u32],
        pos0: usize,
        states: &mut [GpuState],
    ) -> Result<Vec<u32>> {
        self.forward_pass(cpu, tokens, pos0, states, Head::Every, None)
    }

    /// Prefill: consume the whole of `prompt` and return the greedy token that
    /// follows it.
    ///
    /// Runs the same pass as [`Self::forward_batch`], chunked at the GEMV's
    /// four-column width, which amortises each weight read over four positions.
    /// The vocabulary projection is skipped for every chunk but the last: the
    /// head is roughly a fifth of a step's weight stream and a prefill reads the
    /// prediction at one position, not at every one. The head does not feed the
    /// recurrent state, so skipping it cannot change what the next step sees.
    ///
    /// State after the call, and the returned token, are what token-at-a-time
    /// decode would have produced -- bit for bit, which is what `verify_m`
    /// checks. This is a pure speed-up over feeding `forward` one token at a time.
    pub fn prefill(
        &self,
        cpu: &Qwen35Model,
        prompt: &[u32],
        pos0: usize,
        states: &mut [GpuState],
    ) -> Result<u32> {
        if prompt.is_empty() {
            return Err(err("prefill needs at least one token"));
        }
        // Each column of a routed FFN picks its own experts, so a four-column
        // chunk would not amortise the expert read the way it amortises a
        // resident weight -- it would multiply the traffic instead.
        let width = if self.cfg.n_experts > 0 {
            1
        } else {
            PREFILL_CHUNK
        };
        let n_chunks = prompt.len().div_ceil(width);
        let mut out = Vec::new();
        for (i, chunk) in prompt.chunks(width).enumerate() {
            let head = if i + 1 == n_chunks {
                Head::Last
            } else {
                Head::Skip
            };
            out = self.forward_pass(cpu, chunk, pos0 + i * width, states, head, None)?;
        }
        out.pop().ok_or_else(|| err("prefill produced no token"))
    }

    /// Allocate the arena a verify pass fills: one `(state, hist)` pair per
    /// DeltaNet layer per column, minus the last one (see [`Rewind`]). At
    /// Bonsai-2B's shapes that is 18 layers x 3 slots x ~1.05 MB ~= 57 MB.
    pub fn new_rewind(&self) -> Result<Rewind> {
        let cfg = &self.cfg;
        let conv_dim = cfg.ssm_inner + 2 * cfg.ssm_groups * cfg.ssm_state;
        let mut rows = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let mut per_row = Vec::new();
            if matches!(layer, GpuLayer::Delta(_)) {
                for _ in 0..MAX_BATCH_COLS - 1 {
                    let state = Tensor::zeros(
                        &[cfg.ssm_dt_rank * cfg.ssm_state * cfg.ssm_state],
                        DType::F32,
                    )
                    .to(self.device)
                    .map_err(|e| err(&format!("rewind state: {e}")))?;
                    let hist = Tensor::zeros(&[conv_dim * (cfg.ssm_conv - 1)], DType::F32)
                        .to(self.device)
                        .map_err(|e| err(&format!("rewind hist: {e}")))?;
                    per_row.push((state, hist));
                }
            }
            rows.push(per_row);
        }
        Ok(Rewind {
            rows,
            pos0: 0,
            cols: 0,
        })
    }

    /// A verify pass: exactly [`Self::forward_batch`], but with each column's
    /// recurrent state copied aside so [`Self::rewind`] can cut the batch short.
    ///
    /// The extra copies are the whole cost of the feature -- the arithmetic is
    /// bit-identical to a plain batch, which is what makes it safe to accept a
    /// draft: every accepted column ran the same reduction the sequential path
    /// would have run.
    pub fn forward_verify(
        &self,
        cpu: &Qwen35Model,
        tokens: &[u32],
        pos0: usize,
        states: &mut [GpuState],
        rewind: &mut Rewind,
    ) -> Result<Vec<u32>> {
        if rewind.rows.len() != self.layers.len() {
            return Err(err("rewind arena was built for a different model"));
        }
        self.forward_pass(cpu, tokens, pos0, states, Head::Every, Some(rewind))
    }

    /// Unwind the last verify pass so only its first `keep` columns stand.
    ///
    /// `keep` is a column count: `1` undoes the whole batch and `cols` is a
    /// no-op. Full-attention layers are truncated by setting `len` back (their
    /// cache is append-only, so the stale bytes above it are simply overwritten
    /// by the next append); DeltaNet layers are restored from the copies
    /// [`Self::forward_verify`] took.
    pub fn rewind(&self, states: &mut [GpuState], arena: &Rewind, keep: usize) -> Result<()> {
        if arena.cols == 0 || keep == 0 || keep > arena.cols {
            return Err(err(&format!(
                "rewind: keep {keep} out of 1..={}",
                arena.cols
            )));
        }
        if keep == arena.cols {
            return Ok(());
        }
        let b = &self.backend;
        for (i, st) in states.iter_mut().enumerate() {
            match st {
                GpuState::Delta { hist, state } => {
                    let pair = arena
                        .rows
                        .get(i)
                        .and_then(|r| r.get(keep - 1))
                        .ok_or_else(|| err("rewind arena is too small for this keep"))?;
                    b.copy_buffer(state, &pair.0)
                        .map_err(|e| err(&format!("rewind state copy: {e}")))?;
                    b.copy_buffer(hist, &pair.1)
                        .map_err(|e| err(&format!("rewind hist copy: {e}")))?;
                }
                GpuState::Full { len, .. } => *len = arena.pos0 + keep,
            }
        }
        Ok(())
    }

    /// Shared core of [`Self::forward_batch`] and [`Self::prefill`]. `head` says
    /// how much of the vocabulary projection the pass has to produce.
    fn forward_pass(
        &self,
        cpu: &Qwen35Model,
        tokens: &[u32],
        pos0: usize,
        states: &mut [GpuState],
        head: Head,
        mut rewind: Option<&mut Rewind>,
    ) -> Result<Vec<u32>> {
        let b = &self.backend;
        let cfg = &self.cfg;
        let m = tokens.len();
        if m == 0 {
            return Ok(Vec::new());
        }
        if m > MAX_BATCH_COLS {
            return Err(err(&format!(
                "a pass supports at most {MAX_BATCH_COLS} columns (gemv_q2_0 accumulator width)"
            )));
        }
        let hidden = cfg.hidden;
        let ffn = cfg.ffn;
        let n_k = cfg.ssm_groups;
        let n_v = cfg.ssm_dt_rank;
        let d_k = cfg.ssm_state;
        let d_v = cfg.ssm_inner / n_v;
        let conv_dim = cfg.ssm_inner + 2 * cfg.ssm_groups * cfg.ssm_state;
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();

        // Embed every column on the CPU (row dequant + restore), upload as one block.
        let mut e = Vec::with_capacity(m * hidden);
        for &t in tokens {
            e.extend_from_slice(&cpu.embed(t)?);
        }
        let x0 = Tensor::from_f32(&e, &[m, hidden])
            .to(self.device)
            .map_err(|e2| err(&format!("embed upload: {e2}")))?;

        // `x + pending` with a zero pending keeps every block on the same fused
        // add+rms_norm path (`x + 0` is exact), exactly as the single-token path.
        let mut x = x0;
        let mut pending = Tensor::from_f32(&vec![0.0f32; m * hidden], &[m, hidden])
            .to(self.device)
            .map_err(|e2| err(&format!("zero residual upload: {e2}")))?;

        // Staging for the per-column DeltaNet readouts. Every row is rewritten on
        // every Delta layer, so a single allocation serves the whole pass.
        let o = Tensor::zeros(&[m, cfg.ssm_inner], DType::F32)
            .to(self.device)
            .map_err(|e2| err(&format!("readout staging: {e2}")))?;
        // Same idea for the per-column attention readouts of the full-attention
        // layers: gathered on device so the batched gate/FFN tail can consume them.
        let attn_staging = Tensor::zeros(&[m, cfg.n_heads * cfg.head_dim], DType::F32)
            .to(self.device)
            .map_err(|e2| err(&format!("attention staging: {e2}")))?;

        let signs_h = self.signs_or_placeholder(hidden)?;

        for (li, (layer, st)) in self.layers.iter().zip(states.iter_mut()).enumerate() {
            match (layer, st) {
                (GpuLayer::Delta(l), GpuState::Delta { hist, state }) => {
                    let (x_r, h, ht) =
                        b.had_pre(0, self.folded, &x, &pending, &l.attn_norm, signs_h, cfg.eps)?;
                    let qkv = self.wgemv(&l.qkv, &ht)?;
                    let z = self.wgemv(&l.z, &ht)?;

                    // The recurrent scan: conv history and DeltaNet state carry
                    // row to row, so the columns are strictly sequential here.
                    for r in 0..m {
                        let (beta, gate) = b.gdn_gates(
                            &h.slice_rows(r, 1)?,
                            &l.alpha_w,
                            &l.beta_w,
                            &l.dt_bias,
                            &l.a,
                            hidden,
                            n_v,
                        )?;
                        let conv_out = b.gdn_conv(
                            &qkv.slice_rows(r, 1)?,
                            &l.conv,
                            hist,
                            conv_dim,
                            cfg.ssm_conv,
                        )?;
                        let o_r = b.gdn_step(
                            &conv_out,
                            &z.slice_rows(r, 1)?,
                            state,
                            &beta,
                            &gate,
                            &l.ssm_norm,
                            n_v,
                            n_k,
                            d_k,
                            d_v,
                            cfg.eps,
                        )?;
                        b.scatter_row(&o, r, &o_r)?;
                        // Column `r`'s post-state, kept so a verify batch can be
                        // cut short after it. Free for an ordinary pass.
                        if let Some(rw) = rewind.as_mut() {
                            rw.capture(b, li, r, hist, state, pos0, m)?;
                        }
                    }

                    // Batched tail: the readout Hadamard and the FFN stream the
                    // weight once for all columns.
                    let ot = self.had_fwd(&o)?;
                    let proj = self.wgemv(&l.out, &ot)?;
                    let (x_next, down) =
                        self.ffn_gpu(&x_r, &proj, &l.post_norm, &l.ffn, ffn, hidden)?;
                    x = x_next;
                    pending = down;
                }
                (
                    GpuLayer::Full(l),
                    GpuState::Full {
                        kc,
                        vc,
                        k_scales,
                        v_scales,
                        len,
                    },
                ) => {
                    let (x_r, _h, ht) =
                        b.had_pre(0, self.folded, &x, &pending, &l.attn_norm, signs_h, cfg.eps)?;
                    let qg = self.wgemv(&l.wq, &ht)?;
                    let kk = self.wgemv(&l.wk, &ht)?;
                    let vv = self.wgemv(&l.wv, &ht)?;
                    let (q, k, g) = b.qknorm_rope(
                        &qg,
                        &kk,
                        &l.q_norm,
                        &l.k_norm,
                        cfg.n_heads,
                        cfg.n_kv,
                        cfg.head_dim,
                        cfg.rope_dim,
                        pos0,
                        cfg.theta,
                        cfg.eps,
                    )?;

                    let (k_q8_t, k_scale_new) = b.quantize_q8_rows(&k)?;
                    let (v_q8_t, v_scale_new) = b.quantize_q8_rows(&vv)?;
                    let (kv_q8, vv_q8) = b.kv_cache_append_q8(
                        kc,
                        vc,
                        k_scales,
                        v_scales,
                        &k_q8_t,
                        &v_q8_t,
                        &k_scale_new,
                        &v_scale_new,
                        *len,
                    )?;
                    *len += m;

                    // Attention is the one op the split-KV kernel cannot batch: it
                    // is `seq_q == 1` only. Attending the whole batch in one call
                    // therefore forces the single-group kernel, which is the
                    // latency-bound one sharding exists to replace (2.2x at ctx
                    // 4104). So each column is attended on its own, with the shard
                    // count sequential decode would have used at that position and
                    // `seq_k` narrowed to that position. Row r therefore performs
                    // exactly the reduction `forward` performs at position
                    // pos0 + r -- bit for bit, which is what lets a verify pass
                    // accept a draft. It also reads only up to that row's own
                    // `max_t`, whereas one batched call would read the full `seq_k`
                    // for every row.
                    let attn = if m == 1 {
                        let sk = *len;
                        if attn_can_shard(cfg, sk) {
                            b.fused_attention_decode_gqa_q8_split(
                                &q,
                                &kv_q8,
                                &vv_q8,
                                k_scales,
                                v_scales,
                                cfg.n_heads,
                                cfg.n_kv,
                                cfg.head_dim,
                                scale,
                                attn_shard_count(sk),
                                sk,
                            )?
                        } else {
                            b.fused_attention_decode_gqa_q8(
                                &q,
                                &kv_q8,
                                &vv_q8,
                                k_scales,
                                v_scales,
                                cfg.n_heads,
                                cfg.n_kv,
                                cfg.head_dim,
                                scale,
                                sk,
                            )?
                        }
                    } else {
                        for r in 0..m {
                            let sk = *len - m + r + 1;
                            let qr = q.slice_rows(r, 1)?;
                            let row = if attn_can_shard(cfg, sk) {
                                b.fused_attention_decode_gqa_q8_split(
                                    &qr,
                                    &kv_q8,
                                    &vv_q8,
                                    k_scales,
                                    v_scales,
                                    cfg.n_heads,
                                    cfg.n_kv,
                                    cfg.head_dim,
                                    scale,
                                    attn_shard_count(sk),
                                    sk,
                                )?
                            } else {
                                b.fused_attention_decode_gqa_q8(
                                    &qr,
                                    &kv_q8,
                                    &vv_q8,
                                    k_scales,
                                    v_scales,
                                    cfg.n_heads,
                                    cfg.n_kv,
                                    cfg.head_dim,
                                    scale,
                                    sk,
                                )?
                            };
                            b.scatter_row(&attn_staging, r, &row)?;
                        }
                        attn_staging.clone()
                    };

                    let (_a, _bv, ot) =
                        b.had_pre(3, self.folded, &attn, &g, signs_h, signs_h, cfg.eps)?;
                    let proj = self.wgemv(&l.wo, &ot)?;
                    let (x_next, down) =
                        self.ffn_gpu(&x_r, &proj, &l.post_norm, &l.ffn, ffn, hidden)?;
                    x = x_next;
                    pending = down;
                }
                _ => return Err(err("gpu state/layer mismatch")),
            }
        }

        if matches!(head, Head::Skip) {
            // A prefill chunk that is not the last one only advances state.
            // Running the head here would compute `m` predictions nobody reads.
            return Ok(Vec::new());
        }
        let (_r, _h, ht) = b.had_pre(
            0,
            self.folded,
            &x,
            &pending,
            &self.out_norm,
            signs_h,
            cfg.eps,
        )?;
        // A prefill chunk needs the prediction at its last position only. The
        // head is row-independent, so projecting that row on its own is the same
        // arithmetic as reading it out of the full `[m, vocab]` result.
        let ht = if matches!(head, Head::Last) {
            ht.slice_rows(m - 1, 1)?
        } else {
            ht
        };
        let logits = self.wgemv(&self.head, &ht)?;
        b.argmax_last_dim(&logits)
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
        if !tokens.is_empty() {
            next = self.prefill(cpu, tokens, 0, &mut states)?;
        }
        // The profile describes *decode* steps. Without this the prompt's
        // forwards are divided by the generated tokens, which on a 12-token
        // prompt against 8 new tokens inflates every region by 2.5x -- and
        // misattributes the whole prefill to the first step.
        self.reset_moe_times();
        for k in 0..n_new {
            let tok = next;
            if tok == cpu.cfg.eos {
                break;
            }
            out.push(tok);
            next = self.forward(cpu, tok, tokens.len() + k, &mut states)?;
        }
        step_profile_report("greedy");
        Ok(out)
    }
}

#[cfg(test)]
mod weight_path_tests {
    use super::*;

    /// Where does the router's 3 ms per layer actually go? Time the matvec on
    /// its own, with the same shape POCKET uses. Run with `--nocapture`.
    #[test]
    fn router_gemv_throughput() {
        let (m, n) = (2048usize, 256usize);
        let mut s: u32 = 0x9E37_79B9;
        let mut next = || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / 8_388_608.0 - 1.0
        };
        let x: Vec<f32> = (0..m).map(|_| next()).collect();
        let mut w = Vec::with_capacity(n * m * 4);
        for _ in 0..n * m {
            w.extend_from_slice(&next().to_le_bytes());
        }
        let mut out = vec![0.0f32; n];
        for _ in 0..20 {
            gemv_bytes_f32(&w, m, n, &x, &mut out).expect("warm");
        }
        let reps = 200;
        let mut x = x;
        let t = Instant::now();
        for _ in 0..reps {
            // Perturb one input per rep: without this the optimizer can hoist
            // the whole invariant loop out and report a fictitious rate.
            x[0] += 1e-6;
            gemv_bytes_f32(&w, m, n, &x, &mut out).expect("gemv");
            std::hint::black_box(&out);
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
        println!(
            "router gemv {n}x{m}: {us:.1} us  ({:.2} GB/s, {:.2} GMAC/s)",
            (n * m * 4) as f64 / 1e9 / (us / 1e6),
            (n * m) as f64 / 1e9 / (us / 1e6)
        );
    }

    /// The router matvec is a *discrete* decision input, so the rewrite that
    /// removed its four per-element bounds checks must not have reassociated a
    /// single add. Compare against the naive loop bit for bit, not to a
    /// tolerance: `PartialEq` on f32 is exact here, and every value is finite.
    #[test]
    fn unpacked_f32_gemv_is_bit_identical_to_the_naive_loop() {
        // 256 is a multiple of the four-row lane width and 7 is not, so the
        // remainder loop is compared as well as the interleaved one. 100 and
        // 200 are large enough to go parallel (above `PAR_MIN_ELEMS`) and are
        // not multiples of the row block, so a partial last block is compared
        // too -- which is the only way the row blocking could have reassociated
        // anything.
        for &(m, n) in &[
            (2048usize, 256usize),
            (2048, 7),
            (5, 6),
            (2048, 100),
            (300, 200),
        ] {
            let mut s: u32 = 0x9E37_79B9;
            let mut next = || {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / 8_388_608.0 - 1.0
            };
            let x: Vec<f32> = (0..m).map(|_| next()).collect();
            let xf: Vec<f32> = (0..n * m).map(|_| next()).collect();
            let mut w = Vec::with_capacity(n * m * 4);
            for v in &xf {
                w.extend_from_slice(&v.to_le_bytes());
            }

            let mut got = vec![0.0f32; n];
            gemv_bytes_f32(&w, m, n, &x, &mut got).expect("gemv");

            let mut want = vec![0.0f32; n];
            for (j, o) in want.iter_mut().enumerate() {
                let row = &w[j * m * 4..(j + 1) * m * 4];
                let mut acc = 0.0f32;
                for (i, &xi) in x.iter().enumerate().take(m) {
                    let b = i * 4;
                    acc += f32::from_le_bytes([row[b], row[b + 1], row[b + 2], row[b + 3]]) * xi;
                }
                *o = acc;
            }

            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "row {i}: {a} != {b}");
            }
        }
    }

    /// A [`ByteOwner`] over a buffer the test owns, so a bank can be built over
    /// bytes that did not come out of a mapped file.
    struct OwnedVec(Vec<u8>);

    unsafe impl ByteOwner for OwnedVec {
        fn owned_bytes(&self) -> &[u8] {
            &self.0
        }
    }

    /// `n_experts` stacked Q4_K matrices of `n` rows by `m` columns, with a
    /// deterministic payload that differs at every byte.
    fn stacked_q4k(n_experts: usize, m: usize, n: usize) -> QW {
        let kind = QKind::Q4K;
        let per = n * (m / kind.block()) * kind.block_bytes();
        let mut bytes = vec![0u8; per * n_experts];
        let mut s: u32 = 0x1234_5678;
        for b in bytes.iter_mut() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (s >> 24) as u8;
        }
        // A super-block's scales are random bytes after the loop above, and a
        // garbage `d`/`dmin` pair can dequantize to all zeros -- which would
        // make byte-identity vacuous. Give every block a finite scale.
        for blk in bytes.chunks_mut(k_quant::Q4_K_BLOCK_BYTES) {
            blk[0..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
            blk[2..4].copy_from_slice(&half::f16::from_f32(0.25).to_le_bytes());
        }
        let owner: Arc<dyn ByteOwner> = Arc::new(OwnedVec(bytes));
        let shared =
            SharedBytes::from_owner(owner.clone(), 0, per * n_experts).expect("whole-tensor view");
        QW {
            name: "test.exps".into(),
            m,
            n,
            kind,
            off: 0,
            shared,
            owner,
        }
    }

    /// The P6.3 gate: an evicting slot pool must produce the same bits as an
    /// all-resident bank. Both arms run the same kernel over the same payload,
    /// so anything that differs is address arithmetic -- a wrong expert window,
    /// a slot that was not overwritten, or a cache that lied about residency.
    /// A Q4_K matrix with **one super-block per row** -- the shape a routed
    /// expert's `down` projection has (`ne[0] = expert_ffn`, e.g. 192 for the
    /// 27 B MoE, rounded up to a block here) -- must agree with an f64 dot over
    /// its own bit-exact dequantized rows.
    ///
    /// The committed K-quant fixtures pin *dequantization*, not the dot; and the
    /// large dense model this path was built on has 16 blocks per row, so a
    /// one-block row was never numerically checked. It matters because this is
    /// the shape where a different accumulation grouping (llama.cpp's CPU
    /// `q4_K_8x8` repack, which KORE deliberately does not do) turns into a
    /// token flip: on `qwen35moe-tiny-q4km` the KORE and llama.cpp step-4
    /// logits agree to ~8e-4 and rank a near-tie oppositely. Byte-identity is a
    /// KORE-to-KORE contract; llama.cpp is an oracle for *semantics*.
    #[test]
    fn q4k_one_block_per_row_matches_f64() {
        let (m, n) = (256usize, 8usize);
        let host = stacked_q4k(1, m, n);
        assert_eq!(host.byte_len(), n * k_quant::Q4_K_BLOCK_BYTES);

        let x: Vec<f32> = (0..m).map(|i| ((i % 23) as f32 - 11.0) * 0.017).collect();
        let mut out = vec![0.0f32; n];
        host.gemv(&x, &mut out).expect("cpu gemv");

        for r in 0..n {
            let row = host.row_f32(r).expect("dequant row");
            let want: f64 = row
                .iter()
                .zip(&x)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            assert!(
                want.abs() > 1e-3,
                "row {r} reference is ~0 ({want}); the fixture proves nothing"
            );
            let rel = (f64::from(out[r]) - want).abs() / want.abs();
            assert!(
                rel < 1e-5,
                "row {r}: gemv {} vs f64 {want} (rel {rel:.3e})",
                out[r]
            );
        }
    }

    #[test]
    fn streamed_experts_match_resident_by_bit() {
        let device = Device::Vulkan(0);
        // `new` registers the backend with `kore_core`, so `get_backend` can
        // hand back the same instance the engine would use.
        let Ok(vk) = kore_vulkan::VulkanBackend::new() else {
            println!("no Vulkan backend available; skipping");
            return;
        };
        let backend: Arc<dyn DeviceBackend> = Arc::new(vk);
        assert!(get_backend(device).is_some(), "backend did not register");

        let (n_experts, m, n) = (4usize, 256usize, 8usize);
        let host = stacked_q4k(n_experts, m, n);
        let per = host.byte_len();

        // Reference: every expert resident, uploaded once, exactly as `up_w`.
        let mut resident = Vec::with_capacity(n_experts);
        for e in 0..n_experts {
            let bytes = host.expert(e, per).bytes().to_vec();
            let len = bytes.len();
            let t = Tensor::from_storage(
                Storage::from_bytes(DType::U8, len, bytes).expect("host storage"),
                &[len],
            )
            .to(device)
            .expect("resident upload");
            resident.push(GpuW {
                t,
                kind: QKind::Q4K,
                m,
                n,
            });
        }

        // One row, as every fused caller passes it.
        let x: Vec<f32> = (0..m).map(|i| ((i % 17) as f32 - 8.0) * 0.03125).collect();
        let xt = Tensor::from_f32(&x, &[1, m]).to(device).expect("upload x");
        let run = |w: &GpuW| -> Vec<f32> {
            backend
                .gemv_q4_k(&xt, &w.t, n, m)
                .expect("gemv")
                .to(Device::Cpu)
                .expect("readback")
                .as_f32_slice()
                .expect("f32 output")
                .to_vec()
        };

        let want: Vec<Vec<f32>> = resident.iter().map(|w| run(w)).collect();
        assert!(
            want[0].iter().any(|v| *v != 0.0),
            "reference is all zero: the fixture proves nothing"
        );
        // If the expert windows collapsed onto one slab, the comparison below
        // would pass for the wrong reason.
        for e in 1..n_experts {
            assert_ne!(
                want[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want[e].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "experts 0 and {e} decode identically: the windows are not distinct"
            );
        }

        // One slot, four experts: every acquire after the first must evict, and
        // the repeated expert at the end must come back out of the pool.
        let mut bank = DeviceBank::new(host.clone(), n_experts, 1, 1).expect("bank");
        assert_eq!(bank.n_slots(), 1);
        let order = [2usize, 0, 3, 1, 2, 2];
        for &e in &order {
            let got = run(bank.acquire(e, &*backend, device).expect("acquire"));
            for i in 0..n {
                assert_eq!(
                    got[i].to_bits(),
                    want[e][i].to_bits(),
                    "expert {e} lane {i}: {} != {}",
                    got[i],
                    want[e][i]
                );
            }
        }
        assert_eq!(bank.misses(), 5, "five uploads for six acquisitions");
        assert_eq!(bank.hits(), 1, "the repeat must hit the pool, not the file");
        assert_eq!(bank.uploaded(), (per * 5) as u64);
    }

    /// The P6.7 gate: gathering k experts into one contiguous stage and running
    /// a single `k * n`-row matvec must reproduce k separate `n`-row matvecs bit
    /// for bit.
    ///
    /// This is the assumption the batched routed FFN rests on -- that k stacked
    /// expert matrices *are* one `[k * n, blocks]` matrix, so the row a kernel
    /// reads is the same row it would have read from the expert's own buffer.
    /// The reduction inside a row is untouched by `n`, so a difference here can
    /// only be addressing.
    #[test]
    fn batched_expert_gather_matches_one_at_a_time() {
        let device = Device::Vulkan(0);
        let Ok(vk) = kore_vulkan::VulkanBackend::new() else {
            println!("no Vulkan backend available; skipping");
            return;
        };
        let backend: Arc<dyn DeviceBackend> = Arc::new(vk);
        assert!(get_backend(device).is_some(), "backend did not register");

        let (n_experts, m, n) = (4usize, 256usize, 8usize);
        let host = stacked_q4k(n_experts, m, n);
        let per = host.byte_len();

        let mut resident = Vec::with_capacity(n_experts);
        for e in 0..n_experts {
            let bytes = host.expert(e, per).bytes().to_vec();
            let len = bytes.len();
            let t = Tensor::from_storage(
                Storage::from_bytes(DType::U8, len, bytes).expect("host storage"),
                &[len],
            )
            .to(device)
            .expect("resident upload");
            resident.push(GpuW {
                t,
                kind: QKind::Q4K,
                m,
                n,
            });
        }

        let x: Vec<f32> = (0..m).map(|i| ((i % 17) as f32 - 8.0) * 0.03125).collect();
        let xt = Tensor::from_f32(&x, &[1, m]).to(device).expect("upload x");
        let read = |t: Tensor| -> Vec<f32> {
            t.to(Device::Cpu)
                .expect("readback")
                .as_f32_slice()
                .expect("f32 output")
                .to_vec()
        };

        let want: Vec<Vec<f32>> = resident
            .iter()
            .map(|w| read(backend.gemv_q4_k(&xt, &w.t, n, m).expect("gemv")))
            .collect();
        assert!(want[0].iter().any(|v| *v != 0.0), "fixture is all zero");

        let stage = Tensor::from_storage(
            Storage::from_bytes(DType::U8, per * n_experts, vec![0u8; per * n_experts])
                .expect("stage storage"),
            &[per * n_experts],
        )
        .to(device)
        .expect("stage upload");
        for (e, w) in resident.iter().enumerate() {
            backend
                .copy_buffer_range(&stage, e * per, &w.t, 0, per)
                .expect("gather");
        }

        let got = read(
            backend
                .gemv_q4_k(&xt, &stage, n_experts * n, m)
                .expect("batched gemv"),
        );
        assert_eq!(got.len(), n_experts * n);
        for e in 0..n_experts {
            for i in 0..n {
                assert_eq!(
                    got[e * n + i].to_bits(),
                    want[e][i].to_bits(),
                    "expert {e} lane {i}: batched {} != single {}",
                    got[e * n + i],
                    want[e][i]
                );
            }
        }
    }

    /// Per-row scaling replaces one `mul_scalar` dispatch per expert with one
    /// per layer, so it has to be the same arithmetic, bit for bit.
    #[test]
    fn mul_rows_matches_per_row_scalar_multiply() {
        let device = Device::Vulkan(0);
        let Ok(vk) = kore_vulkan::VulkanBackend::new() else {
            println!("no Vulkan backend available; skipping");
            return;
        };
        let backend: Arc<dyn DeviceBackend> = Arc::new(vk);
        assert!(get_backend(device).is_some(), "backend did not register");

        let (k, cols) = (4usize, 8usize);
        let act: Vec<f32> = (0..k * cols).map(|i| (i as f32 - 7.0) * 0.1875).collect();
        let scales: Vec<f32> = vec![0.5, -1.25, 0.03125, 2.0];

        let at = Tensor::from_f32(&act, &[k, cols])
            .to(device)
            .expect("upload");
        let st = Tensor::from_f32(&scales, &[k]).to(device).expect("upload");
        let scaled = backend
            .mul_rows(&at, &st, k, cols)
            .expect("mul_rows")
            .to(Device::Cpu)
            .expect("readback")
            .as_f32_slice()
            .expect("f32")
            .to_vec();
        for e in 0..k {
            for j in 0..cols {
                let want = act[e * cols + j] * scales[e];
                assert_eq!(
                    scaled[e * cols + j].to_bits(),
                    want.to_bits(),
                    "row {e} col {j}"
                );
            }
        }
    }
}
