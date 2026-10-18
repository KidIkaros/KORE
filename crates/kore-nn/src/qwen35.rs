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

use std::path::Path;
use std::sync::Arc;

use kore_core::{KoreError, Result};

use crate::gguf_min::{MetaVal, MiniGguf};
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

/// The widest batch any pass can feed the ternary GEMV: its accumulator keeps
/// four columns live per workgroup (`grid.y = ceil(m / 4)`).
pub const MAX_BATCH_COLS: usize = 4;

/// Prefill chunk width: the ternary GEMV's four-column accumulator.
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
    /// K-quants, 256 weights per super-block.
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
            _ => k_quant::QK_K,
        }
    }

    /// Bytes per packed block.
    const fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q2_0 => kore_kernels::q2_0::Q2_0_BLOCK_BYTES,
            Self::Q4K => k_quant::Q4_K_BLOCK_BYTES,
            Self::Q5K => k_quant::Q5_K_BLOCK_BYTES,
            Self::Q6K => k_quant::Q6_K_BLOCK_BYTES,
        }
    }

    /// The `oxillama` type tag, which is how the CPU lane names a K-quant kernel.
    ///
    /// Only the K-quant path goes through `oxillama`; Q2_0 and the unpacked
    /// formats have their own loops and never ask for a tag, so their value
    /// here is a placeholder.
    fn gguf(self) -> GgufTensorType {
        match self {
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
        match self.kind {
            QKind::F32 => Ok(row
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            QKind::F16 => Ok(row
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()),
            QKind::Q2_0 => kore_kernels::q2_0::dequantize_q2_0(row, self.m),
            kind => k_quant::dequantize(kind.type_id(), row, self.m),
        }
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
    for (j, o) in out.iter_mut().enumerate().take(n) {
        let row = &w[j * m * 4..(j + 1) * m * 4];
        let mut acc = 0.0f32;
        for (i, &xi) in x.iter().enumerate().take(m) {
            let b = i * 4;
            acc += f32::from_le_bytes([row[b], row[b + 1], row[b + 2], row[b + 3]]) * xi;
        }
        *o = acc;
    }
    Ok(())
}

/// `out[j] = dot(W row j, x)` for an unpacked f16 matrix.
fn gemv_bytes_f16(w: &[u8], m: usize, n: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
    let need = n * m * 2;
    if w.len() < need {
        return Err(err(&format!("f16 gemv: {} bytes < {need}", w.len())));
    }
    for (j, o) in out.iter_mut().enumerate().take(n) {
        let row = &w[j * m * 2..(j + 1) * m * 2];
        let mut acc = 0.0f32;
        for (i, &xi) in x.iter().enumerate().take(m) {
            let b = i * 2;
            acc += half::f16::from_le_bytes([row[b], row[b + 1]]).to_f32() * xi;
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
            ffn: get_uk("feed_forward_length")?,
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
                     kernel for (supported: F32=0, F16=1, Q4_K=12, Q5_K=13, \
                     Q6_K=14, Q2_0=42)",
                    t.type_id
                ))
            })?;
            if t.dims.len() < 2 {
                return Err(err(&format!("tensor '{name}' is not a 2-D weight")));
            }
            let (m, n) = (t.dims[0], t.dims[1]);
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

        let mut probs = vec![0.0f32; ne];
        m.router.gemv(h, &mut probs)?;
        softmax_inplace(&mut probs);

        // Descending by probability. `total_cmp` so a NaN neither panics nor
        // silently wins the sort.
        let mut order: Vec<usize> = (0..ne).collect();
        order.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));

        let sum: f32 = order[..k].iter().map(|&e| probs[e]).sum();
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };

        let mut out = vec![0.0f32; cfg.hidden];
        let mut g = vec![0.0f32; cfg.moe_ffn];
        let mut u = vec![0.0f32; cfg.moe_ffn];
        let mut act = vec![0.0f32; cfg.moe_ffn];
        let mut d = vec![0.0f32; cfg.hidden];
        for &e in &order[..k] {
            let w = probs[e] * inv;
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

/// The dense feed-forward weights, resident on the device.
///
/// The routed `qwen35moe` FFN has no resident form yet: its experts are meant to
/// stream, which is P6.3/P6.4, so this type only ever holds a dense SwiGLU.
struct GpuFfn {
    gate: GpuW,
    up: GpuW,
    down: GpuW,
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
        let up_w = |w: &QW| -> Result<GpuW> {
            let bytes = match w.kind {
                // The ternary kernel reads a padded, reordered copy (four rows
                // per workgroup). The K-quant kernels read the file's own
                // packing, so their payload goes up untouched.
                QKind::Q2_0 => kore_kernels::q2_0::repack_q2_0(w.bytes(), w.m, w.n)?,
                _ => w.bytes().to_vec(),
            };
            let len = bytes.len();
            let s = Storage::from_bytes(DType::U8, len, bytes)?;
            let t = Tensor::from_storage(s, &[len])
                .to(device)
                .map_err(|e| err(&format!("upload '{}': {e}", w.name)))?;
            Ok(GpuW {
                t,
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
                FfnW::Dense { gate, up, down } => Ok(GpuFfn {
                    gate: up_w(gate)?,
                    up: up_w(up)?,
                    down: up_w(down)?,
                }),
                FfnW::Moe(_) => Err(err(
                    "qwen35moe: the device lane does not build the routed FFN yet \
                     (it needs the streaming expert engine); use the CPU lane",
                )),
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
    fn wgemv(&self, w: &GpuW, x: &Tensor) -> Result<Tensor> {
        let b = &self.backend;
        match w.kind {
            QKind::Q2_0 => b.gemv_q2_0(x, &w.t, w.n, w.m),
            QKind::Q4K => b.gemv_q4_k(x, &w.t, w.n, w.m),
            QKind::Q5K => b.gemv_q5_k(x, &w.t, w.n, w.m),
            QKind::Q6K => b.gemv_q6_k(x, &w.t, w.n, w.m),
            QKind::F32 | QKind::F16 => Err(err(
                "unpacked F32/F16 weights have no device kernel; run the CPU lane",
            )),
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
        let gv = self.wgemv(&w.gate, &ht)?;
        let uv = self.wgemv(&w.up, &ht)?;
        // Fused SwiGLU [+ Hadamard] (the second half of the FFN's activation pair).
        let signs_ffn = self.signs_or_placeholder(ffn)?;
        let (_gv, _uv, actt) =
            b.had_pre(2, self.folded, &gv, &uv, signs_ffn, signs_ffn, self.cfg.eps)?;
        let down = self.wgemv(&w.down, &actt)?;
        Ok((x_r, down))
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
        let toks = b.argmax_last_dim(&logits)?;
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
        let n_chunks = prompt.len().div_ceil(PREFILL_CHUNK);
        let mut out = Vec::new();
        for (i, chunk) in prompt.chunks(PREFILL_CHUNK).enumerate() {
            let head = if i + 1 == n_chunks {
                Head::Last
            } else {
                Head::Skip
            };
            out = self.forward_pass(cpu, chunk, pos0 + i * PREFILL_CHUNK, states, head, None)?;
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
