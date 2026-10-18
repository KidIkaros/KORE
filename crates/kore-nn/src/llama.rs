//! GPU-resident Causal LLaMA reference model (`MiniCPM5` / `LLaMA-3` style).
//!
//! Fully resident in GPU VRAM (or unified buffers) using Vulkan compute shaders:
//! - On-chip Small-Batch GEMV for Q/K/V/O projections and SwiGLU MLP
//! - Fused Flash-Decode attention over resident KV-caches
//! - On-GPU RMSNorm and SwiGLU activation
//! - Zero-PCIe decode loop with GPU-side greedy argmax sampling

use kore_attention::KvCacheQuantMode;
use kore_core::{DType, Device, KoreError, Result, Tensor};

use crate::activations::{silu, swiglu};
use crate::embedding::Embedding;
use crate::linear::{Linear, PackedQuant};
use crate::module::Module;
use crate::rms_norm::RMSNorm;
use crate::sampler::SamplerConfig;
use crate::transformer::MtpDraftHead;

/// `KORE_NO_FUSE=1` disables the fused GPU dispatch paths (split3 GEMV,
/// rope/kv pairs, add+rms_norm) for A/B benchmarking.
fn fusion_enabled() -> bool {
    static ONCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ONCE.get_or_init(|| std::env::var("KORE_NO_FUSE").ok().as_deref() != Some("1"))
}

/// Zero-VRAM Prompt Lookup Drafter (`Strata` style).
///
/// Scans the prompt + generated token history for suffix n-gram matches (`n = 3, 2, 1`)
/// and drafts up to `max_draft` continuation tokens in O(N) CPU time with zero GPU cost.
#[derive(Debug, Clone)]
pub struct PromptLookupDrafter {
    pub max_draft: usize,
}

impl Default for PromptLookupDrafter {
    fn default() -> Self {
        Self { max_draft: 3 }
    }
}

impl PromptLookupDrafter {
    pub fn new(max_draft: usize) -> Self {
        Self {
            max_draft: max_draft.clamp(1, 3),
        }
    }

    /// Propose up to `self.max_draft` draft tokens continuing `history`.
    pub fn draft(&self, history: &[usize]) -> Vec<usize> {
        let len = history.len();
        if len < 2 {
            return Vec::new();
        }
        for &ngram in &[3usize, 2, 1] {
            if len <= ngram {
                continue;
            }
            let suffix = &history[len - ngram..];
            let search_end = len - ngram;
            for pos in (0..search_end).rev() {
                if &history[pos..pos + ngram] == suffix {
                    let cont_start = pos + ngram;
                    let cont_end = (cont_start + self.max_draft).min(len);
                    if cont_end > cont_start {
                        return history[cont_start..cont_end].to_vec();
                    }
                }
            }
        }
        Vec::new()
    }
}

/// Architecture configuration for a Causal LLaMA model.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub head_dim: usize,
}

impl LlamaConfig {
    /// MiniCPM5-2B architecture configuration.
    pub fn minicpm5_2b() -> Self {
        Self {
            vocab_size: 130560,
            hidden_size: 2048,
            intermediate_size: 6144,
            num_hidden_layers: 42,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-6,
            rope_theta: 5000000.0,
            head_dim: 128,
        }
    }

    /// Precompute RoPE inverse frequencies: `1 / (theta^(2i / head_dim))`
    pub fn rope_inv_freq(&self) -> Vec<f32> {
        (0..self.head_dim / 2)
            .map(|i| 1.0 / self.rope_theta.powf(2.0 * i as f32 / self.head_dim as f32))
            .collect()
    }
}

/// Pre-allocated GPU / CPU KV cache for a single transformer layer.
///
/// Supports `F32`, `Q8_0`, and Hadamard-rotated `K8V4` quantization modes,
/// multi-token batch appends (`M >= 1`), and $O(1)$ speculative rollback via `truncate`.
pub struct LlamaKvCacheLayer {
    pub k: Tensor,
    pub v: Tensor,
    pub len: usize,
    pub max_seq: usize,
    pub quant_mode: KvCacheQuantMode,
    dim: usize,
    q_keys: Vec<i8>,
    k_scales: Vec<f32>,
    q8_values: Vec<i8>,
    q4_values: Vec<u8>,
    v_scales: Vec<f32>,
}

impl LlamaKvCacheLayer {
    pub fn new(
        max_seq: usize,
        num_kv_heads: usize,
        head_dim: usize,
        device: Device,
    ) -> Result<Self> {
        Self::with_quant_mode(
            max_seq,
            num_kv_heads,
            head_dim,
            device,
            KvCacheQuantMode::F32,
        )
    }

    pub fn with_quant_mode(
        max_seq: usize,
        num_kv_heads: usize,
        head_dim: usize,
        device: Device,
        quant_mode: KvCacheQuantMode,
    ) -> Result<Self> {
        let dim = num_kv_heads * head_dim;
        let k = Tensor::zeros(&[max_seq, dim], DType::F32).to(device)?;
        let v = Tensor::zeros(&[max_seq, dim], DType::F32).to(device)?;
        Ok(Self {
            k,
            v,
            len: 0,
            max_seq,
            quant_mode,
            dim,
            q_keys: Vec::new(),
            k_scales: Vec::new(),
            q8_values: Vec::new(),
            q4_values: Vec::new(),
            v_scales: Vec::new(),
        })
    }

    pub fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        let new_tokens = k_new.shape().dims()[0];
        if self.len + new_tokens > self.max_seq {
            return Err(KoreError::StorageError(format!(
                "KV cache overflow: {} + {} > max_seq {}",
                self.len, new_tokens, self.max_seq
            )));
        }

        // If quantized KV mode (Q8_0 or K8V4) is active, quantize -> dequantize (with Hadamard rotation for K8V4)
        let (k_eff, v_eff) = if self.quant_mode != KvCacheQuantMode::F32 {
            let k_cpu = k_new.to(Device::Cpu)?.contiguous();
            let v_cpu = v_new.to(Device::Cpu)?.contiguous();
            let k_slice = k_cpu.as_f32_slice().unwrap();
            let v_slice = v_cpu.as_f32_slice().unwrap();

            let mut k_deq = Vec::with_capacity(new_tokens * self.dim);
            for row in k_slice.chunks_exact(self.dim) {
                let absmax = row.iter().fold(0.0f32, |m, &val| m.max(val.abs()));
                let scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
                let inv = 1.0 / scale;
                self.k_scales.push(scale);
                for &val in row {
                    let q = (val * inv).round().clamp(-127.0, 127.0) as i8;
                    self.q_keys.push(q);
                    k_deq.push(q as f32 * scale);
                }
            }

            let v_deq_tensor = if self.quant_mode == KvCacheQuantMode::Q8_0 {
                let mut v_deq = Vec::with_capacity(new_tokens * self.dim);
                for row in v_slice.chunks_exact(self.dim) {
                    let absmax = row.iter().fold(0.0f32, |m, &val| m.max(val.abs()));
                    let scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
                    let inv = 1.0 / scale;
                    self.v_scales.push(scale);
                    for &val in row {
                        let q = (val * inv).round().clamp(-127.0, 127.0) as i8;
                        self.q8_values.push(q);
                        v_deq.push(q as f32 * scale);
                    }
                }
                Tensor::from_f32(&v_deq, &[new_tokens, self.dim])
            } else {
                // K8V4: Hadamard rotate V before 4-bit quantization (Strata / QuIP# style)
                let v_rot = v_cpu.hadamard()?;
                let v_rot_slice = v_rot.as_f32_slice().unwrap();
                let row_bytes = self.dim.div_ceil(2);
                let mut v_rot_deq = Vec::with_capacity(new_tokens * self.dim);
                for row in v_rot_slice.chunks_exact(self.dim) {
                    let absmax = row.iter().fold(0.0f32, |m, &val| m.max(val.abs()));
                    let scale = if absmax > 0.0 { absmax / 7.0 } else { 1.0 };
                    let inv = 1.0 / scale;
                    self.v_scales.push(scale);
                    for b_idx in 0..row_bytes {
                        let i0 = b_idx * 2;
                        let q0 = (row[i0] * inv).round().clamp(-8.0, 7.0) as i8;
                        let q1 = if i0 + 1 < self.dim {
                            (row[i0 + 1] * inv).round().clamp(-8.0, 7.0) as i8
                        } else {
                            0
                        };
                        let packed = ((q0 + 8) as u8 & 0x0F) | (((q1 + 8) as u8 & 0x0F) << 4);
                        self.q4_values.push(packed);
                        v_rot_deq.push(q0 as f32 * scale);
                        if i0 + 1 < self.dim {
                            v_rot_deq.push(q1 as f32 * scale);
                        }
                    }
                }
                Tensor::from_f32(&v_rot_deq, &[new_tokens, self.dim]).hadamard()?
            };

            let k_t = Tensor::from_f32(&k_deq, &[new_tokens, self.dim]).to(k_new.device())?;
            let v_t = v_deq_tensor.to(v_new.device())?;
            (k_t, v_t)
        } else {
            (k_new.clone(), v_new.clone())
        };

        if !self.k.device().is_cpu() {
            let backend = kore_core::device::get_backend(self.k.device()).ok_or_else(|| {
                KoreError::StorageError("no backend registered for device".into())
            })?;
            let (k, v) = if fusion_enabled() {
                backend.kv_cache_append_pair(&self.k, &self.v, &k_eff, &v_eff, self.len)
            } else {
                Err(KoreError::StorageError("fusion disabled".into()))
            }
            .or_else(|_| {
                let k = backend.kv_cache_append(&self.k, &k_eff, self.len);
                let v = backend.kv_cache_append(&self.v, &v_eff, self.len);
                k.and_then(|k| v.map(|v| (k, v)))
            })?;
            self.k = k;
            self.v = v;
        } else {
            let k_new_slice = k_eff.contiguous();
            let k_new_data = k_new_slice.as_f32_slice().unwrap();
            let v_new_slice = v_eff.contiguous();
            let v_new_data = v_new_slice.as_f32_slice().unwrap();

            let dim = self.dim;
            let offset = self.len * dim;
            let count = new_tokens * dim;
            let mut k_cpu = self.k.clone().contiguous();
            let mut v_cpu = self.v.clone().contiguous();
            let k_mut = k_cpu.as_f32_slice_mut().unwrap();
            let v_mut = v_cpu.as_f32_slice_mut().unwrap();
            k_mut[offset..offset + count].copy_from_slice(&k_new_data[..count]);
            v_mut[offset..offset + count].copy_from_slice(&v_new_data[..count]);
            self.k = k_cpu;
            self.v = v_cpu;
        }
        self.len += new_tokens;
        Ok(())
    }

    /// Roll back the KV-cache to `new_len` tokens in $O(1)$ time (used when a speculative draft token is rejected).
    pub fn truncate(&mut self, new_len: usize) {
        if new_len >= self.len {
            return;
        }
        self.len = new_len;
        if self.quant_mode != KvCacheQuantMode::F32 {
            self.q_keys.truncate(new_len * self.dim);
            self.k_scales.truncate(new_len);
            self.v_scales.truncate(new_len);
            if self.quant_mode == KvCacheQuantMode::Q8_0 {
                self.q8_values.truncate(new_len * self.dim);
            } else {
                self.q4_values.truncate(new_len * self.dim.div_ceil(2));
            }
        }
        if !self.k.device().is_cpu() && new_len > 0 {
            if let Some(backend) = kore_core::device::get_backend(self.k.device()) {
                let empty = Tensor::zeros(&[0, self.dim], DType::F32);
                if let (Ok(kv), Ok(vv)) = (
                    backend.kv_cache_append(&self.k, &empty, new_len),
                    backend.kv_cache_append(&self.v, &empty, new_len),
                ) {
                    self.k = kv;
                    self.v = vv;
                }
            }
        }
    }

    /// Effective memory footprint in bytes of the cached K/V states.
    pub fn memory_bytes(&self) -> usize {
        match self.quant_mode {
            KvCacheQuantMode::F32 => self.len * self.dim * 2 * 4,
            KvCacheQuantMode::Q8_0 => {
                self.q_keys.len()
                    + self.q8_values.len()
                    + (self.k_scales.len() + self.v_scales.len()) * 4
            }
            KvCacheQuantMode::K8V4 => {
                self.q_keys.len()
                    + self.q4_values.len()
                    + (self.k_scales.len() + self.v_scales.len()) * 4
            }
        }
    }

    pub fn clear(&mut self) {
        self.truncate(0);
    }
}

/// Multi-head Grouped Query Attention (GQA) layer with RoPE.
pub struct LlamaAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub rope_inv_freq: Vec<f32>,
    /// Fused q|k|v packed weight buffer (same quant kind only), computed once
    /// at construction so decode can issue a single GEMV dispatch per layer.
    pub qkv_raw: Option<(Tensor, PackedQuant)>,
}

impl LlamaAttention {
    pub fn new(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_theta: f32,
        rope_inv_freq: Vec<f32>,
    ) -> Self {
        let qkv_raw = Linear::concat_packed3(&q_proj, &k_proj, &v_proj);
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            rope_inv_freq,
            qkv_raw,
        }
    }

    pub fn to(&mut self, device: Device) -> Result<()> {
        self.q_proj.to(device)?;
        self.k_proj.to(device)?;
        self.v_proj.to(device)?;
        self.o_proj.to(device)?;
        if let Some((raw, _)) = &mut self.qkv_raw {
            *raw = raw.to(device)?;
        }
        Ok(())
    }

    pub fn forward(
        &self,
        x: &Tensor,
        pos: usize,
        kv_cache: &mut LlamaKvCacheLayer,
        device: Device,
    ) -> Result<Tensor> {
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        if !device.is_cpu() {
            let backend = kore_core::device::get_backend(device)
                .ok_or_else(|| KoreError::StorageError(format!("No backend for {}", device)))?;

            // Fused Q|K|V projection when all three share a packed quant kind.
            let k_in = self.q_proj.in_features();
            let split3 = match if fusion_enabled() {
                self.qkv_raw.as_ref()
            } else {
                None
            } {
                Some((raw, PackedQuant::Q4K)) => backend.gemv_q4_k_split3(
                    x,
                    raw,
                    self.q_proj.out_features(),
                    self.k_proj.out_features(),
                    self.v_proj.out_features(),
                    k_in,
                ),
                Some((raw, PackedQuant::Q6K)) => backend.gemv_q6_k_split3(
                    x,
                    raw,
                    self.q_proj.out_features(),
                    self.k_proj.out_features(),
                    self.v_proj.out_features(),
                    k_in,
                ),
                None => Err(KoreError::StorageError("no fused qkv weights".into())),
            };
            let (q, k, v) = match split3 {
                Ok(t) => t,
                Err(_) => (
                    self.q_proj.forward(x)?,
                    self.k_proj.forward(x)?,
                    self.v_proj.forward(x)?,
                ),
            };

            let rope_res = if fusion_enabled() {
                backend.rope_pair(
                    &q,
                    &k,
                    pos,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.rope_theta,
                )
            } else {
                Err(KoreError::StorageError("fusion disabled".into()))
            };
            let (q_rope, k_rope) = match rope_res {
                Ok(pair) => pair,
                Err(_) => (
                    backend.rope(&q, pos, self.num_heads, self.head_dim, self.rope_theta)?,
                    backend.rope(&k, pos, self.num_kv_heads, self.head_dim, self.rope_theta)?,
                ),
            };

            kv_cache.append(&k_rope, &v)?;

            let attn_out = backend.fused_attention_decode_gqa(
                &q_rope,
                &kv_cache.k,
                &kv_cache.v,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                scale,
            )?;

            self.o_proj.forward(&attn_out)
        } else {
            let q = self.q_proj.forward(x)?;
            let k = self.k_proj.forward(x)?;
            let v = self.v_proj.forward(x)?;
            let q_rope = apply_rope(&q, pos, self.head_dim, &self.rope_inv_freq)?;
            let k_rope = apply_rope(&k, pos, self.head_dim, &self.rope_inv_freq)?;

            kv_cache.append(&k_rope, &v)?;

            let seq_q = q_rope.shape().dims()[0];
            let total_seq = kv_cache.len;
            let causal_mask = if seq_q > 1 {
                let mut m_data = vec![0.0f32; seq_q * total_seq];
                let base_pos = total_seq - seq_q;
                for r in 0..seq_q {
                    let max_col = base_pos + r;
                    for c in (max_col + 1)..total_seq {
                        m_data[r * total_seq + c] = f32::NEG_INFINITY;
                    }
                }
                Some(Tensor::from_f32(&m_data, &[seq_q, total_seq]))
            } else {
                None
            };

            let kv_group_size = self.num_heads / self.num_kv_heads;
            let mut head_outputs = Vec::with_capacity(self.num_heads);
            for h in 0..self.num_heads {
                let kv_h = h / kv_group_size;
                let q_head = slice_head(&q_rope, h, self.head_dim, None)?;
                let k_head = slice_head(&kv_cache.k, kv_h, self.head_dim, Some(kv_cache.len))?;
                let v_head = slice_head(&kv_cache.v, kv_h, self.head_dim, Some(kv_cache.len))?;
                let out_h = kore_attention::scaled_dot::scaled_dot_product_attention(
                    &q_head,
                    &k_head,
                    &v_head,
                    causal_mask.as_ref(),
                    None,
                )?;
                head_outputs.push(out_h);
            }

            let concat = concat_heads(&head_outputs, seq_q, self.num_heads, self.head_dim)?;
            self.o_proj.forward(&concat)
        }
    }
}

/// SwiGLU Feed-Forward Network: `down_proj(silu(gate_proj(x)) * up_proj(x))`
/// Automatically fuses `gate_proj` and `up_proj` into a single `gate_up_proj` GEMV when quant types match.
pub struct LlamaMlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
    pub gate_up_proj: Option<Linear>,
}

impl LlamaMlp {
    pub fn new(gate_proj: Linear, up_proj: Linear, down_proj: Linear) -> Self {
        let gate_up_proj = Linear::try_concat(&gate_proj, &up_proj);
        Self {
            gate_proj,
            up_proj,
            down_proj,
            gate_up_proj,
        }
    }

    pub fn to(&mut self, device: Device) -> Result<()> {
        if let Some(ref mut gu) = self.gate_up_proj {
            gu.to(device)?;
        } else {
            self.gate_proj.to(device)?;
            self.up_proj.to(device)?;
        }
        self.down_proj.to(device)?;
        Ok(())
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(ref gate_up) = self.gate_up_proj {
            let gu = gate_up.forward(x)?;
            let inter = self.up_proj.out_features();
            let act = if !gu.device().is_cpu() {
                let backend = kore_core::device::get_backend(gu.device()).ok_or_else(|| {
                    KoreError::StorageError(format!("No backend for {}", gu.device()))
                })?;
                backend.swiglu_split(&gu, inter)?
            } else {
                let gu_cont = gu.contiguous();
                let gu_slice = gu_cont.as_f32_slice().unwrap();
                let rows = gu.shape().dims()[0];
                let mut out = vec![0.0f32; rows * inter];
                for r in 0..rows {
                    let row_in = &gu_slice[r * 2 * inter..(r + 1) * 2 * inter];
                    let row_out = &mut out[r * inter..(r + 1) * inter];
                    for i in 0..inter {
                        let g = row_in[i];
                        let u = row_in[inter + i];
                        row_out[i] = (g / (1.0 + (-g).exp())) * u;
                    }
                }
                let _ = silu;
                Tensor::from_f32(&out, &[rows, inter])
            };
            self.down_proj.forward(&act)
        } else {
            let gate = self.gate_proj.forward(x)?;
            let up = self.up_proj.forward(x)?;
            let act = swiglu(&gate, &up)?;
            self.down_proj.forward(&act)
        }
    }
}

/// Single LLaMA transformer decoder block.
pub struct LlamaDecoderLayer {
    pub input_layernorm: RMSNorm,
    pub self_attn: LlamaAttention,
    pub post_attention_layernorm: RMSNorm,
    pub mlp: LlamaMlp,
}

impl LlamaDecoderLayer {
    pub fn layer_device(&self) -> Device {
        self.self_attn.q_proj.weight().device()
    }

    pub fn to(&mut self, device: Device) -> Result<()> {
        self.input_layernorm.to(device)?;
        self.self_attn.to(device)?;
        self.post_attention_layernorm.to(device)?;
        self.mlp.to(device)?;
        Ok(())
    }

    pub fn forward(
        &self,
        x: &Tensor,
        pos: usize,
        kv_cache: &mut LlamaKvCacheLayer,
        device: Device,
    ) -> Result<Tensor> {
        let norm1 = self.input_layernorm.forward(x)?;
        let attn = self.self_attn.forward(&norm1, pos, kv_cache, device)?;

        // Fused residual add + RMSNorm on GPU: one dispatch produces both the
        // residual stream h = x + attn and its normalized form for the MLP.
        let fused = if !device.is_cpu() && fusion_enabled() {
            kore_core::device::get_backend(device).and_then(|backend| {
                backend
                    .add_rms_norm(
                        x,
                        &attn,
                        self.post_attention_layernorm.gamma(),
                        self.post_attention_layernorm.eps(),
                    )
                    .ok()
            })
        } else {
            None
        };
        let (h, norm2) = match fused {
            Some(pair) => pair,
            None => {
                let h = x.add(&attn)?;
                let n = self.post_attention_layernorm.forward(&h)?;
                (h, n)
            }
        };

        let mlp = self.mlp.forward(&norm2)?;
        h.add(&mlp)
    }

    /// GPU-only fused forward: takes the residual stream `x` and its
    /// `input_layernorm(x)`, returns the next residual plus the next normed
    /// input (normed with `next_norm`, i.e. layer i+1's input_layernorm or the
    /// model's final norm). Lets both residual adds and both norms collapse
    /// into two `add_rms_norm` dispatches per layer.
    pub fn forward_fused(
        &self,
        x: &Tensor,
        x_normed: &Tensor,
        next_norm: &RMSNorm,
        pos: usize,
        kv_cache: &mut LlamaKvCacheLayer,
        device: Device,
    ) -> Result<(Tensor, Tensor)> {
        let backend = kore_core::device::get_backend(device)
            .ok_or_else(|| KoreError::StorageError(format!("No backend for {}", device)))?;
        let attn = self.self_attn.forward(x_normed, pos, kv_cache, device)?;
        let (h, norm2) = match backend.add_rms_norm(
            x,
            &attn,
            self.post_attention_layernorm.gamma(),
            self.post_attention_layernorm.eps(),
        ) {
            Ok(pair) => pair,
            Err(_) => {
                let h = x.add(&attn)?;
                let n = self.post_attention_layernorm.forward(&h)?;
                (h, n)
            }
        };
        let mlp = self.mlp.forward(&norm2)?;
        match backend.add_rms_norm(&h, &mlp, next_norm.gamma(), next_norm.eps()) {
            Ok(pair) => Ok(pair),
            Err(_) => {
                let x_next = h.add(&mlp)?;
                let nxn = next_norm.forward(&x_next)?;
                Ok((x_next, nxn))
            }
        }
    }
}

/// Complete GPU-resident or Hybrid GPU+CPU Causal LLaMA Model.
pub struct LlamaModel {
    pub config: LlamaConfig,
    pub embed_tokens: Embedding,
    pub layers: Vec<LlamaDecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: Linear,
    pub device: Device,
    /// Optional trained MTP draft head used by speculative decoding (`--mtp`).
    /// `None` → PLD n-gram drafting only (a random-init head proposes near-guaranteed
    /// rejects, so it is only worth attaching when trained weights exist).
    pub mtp_draft_head: Option<MtpDraftHead>,
}

impl LlamaModel {
    /// Move all model compute layers to target device (e.g. `Device::Vulkan(0)`).
    /// Embeddings remain host-side so token row slicing is instantaneous without consuming VRAM.
    pub fn to(&mut self, device: Device) -> Result<()> {
        for layer in &mut self.layers {
            layer.to(device)?;
        }
        self.norm.to(device)?;
        self.lm_head.to(device)?;
        self.device = device;
        Ok(())
    }

    /// Strata-style Hybrid GPU + CPU tiered placement:
    /// Places `0..gpu_layers` in GPU VRAM (`device`) and keeps remaining layers in CPU RAM
    /// (executed via zero-allocation in-place Rayon+SIMD kernels).
    pub fn to_hybrid(&mut self, device: Device, gpu_layers: usize) -> Result<()> {
        let n_gpu = gpu_layers.min(self.layers.len());
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if i < n_gpu {
                layer.to(device)?;
            } else {
                layer.to(Device::Cpu)?;
            }
        }
        let head_dev = if n_gpu == self.layers.len() {
            device
        } else {
            Device::Cpu
        };
        self.norm.to(head_dev)?;
        self.lm_head.to(head_dev)?;
        self.device = if n_gpu > 0 { device } else { Device::Cpu };
        Ok(())
    }

    /// Attach a trained `MtpDraftHead` for speculative decoding (`--mtp`).
    /// Moves it to the lm_head's device for GPU-side draft proposal.
    pub fn attach_mtp_head(&mut self, mut head: MtpDraftHead) -> Result<()> {
        let head_dev = self.lm_head.weight().device();
        if !head_dev.is_cpu() {
            head.to(head_dev)?;
        }
        self.mtp_draft_head = Some(head);
        Ok(())
    }

    /// Compute one autoregressive decode step for `token_id` at sequence position `pos`.
    /// Returns the sampled greedy next token ID.
    pub fn step(
        &self,
        token_id: usize,
        pos: usize,
        kv_caches: &mut [LlamaKvCacheLayer],
    ) -> Result<usize> {
        let logits = self.forward_step(token_id, pos, kv_caches)?;
        if !logits.device().is_cpu() {
            let tokens = logits.argmax_tokens()?;
            Ok(tokens[0] as usize)
        } else {
            let slice = logits.as_f32_slice().unwrap();
            Ok(argmax_slice(&slice[..self.config.vocab_size]))
        }
    }

    /// Compute one forward decode step, producing output logits `[1, vocab_size]`.
    pub fn forward_step(
        &self,
        token_id: usize,
        pos: usize,
        kv_caches: &mut [LlamaKvCacheLayer],
    ) -> Result<Tensor> {
        self.forward_batch(&[token_id], pos, kv_caches)
    }

    /// Compute a small-batch (`M = 1..4`) or prefill forward pass starting at `start_pos`,
    /// returning output logits `[M, vocab_size]` and last normed hidden state `[1, hidden_size]`.
    pub fn forward_batch_with_hidden(
        &self,
        token_ids: &[usize],
        start_pos: usize,
        kv_caches: &mut [LlamaKvCacheLayer],
    ) -> Result<(Tensor, Tensor)> {
        let tok_floats: Vec<f32> = token_ids.iter().map(|&id| id as f32).collect();
        let tok_t = Tensor::from_f32(&tok_floats, &[token_ids.len()]);
        let mut x = self.embed_tokens.forward(&tok_t)?;

        // When set, x_normed is the next layer's input_layernorm(x), produced
        // by the previous layer's fused tail add+norm dispatch.
        let mut x_normed: Option<Tensor> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let l_dev = layer.layer_device();
            if x.device() != l_dev {
                x = x.to(l_dev)?;
                x_normed = None;
            }
            let n_in = match x_normed.take() {
                Some(n) if n.device() == l_dev => Some(n),
                _ if !l_dev.is_cpu() && fusion_enabled() => layer.input_layernorm.forward(&x).ok(),
                _ => None,
            };
            let next_norm = if i + 1 < self.layers.len() {
                &self.layers[i + 1].input_layernorm
            } else {
                &self.norm
            };
            x = match n_in.and_then(|n| {
                layer
                    .forward_fused(&x, &n, next_norm, start_pos, &mut kv_caches[i], l_dev)
                    .ok()
            }) {
                Some((nx, nxn)) => {
                    x_normed = Some(nxn);
                    nx
                }
                None => layer.forward(&x, start_pos, &mut kv_caches[i], l_dev)?,
            };
        }

        let head_dev = self.lm_head.weight().device();
        let normed = match x_normed {
            Some(n) if n.device() == head_dev => n,
            Some(n) => n.to(head_dev)?,
            None => {
                if x.device() != head_dev {
                    x = x.to(head_dev)?;
                }
                self.norm.forward(&x)?
            }
        };
        let m = normed.shape().dims()[0];
        let last_hidden = if m > 1 {
            normed.slice_rows(m - 1, 1)?
        } else {
            normed.clone()
        };
        let logits = self.lm_head.forward(&normed)?;
        Ok((logits, last_hidden))
    }

    /// Compute a small-batch (`M = 1..4`) or prefill forward pass starting at `start_pos`,
    /// producing output logits `[M, vocab_size]`.
    pub fn forward_batch(
        &self,
        token_ids: &[usize],
        start_pos: usize,
        kv_caches: &mut [LlamaKvCacheLayer],
    ) -> Result<Tensor> {
        let (logits, _) = self.forward_batch_with_hidden(token_ids, start_pos, kv_caches)?;
        Ok(logits)
    }

    /// Create new KV-cache buffers for this model, matching each layer's resident device.
    pub fn create_kv_caches(&self, max_seq: usize) -> Result<Vec<LlamaKvCacheLayer>> {
        self.create_kv_caches_with_mode(max_seq, KvCacheQuantMode::F32)
    }

    /// Create new KV-cache buffers with the specified quantization mode (`F32`, `Q8_0`, or `K8V4`).
    pub fn create_kv_caches_with_mode(
        &self,
        max_seq: usize,
        quant_mode: KvCacheQuantMode,
    ) -> Result<Vec<LlamaKvCacheLayer>> {
        let mut caches = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            caches.push(LlamaKvCacheLayer::with_quant_mode(
                max_seq,
                self.config.num_key_value_heads,
                self.config.head_dim,
                layer.layer_device(),
                quant_mode,
            )?);
        }
        Ok(caches)
    }

    /// Autoregressive text generation loop given prompt token IDs.
    ///
    /// Uses 4-token batched prefill (`M = 4` small-batch GEMV) to process the prompt in 1/4th the weight loads.
    /// Generates up to `max_new_tokens` tokens, calling `on_token` callback for each generated token ID.
    pub fn generate<F>(
        &self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        eos_token_id: Option<usize>,
        on_token: F,
    ) -> Result<Vec<usize>>
    where
        F: FnMut(usize),
    {
        self.generate_with_options(
            prompt_tokens,
            max_new_tokens,
            eos_token_id,
            false,
            KvCacheQuantMode::F32,
            on_token,
        )
        .map(|(tokens, _, _)| tokens)
    }

    /// Autoregressive text generation with optional speculative decoding (`PromptLookupDrafter` + `MtpDraftHead`)
    /// and configurable KV-cache quantization (`F32`, `Q8_0`, `K8V4`).
    ///
    /// Returns `(generated_tokens, accepted_draft_tokens, total_drafted_tokens)`.
    pub fn generate_with_options<F>(
        &self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        eos_token_id: Option<usize>,
        speculative: bool,
        kv_mode: KvCacheQuantMode,
        mut on_token: F,
    ) -> Result<(Vec<usize>, usize, usize)>
    where
        F: FnMut(usize),
    {
        if prompt_tokens.is_empty() {
            return Err(KoreError::StorageError(
                "prompt_tokens cannot be empty".into(),
            ));
        }

        let max_seq = prompt_tokens.len() + max_new_tokens + 16;
        let mut kv_caches = self.create_kv_caches_with_mode(max_seq, kv_mode)?;

        // Batched prefill in chunks of up to 4 tokens (batch-invariant small-batch GEMV)
        let mut next_token = 0;
        let mut last_hidden = Tensor::zeros(&[1, self.config.hidden_size], DType::F32);
        let mut pos = 0;
        while pos < prompt_tokens.len() {
            let chunk_len = (prompt_tokens.len() - pos).min(4);
            let chunk = &prompt_tokens[pos..pos + chunk_len];
            let (logits, hidden) = self.forward_batch_with_hidden(chunk, pos, &mut kv_caches)?;
            last_hidden = hidden;
            next_token = sample_last_greedy(&logits, self.config.vocab_size)?;
            pos += chunk_len;
        }

        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut history = Vec::with_capacity(prompt_tokens.len() + max_new_tokens);
        history.extend_from_slice(prompt_tokens);

        let mut curr_token = next_token;
        let mut curr_pos = prompt_tokens.len();
        let mut accepted_drafts = 0usize;
        let mut total_drafts = 0usize;

        let pld = PromptLookupDrafter::new(3);
        // Only use an MTP draft head when trained weights were attached to the model —
        // a random-init head proposes near-guaranteed rejects that still cost a verify pass.
        let mtp_head = self.mtp_draft_head.as_ref();
        let greedy_cfg = SamplerConfig::greedy();

        // Strata-style adaptive draft policy (draft_policy.cpp): speculative rounds only pay off
        // when (tokens produced per round) / (round cost) beats single-token decode with a margin.
        // Track EMAs of verify-round cost, single-step cost, and tokens produced per round; disable
        // drafting when the measured throughput ratio falls below the margin.
        let mut spec_rounds = 0usize;
        let mut ema_verify_ms = 0.0f64;
        let mut ema_single_ms = 0.0f64;
        let mut ema_tok_per_round = 0.0f64;
        let mut single_steps = 0usize;
        // Spec starts enabled; if no single-step baseline exists after the warmup window
        // (PLD drafts every round), pause to calibrate before latching the policy decision.
        let mut spec_enabled = speculative;
        let mut spec_disabled = false;
        const SPEC_WARMUP: usize = 8;
        const SPEC_MIN_STEPS: usize = 8;
        const SPEC_MARGIN: f64 = 1.10;

        while generated.len() < max_new_tokens {
            if let Some(eos) = eos_token_id {
                if curr_token == eos {
                    break;
                }
            }
            generated.push(curr_token);
            history.push(curr_token);
            on_token(curr_token);
            if generated.len() >= max_new_tokens {
                break;
            }

            if spec_enabled {
                // 1. Try Strata zero-cost Prompt Lookup Drafter (n-gram context match) first
                let mut drafts = pld.draft(&history);
                // 2. Fallback to 1-layer MTP Draft Head if PLD found no n-gram match in context
                if drafts.is_empty() {
                    if let Some(mtp) = mtp_head {
                        if let Ok(draft_logits) = mtp.forward_draft(
                            &last_hidden,
                            curr_token,
                            &self.embed_tokens,
                            &self.lm_head,
                        ) {
                            if let Ok(d_tok) = crate::sampler::sample_from_tensor(
                                &draft_logits,
                                &[],
                                &greedy_cfg,
                                42,
                                curr_pos,
                            ) {
                                drafts.push(d_tok);
                            }
                        }
                    }
                }

                if !drafts.is_empty() {
                    let max_rem = max_new_tokens - generated.len();
                    drafts.truncate(max_rem.min(3));
                    let n_drafts = drafts.len();
                    total_drafts += n_drafts;

                    let mut verify_batch = Vec::with_capacity(1 + n_drafts);
                    verify_batch.push(curr_token);
                    verify_batch.extend_from_slice(&drafts);

                    let t_verify = std::time::Instant::now();
                    let (logits_batch, hidden_last) =
                        self.forward_batch_with_hidden(&verify_batch, curr_pos, &mut kv_caches)?;
                    last_hidden = hidden_last;
                    let verified_tokens = sample_all_greedy(&logits_batch, self.config.vocab_size)?;
                    let verify_ms = t_verify.elapsed().as_secs_f64() * 1000.0;

                    let mut n_accepted = 0usize;
                    for i in 0..n_drafts {
                        if verified_tokens[i] == drafts[i] {
                            n_accepted += 1;
                            let accepted_tok = drafts[i];
                            if let Some(eos) = eos_token_id {
                                if accepted_tok == eos {
                                    break;
                                }
                            }
                            if generated.len() < max_new_tokens {
                                generated.push(accepted_tok);
                                history.push(accepted_tok);
                                on_token(accepted_tok);
                            }
                        } else {
                            break;
                        }
                    }
                    accepted_drafts += n_accepted;

                    // Adaptive policy update (Strata draft_policy): the round produced
                    // `1 + n_accepted` tokens in `verify_ms`; keep spec only while
                    // tokens/round ÷ cost-per-token beats single-step decode by a margin.
                    spec_rounds += 1;
                    let round_toks = 1 + n_accepted;
                    ema_verify_ms = if spec_rounds == 1 {
                        verify_ms
                    } else {
                        ema_verify_ms * 0.7 + verify_ms * 0.3
                    };
                    ema_tok_per_round = if spec_rounds == 1 {
                        round_toks as f64
                    } else {
                        ema_tok_per_round * 0.7 + round_toks as f64 * 0.3
                    };
                    if std::env::var("KORE_SPEC_DEBUG").is_ok() {
                        eprintln!(
                            "[spec] round {}: verify={:.2}ms accepted={}/{} ema_tok/r={:.2} ema_verify={:.2}ms ema_single={:.2}ms",
                            spec_rounds, verify_ms, n_accepted, n_drafts,
                            ema_tok_per_round, ema_verify_ms, ema_single_ms
                        );
                    }
                    if spec_rounds >= SPEC_WARMUP {
                        if single_steps >= SPEC_MIN_STEPS {
                            let spec_tps = ema_tok_per_round / ema_verify_ms;
                            let single_tps = 1.0 / ema_single_ms;
                            if spec_tps < single_tps * SPEC_MARGIN {
                                spec_enabled = false;
                                spec_disabled = true;
                            }
                        } else {
                            // No baseline yet — pause speculation to calibrate.
                            spec_enabled = false;
                        }
                    }

                    // Roll back rejected KV-cache entries in O(1)
                    let valid_kv_len = curr_pos + 1 + n_accepted;
                    if n_accepted < n_drafts {
                        for cache in &mut kv_caches {
                            cache.truncate(valid_kv_len);
                        }
                    }
                    curr_pos = valid_kv_len;
                    curr_token = verified_tokens[n_accepted];
                    continue;
                }
            }

            let t_step = std::time::Instant::now();
            let (logits, hidden) =
                self.forward_batch_with_hidden(&[curr_token], curr_pos, &mut kv_caches)?;
            last_hidden = hidden;
            curr_token = sample_last_greedy(&logits, self.config.vocab_size)?;
            let step_ms = t_step.elapsed().as_secs_f64() * 1000.0;
            single_steps += 1;
            ema_single_ms = if single_steps == 1 {
                step_ms
            } else {
                ema_single_ms * 0.7 + step_ms * 0.3
            };
            // Baseline calibrated — engage speculation if requested and not ruled out.
            if speculative && !spec_disabled && single_steps >= SPEC_MIN_STEPS {
                spec_enabled = true;
            }
            curr_pos += 1;
        }

        Ok((generated, accepted_drafts, total_drafts))
    }

    /// Load a Causal LLaMA model from a GGUF checkpoint.
    ///
    /// If `max_layers` is specified, loads up to `max_layers` (useful for fast verification
    /// or resource-bounded testing).
    pub fn from_gguf(path: &std::path::Path, max_layers: Option<usize>) -> Result<Self> {
        let gguf = oxillama_gguf::GgufModel::load(path).map_err(|e| {
            KoreError::StorageError(format!("failed to load GGUF {}: {e}", path.display()))
        })?;
        let meta = &gguf.file.metadata;

        let total_layers = meta
            .get("llama.block_count")
            .and_then(|v| v.as_u32())
            .ok_or_else(|| KoreError::StorageError("missing llama.block_count in GGUF".into()))?
            as usize;

        let num_layers = max_layers.unwrap_or(total_layers).min(total_layers);

        let hidden_size = meta
            .get("llama.embedding_length")
            .and_then(|v| v.as_u32())
            .ok_or_else(|| {
                KoreError::StorageError("missing llama.embedding_length in GGUF".into())
            })? as usize;

        let vocab_size = meta
            .get("llama.vocab_size")
            .and_then(|v| v.as_u32())
            .map(|v| v as usize)
            .or_else(|| {
                meta.get("tokenizer.ggml.tokens")
                    .and_then(|v| v.as_array().map(|a| a.len()))
            })
            .unwrap_or(32000);

        let num_attention_heads = meta
            .get("llama.attention.head_count")
            .and_then(|v| v.as_u32())
            .ok_or_else(|| {
                KoreError::StorageError("missing llama.attention.head_count in GGUF".into())
            })? as usize;

        let num_key_value_heads = meta
            .get("llama.attention.head_count_kv")
            .and_then(|v| v.as_u32())
            .map(|v| v as usize)
            .unwrap_or(num_attention_heads);

        let intermediate_size = meta
            .get("llama.feed_forward_length")
            .and_then(|v| v.as_u32())
            .map(|v| v as usize)
            .unwrap_or(hidden_size * 4);

        let rms_norm_eps = meta
            .get("llama.attention.layer_norm_rms_epsilon")
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-5);

        let rope_theta = meta
            .get("llama.rope.freq_base")
            .and_then(|v| v.as_f32())
            .unwrap_or(10000.0);

        let head_dim = hidden_size / num_attention_heads;

        let config = LlamaConfig {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers: num_layers,
            num_attention_heads,
            num_key_value_heads,
            rms_norm_eps,
            rope_theta,
            head_dim,
        };

        let dequant_to_tensor = |name: &str, shape: &[usize]| -> Result<Tensor> {
            let info =
                gguf.file.tensors.get(name).map_err(|e| {
                    KoreError::StorageError(format!("missing tensor '{name}': {e}"))
                })?;
            let raw = gguf.tensor_data(name).map_err(|e| {
                KoreError::StorageError(format!("cannot read tensor '{name}': {e}"))
            })?;
            let numel: usize = shape.iter().product();
            let f32_data = oxillama_quant::dequantize_to_f32(raw, info.tensor_type, numel)
                .map_err(|e| KoreError::StorageError(format!("dequant '{name}' failed: {e}")))?;
            Ok(Tensor::from_f32(&f32_data, shape))
        };

        let load_linear = |name: &str, in_features: usize, out_features: usize| -> Result<Linear> {
            let info =
                gguf.file.tensors.get(name).map_err(|e| {
                    KoreError::StorageError(format!("missing tensor '{name}': {e}"))
                })?;
            let raw = gguf.tensor_data(name).map_err(|e| {
                KoreError::StorageError(format!("cannot read tensor '{name}': {e}"))
            })?;
            if info.tensor_type == oxillama_gguf::GgufTensorType::Q4K {
                Ok(Linear::from_q4_k(
                    raw.to_vec(),
                    in_features,
                    out_features,
                    None,
                ))
            } else if info.tensor_type == oxillama_gguf::GgufTensorType::Q6K {
                Ok(Linear::from_q6_k(
                    raw.to_vec(),
                    in_features,
                    out_features,
                    None,
                ))
            } else {
                let numel = in_features * out_features;
                let f32_data = oxillama_quant::dequantize_to_f32(raw, info.tensor_type, numel)
                    .map_err(|e| {
                        KoreError::StorageError(format!("dequant '{name}' failed: {e}"))
                    })?;
                let w = Tensor::from_f32(&f32_data, &[out_features, in_features]);
                let w_t = w.transpose()?.contiguous();
                Ok(Linear::from_weight_transposed(w_t, None))
            }
        };

        let embed_tokens = {
            let raw = gguf.tensor_data("token_embd.weight").map_err(|e| {
                KoreError::StorageError(format!("cannot read token_embd.weight: {e}"))
            })?;
            let info = gguf.file.tensors.get("token_embd.weight").map_err(|e| {
                KoreError::StorageError(format!("missing tensor token_embd.weight: {e}"))
            })?;
            if info.tensor_type == oxillama_gguf::GgufTensorType::Q4K {
                Embedding::from_q4_k(raw.to_vec(), vocab_size, hidden_size)
            } else {
                let embed_w = dequant_to_tensor("token_embd.weight", &[vocab_size, hidden_size])?;
                Embedding::from_weight(embed_w)
            }
        };

        let rope_inv_freq = config.rope_inv_freq();
        use rayon::prelude::*;

        let layers: Vec<LlamaDecoderLayer> = (0..num_layers)
            .into_par_iter()
            .map(|i| -> Result<LlamaDecoderLayer> {
                let in_norm_w =
                    dequant_to_tensor(&format!("blk.{i}.attn_norm.weight"), &[hidden_size])?;
                let input_layernorm = RMSNorm::from_weight(in_norm_w, rms_norm_eps);

                let q_proj = load_linear(
                    &format!("blk.{i}.attn_q.weight"),
                    hidden_size,
                    num_attention_heads * head_dim,
                )?;
                let k_proj = load_linear(
                    &format!("blk.{i}.attn_k.weight"),
                    hidden_size,
                    num_key_value_heads * head_dim,
                )?;
                let v_proj = load_linear(
                    &format!("blk.{i}.attn_v.weight"),
                    hidden_size,
                    num_key_value_heads * head_dim,
                )?;
                let o_proj = load_linear(
                    &format!("blk.{i}.attn_output.weight"),
                    num_attention_heads * head_dim,
                    hidden_size,
                )?;

                let self_attn = LlamaAttention::new(
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    num_attention_heads,
                    num_key_value_heads,
                    head_dim,
                    rope_theta,
                    rope_inv_freq.clone(),
                );

                let post_norm_w =
                    dequant_to_tensor(&format!("blk.{i}.ffn_norm.weight"), &[hidden_size])?;
                let post_attention_layernorm = RMSNorm::from_weight(post_norm_w, rms_norm_eps);

                let gate_proj = load_linear(
                    &format!("blk.{i}.ffn_gate.weight"),
                    hidden_size,
                    intermediate_size,
                )?;
                let up_proj = load_linear(
                    &format!("blk.{i}.ffn_up.weight"),
                    hidden_size,
                    intermediate_size,
                )?;
                let down_proj = load_linear(
                    &format!("blk.{i}.ffn_down.weight"),
                    intermediate_size,
                    hidden_size,
                )?;
                let mlp = LlamaMlp::new(gate_proj, up_proj, down_proj);

                Ok(LlamaDecoderLayer {
                    input_layernorm,
                    self_attn,
                    post_attention_layernorm,
                    mlp,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let norm_w = dequant_to_tensor("output_norm.weight", &[hidden_size])?;
        let norm = RMSNorm::from_weight(norm_w, rms_norm_eps);

        let lm_head_name = if gguf.file.tensors.get("output.weight").is_ok() {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let lm_head = load_linear(lm_head_name, hidden_size, vocab_size)?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: Device::Cpu,
            mtp_draft_head: None,
        })
    }
}

// ── Helper math functions ──────────────────────────────────────────────────

fn argmax_slice(slice: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &v) in slice.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = idx;
        }
    }
    best_idx
}

fn sample_last_greedy(logits: &Tensor, vocab_size: usize) -> Result<usize> {
    if !logits.device().is_cpu() {
        let tokens = logits.argmax_tokens()?;
        Ok(*tokens.last().unwrap_or(&0) as usize)
    } else {
        let slice = logits.as_f32_slice().unwrap();
        let last_row = &slice[slice.len() - vocab_size..];
        Ok(argmax_slice(last_row))
    }
}

fn sample_all_greedy(logits: &Tensor, vocab_size: usize) -> Result<Vec<usize>> {
    if !logits.device().is_cpu() {
        let tokens = logits.argmax_tokens()?;
        Ok(tokens.into_iter().map(|t| t as usize).collect())
    } else {
        let slice = logits.as_f32_slice().unwrap();
        Ok(slice.chunks_exact(vocab_size).map(argmax_slice).collect())
    }
}

fn apply_rope(x: &Tensor, pos: usize, head_dim: usize, inv_freq: &[f32]) -> Result<Tensor> {
    let dev = x.device();
    let cpu_x = if !dev.is_cpu() {
        x.to(Device::Cpu)?
    } else {
        x.clone()
    };
    let data = cpu_x.as_f32_slice().unwrap();
    let mut out = data.to_vec();

    let dims = x.shape().dims();
    let rows = if dims.is_empty() { 1 } else { dims[0] };
    let row_dim = data.len() / rows;
    let num_heads_per_row = row_dim / head_dim;
    let num_pairs = head_dim / 2;

    for r in 0..rows {
        let token_pos = (pos + r) as f32;
        let row_offset = r * row_dim;
        for h in 0..num_heads_per_row {
            let head_offset = row_offset + h * head_dim;
            for pair in 0..num_pairs {
                let freq = inv_freq[pair];
                let angle = token_pos * freq;
                let cos = angle.cos();
                let sin = angle.sin();

                let i0 = head_offset + 2 * pair;
                let i1 = head_offset + 2 * pair + 1;
                let x0 = data[i0];
                let x1 = data[i1];

                out[i0] = x0 * cos - x1 * sin;
                out[i1] = x0 * sin + x1 * cos;
            }
        }
    }

    let result = Tensor::from_f32(&out, x.shape().dims());
    if !dev.is_cpu() {
        result.to(dev)
    } else {
        Ok(result)
    }
}

fn slice_head(
    x: &Tensor,
    head_idx: usize,
    head_dim: usize,
    max_rows: Option<usize>,
) -> Result<Tensor> {
    let start = head_idx * head_dim;
    let dev = x.device();
    let cpu_x = if !dev.is_cpu() {
        x.to(Device::Cpu)?
    } else {
        x.clone()
    };
    let slice = cpu_x.as_f32_slice().unwrap();
    let total_rows = x.shape().dims()[0];
    let rows = max_rows.unwrap_or(total_rows).min(total_rows);
    let row_dim = x.shape().dims()[1];

    let mut out = Vec::with_capacity(rows * head_dim);
    for r in 0..rows {
        let offset = r * row_dim + start;
        out.extend_from_slice(&slice[offset..offset + head_dim]);
    }

    let res = Tensor::from_f32(&out, &[rows, head_dim]);
    if !dev.is_cpu() {
        res.to(dev)
    } else {
        Ok(res)
    }
}

fn concat_heads(
    heads: &[Tensor],
    rows: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let total_dim = num_heads * head_dim;
    let mut result = vec![0.0f32; rows * total_dim];
    let dev = heads[0].device();

    for (h, head) in heads.iter().enumerate() {
        let cpu_head = if !dev.is_cpu() {
            head.to(Device::Cpu)?
        } else {
            head.clone()
        };
        let slice = cpu_head.as_f32_slice().unwrap();
        for r in 0..rows {
            let src = r * head_dim;
            let dst = r * total_dim + h * head_dim;
            result[dst..dst + head_dim].copy_from_slice(&slice[src..src + head_dim]);
        }
    }

    let res = Tensor::from_f32(&result, &[rows, total_dim]);
    if !dev.is_cpu() {
        res.to(dev)
    } else {
        Ok(res)
    }
}
