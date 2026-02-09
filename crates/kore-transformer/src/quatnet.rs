//! QuatNet transformer — all linear projections quantized to 2-bit quaternary.
//!
//! Architecture: Llama-style decoder with quaternary {-3, -1, +1, +3} weights
//! replacing all `Linear` layers. Embeddings, norms, and LM head remain in f32.
//!
//! Memory: ~8× compression vs f32 for weight matrices (vs ~20× for BitNet ternary).
//! Higher fidelity than ternary: 4 quantization levels instead of 3.
//!
//! Usage:
//! ```ignore
//! let config = TransformerConfig::tiny();
//! let model = Transformer::new(config.clone());
//! let quatnet = QuatNetTransformer::from_transformer(&model);
//! let logits = quatnet.forward(&[0, 1, 2], false).unwrap();
//! ```

use kore_core::{KoreError, Tensor};
use kore_kernels::cpu_quat_matmul::{quat_matmul, pack_weights_quaternary};
use kore_attention::kv_cache::KvCache;

use crate::embedding::Embedding;
use crate::rms_norm::RMSNorm;
use crate::rope::RopeTable;
use crate::model::TransformerConfig;
use crate::sampler::{self, SamplerConfig, Rng};

/// Packed quaternary weight matrix with per-row scales.
struct QuatPackedWeight {
    packed: Vec<u8>,
    scales: Vec<f32>,
    out_dim: usize,
    in_dim: usize,
}

impl QuatPackedWeight {
    /// Pack a weight tensor stored as [out_features, in_features].
    fn from_tensor(weight: &Tensor) -> Self {
        let dims = weight.shape().dims();
        let out_dim = dims[0];
        let in_dim = dims[1];
        let w_data = weight.contiguous();
        let w_slice = w_data.as_f32_slice().expect("f32 weight required");
        let (packed, scales) = pack_weights_quaternary(w_slice, out_dim, in_dim);
        Self { packed, scales, out_dim, in_dim }
    }

    /// Pack a weight tensor stored as [in_features, out_features].
    /// Transposes to [out_features, in_features] before packing.
    fn from_tensor_transposed(weight: &Tensor) -> Self {
        let wt = weight.transpose().expect("transpose failed").contiguous();
        Self::from_tensor(&wt)
    }

    /// Compute output = input @ W^T.
    /// input: [seq_len, in_dim] → output: [seq_len, out_dim]
    fn forward(&self, input: &Tensor) -> Result<Tensor, KoreError> {
        let seq_len = input.shape().dims()[0];
        let input_t = input.transpose()?.contiguous();
        let out = quat_matmul(
            &self.packed, &self.scales, &input_t,
            self.out_dim, seq_len, self.in_dim,
        )?;
        out.transpose().map(|t| t.contiguous())
    }

    fn memory_bytes(&self) -> usize {
        self.packed.len() + self.scales.len() * 4
    }
}

/// QuatNet attention block — QKV + output projections are quaternary-packed.
struct QuatNetAttention {
    n_heads: usize,
    n_kv_heads: usize,
    d_model: usize,
    d_head: usize,
    wq: QuatPackedWeight,
    wk: QuatPackedWeight,
    wv: QuatPackedWeight,
    wo: QuatPackedWeight,
    kv_caches: Vec<KvCache>,
    rope: Option<RopeTable>,
    seq_offset: usize,
}

impl QuatNetAttention {
    fn from_mha(mha: &crate::mha::MultiHeadAttention, max_seq_len: usize) -> Self {
        let d_head = mha.d_head;
        let rope = mha.rope.as_ref().map(|r| RopeTable::new(r.d_head, r.max_seq_len, r.base));
        Self {
            n_heads: mha.n_heads,
            n_kv_heads: mha.n_kv_heads,
            d_model: mha.d_model,
            d_head,
            wq: QuatPackedWeight::from_tensor_transposed(&mha.wq),
            wk: QuatPackedWeight::from_tensor_transposed(&mha.wk),
            wv: QuatPackedWeight::from_tensor_transposed(&mha.wv),
            wo: QuatPackedWeight::from_tensor_transposed(&mha.wo),
            kv_caches: (0..mha.n_kv_heads)
                .map(|_| KvCache::new(d_head, d_head, max_seq_len))
                .collect(),
            rope,
            seq_offset: 0,
        }
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, use_cache: bool) -> Result<Tensor, KoreError> {
        let seq_len = x.shape().dims()[0];
        let dh = self.d_head;
        let nh = self.n_heads;
        let nkv = self.n_kv_heads;
        let heads_per_kv = nh / nkv;
        let q_dim = nh * dh;
        let kv_dim = nkv * dh;

        let q_tensor = self.wq.forward(x)?;
        let k_tensor = self.wk.forward(x)?;
        let v_tensor = self.wv.forward(x)?;

        let q_all = q_tensor.as_f32_slice().unwrap();
        let k_all = k_tensor.as_f32_slice().unwrap();
        let v_all = v_tensor.as_f32_slice().unwrap();

        let mut head_outputs = vec![0.0f32; seq_len * q_dim];

        for kv_h in 0..nkv {
            let mut k_h = extract_head(k_all, seq_len, kv_dim, kv_h, dh);
            let v_h = extract_head(v_all, seq_len, kv_dim, kv_h, dh);

            if let Some(ref rope) = self.rope {
                rope.apply_single(&mut k_h, self.seq_offset, seq_len);
            }

            let (k_full_tensor, v_full_tensor) = if use_cache {
                let cache = &mut self.kv_caches[kv_h];
                let kt = Tensor::from_f32(&k_h, &[seq_len, dh]);
                let vt = Tensor::from_f32(&v_h, &[seq_len, dh]);
                cache.update(&kt, &vt)?
            } else {
                (
                    Tensor::from_f32(&k_h, &[seq_len, dh]),
                    Tensor::from_f32(&v_h, &[seq_len, dh]),
                )
            };

            let kv_seq_len = k_full_tensor.shape().dims()[0];
            let k_t_transposed = k_full_tensor.transpose()?;

            for qh_offset in 0..heads_per_kv {
                let h = kv_h * heads_per_kv + qh_offset;
                let mut q_h = extract_head(q_all, seq_len, q_dim, h, dh);

                if let Some(ref rope) = self.rope {
                    rope.apply_single(&mut q_h, self.seq_offset, seq_len);
                }

                let q_t = Tensor::from_f32(&q_h, &[seq_len, dh]);
                let scores_tensor = q_t.matmul(&k_t_transposed.contiguous())?;
                let mut scores = scores_tensor.as_f32_slice().unwrap().to_vec();

                let scale = 1.0 / (dh as f32).sqrt();
                for s in scores.iter_mut() {
                    *s *= scale;
                }

                if let Some(m) = mask {
                    let m_data = m.as_f32_slice().ok_or(KoreError::StorageError("expected f32 mask".into()))?;
                    let m_dims = m.shape().dims();
                    if m_dims[0] >= seq_len && m_dims[1] >= kv_seq_len {
                        for i in 0..seq_len {
                            for j in 0..kv_seq_len {
                                scores[i * kv_seq_len + j] += m_data[i * m_dims[1] + j];
                            }
                        }
                    }
                }

                for i in 0..seq_len {
                    let row = &mut scores[i * kv_seq_len..(i + 1) * kv_seq_len];
                    let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for v in row.iter_mut() {
                        *v = (*v - max_val).exp();
                        sum += *v;
                    }
                    if sum > 0.0 {
                        for v in row.iter_mut() {
                            *v /= sum;
                        }
                    }
                }

                let scores_t = Tensor::from_f32(&scores, &[seq_len, kv_seq_len]);
                let attn_out_tensor = scores_t.matmul(&v_full_tensor)?;
                let attn_out = attn_out_tensor.as_f32_slice().unwrap();

                for i in 0..seq_len {
                    for j in 0..dh {
                        head_outputs[i * q_dim + h * dh + j] = attn_out[i * dh + j];
                    }
                }
            }
        }

        let head_tensor = Tensor::from_f32(&head_outputs, &[seq_len, q_dim]);
        let final_out = self.wo.forward(&head_tensor)?;

        if use_cache {
            self.seq_offset += seq_len;
        }

        Ok(final_out)
    }

    fn reset_cache(&mut self) {
        for cache in &mut self.kv_caches {
            cache.clear();
        }
        self.seq_offset = 0;
    }
}

/// QuatNet feed-forward block — gate/up/down projections are quaternary-packed.
struct QuatNetFeedForward {
    w1: QuatPackedWeight,
    w2: QuatPackedWeight,
    w3: QuatPackedWeight,
    d_ff: usize,
}

impl QuatNetFeedForward {
    fn from_ffn(ffn: &crate::feed_forward::FeedForward) -> Self {
        Self {
            w1: QuatPackedWeight::from_tensor_transposed(&ffn.w1),
            w2: QuatPackedWeight::from_tensor_transposed(&ffn.w2),
            w3: QuatPackedWeight::from_tensor_transposed(&ffn.w3),
            d_ff: ffn.d_ff,
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, KoreError> {
        let seq_len = x.shape().dims()[0];
        let ff = self.d_ff;

        let gate_tensor = self.w1.forward(x)?;
        let up_tensor = self.w3.forward(x)?;

        let gate = gate_tensor.as_f32_slice().unwrap();
        let up = up_tensor.as_f32_slice().unwrap();
        let mut hidden = vec![0.0f32; seq_len * ff];
        for i in 0..hidden.len() {
            let g = gate[i];
            let swish = g * (1.0 / (1.0 + (-g).exp()));
            hidden[i] = swish * up[i];
        }

        let hidden_tensor = Tensor::from_f32(&hidden, &[seq_len, ff]);
        self.w2.forward(&hidden_tensor)
    }
}

/// QuatNet transformer block.
struct QuatNetBlock {
    attn_norm: RMSNorm,
    attn: QuatNetAttention,
    ffn_norm: RMSNorm,
    ffn: QuatNetFeedForward,
    d_model: usize,
}

impl QuatNetBlock {
    fn from_block(block: &crate::block::TransformerBlock, max_seq_len: usize) -> Self {
        Self {
            attn_norm: RMSNorm::new(block.d_model, block.attn_norm.eps),
            attn: QuatNetAttention::from_mha(&block.attn, max_seq_len),
            ffn_norm: RMSNorm::new(block.d_model, block.ffn_norm.eps),
            ffn: QuatNetFeedForward::from_ffn(&block.ffn),
            d_model: block.d_model,
        }
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, use_cache: bool) -> Result<Tensor, KoreError> {
        let x_data = x.as_f32_slice().ok_or(KoreError::StorageError("expected f32".into()))?;
        let n = x_data.len();

        let normed = self.attn_norm.forward(x)?;
        let attn_out = self.attn.forward(&normed, mask, use_cache)?;
        let attn_data = attn_out.as_f32_slice().unwrap();

        let mut h = vec![0.0f32; n];
        for i in 0..n {
            h[i] = x_data[i] + attn_data[i];
        }
        let h_tensor = Tensor::from_f32(&h, x.shape().dims());

        let normed2 = self.ffn_norm.forward(&h_tensor)?;
        let ffn_out = self.ffn.forward(&normed2)?;
        let ffn_data = ffn_out.as_f32_slice().unwrap();

        let mut out = vec![0.0f32; n];
        for i in 0..n {
            out[i] = h[i] + ffn_data[i];
        }

        Ok(Tensor::from_f32(&out, x.shape().dims()))
    }

    fn reset_cache(&mut self) {
        self.attn.reset_cache();
    }
}

/// Full QuatNet 2-bit transformer decoder.
///
/// All linear projections (QKV, output, gate/up/down) are quantized to
/// 2-bit quaternary {-3, -1, +1, +3}. Embeddings, norms, and LM head remain in f32.
///
/// Compared to BitNet (1.58-bit ternary):
/// - Higher fidelity: 4 quantization levels vs 3
/// - Slightly more memory: ~8× compression vs ~20× for ternary
/// - Better accuracy for models sensitive to quantization error
pub struct QuatNetTransformer {
    pub config: TransformerConfig,
    pub embedding: Embedding,
    pub layers: Vec<QuatNetBlock>,
    pub final_norm: RMSNorm,
    pub lm_head: Tensor,
}

impl QuatNetTransformer {
    /// Quantize an existing f32 Transformer to QuatNet 2-bit.
    pub fn from_transformer(model: &crate::model::Transformer) -> Self {
        let layers: Vec<QuatNetBlock> = model.layers.iter()
            .map(|block| QuatNetBlock::from_block(block, model.config.max_seq_len))
            .collect();

        let final_norm = RMSNorm::new(model.config.d_model, model.config.norm_eps);

        Self {
            config: model.config.clone(),
            embedding: Embedding::new(
                model.config.vocab_size,
                model.config.d_model,
                Some(model.config.max_seq_len),
            ),
            layers,
            final_norm,
            lm_head: model.lm_head.clone(),
        }
    }

    /// Forward pass: token_ids → logits.
    pub fn forward(&mut self, token_ids: &[usize], use_cache: bool) -> Result<Tensor, KoreError> {
        let seq_len = token_ids.len();
        let mut x = self.embedding.forward(token_ids)?;

        let mask = kore_attention::mask::causal_mask(seq_len);

        for layer in &mut self.layers {
            x = layer.forward(&x, Some(&mask), use_cache)?;
        }

        x = self.final_norm.forward(&x)?;

        let x_data = x.as_f32_slice().ok_or(KoreError::StorageError("expected f32".into()))?;
        let lm_data = self.lm_head.as_f32_slice().unwrap();
        let d = self.config.d_model;
        let v = self.config.vocab_size;

        let mut logits = vec![0.0f32; seq_len * v];
        for i in 0..seq_len {
            for j in 0..v {
                let mut acc = 0.0f32;
                for k in 0..d {
                    acc += x_data[i * d + k] * lm_data[k * v + j];
                }
                logits[i * v + j] = acc;
            }
        }

        Ok(Tensor::from_f32(&logits, &[seq_len, v]))
    }

    /// Generate tokens autoregressively.
    pub fn generate(&mut self, prompt: &[usize], max_new_tokens: usize) -> Result<Vec<usize>, KoreError> {
        self.generate_with_config(prompt, max_new_tokens, &SamplerConfig::greedy(), &mut Rng::new(42))
    }

    /// Generate with configurable sampling.
    pub fn generate_with_config(
        &mut self,
        prompt: &[usize],
        max_new_tokens: usize,
        sampler_config: &SamplerConfig,
        rng: &mut Rng,
    ) -> Result<Vec<usize>, KoreError> {
        self.reset_cache();
        let mut tokens = prompt.to_vec();
        let v = self.config.vocab_size;

        let logits = self.forward(&tokens, true)?;
        let logits_data = logits.as_f32_slice().ok_or(KoreError::StorageError("expected f32".into()))?;
        let last_row = &logits_data[(tokens.len() - 1) * v..tokens.len() * v];
        let next = sampler::sample(last_row, &tokens, sampler_config, rng);
        tokens.push(next);

        if sampler_config.eos_token_id == Some(next) {
            return Ok(tokens);
        }

        for _ in 1..max_new_tokens {
            let last_token = *tokens.last().unwrap();
            let logits = self.forward(&[last_token], true)?;
            let logits_data = logits.as_f32_slice().ok_or(KoreError::StorageError("expected f32".into()))?;
            let next = sampler::sample(logits_data, &tokens, sampler_config, rng);
            tokens.push(next);

            if sampler_config.eos_token_id == Some(next) {
                break;
            }
        }

        Ok(tokens)
    }

    /// Reset all KV caches.
    pub fn reset_cache(&mut self) {
        for layer in &mut self.layers {
            layer.reset_cache();
        }
    }

    /// Total weight memory in bytes.
    pub fn weight_memory_bytes(&self) -> usize {
        let mut total = 0usize;
        for layer in &self.layers {
            total += layer.attn.wq.memory_bytes();
            total += layer.attn.wk.memory_bytes();
            total += layer.attn.wv.memory_bytes();
            total += layer.attn.wo.memory_bytes();
            total += layer.ffn.w1.memory_bytes();
            total += layer.ffn.w2.memory_bytes();
            total += layer.ffn.w3.memory_bytes();
        }
        total += self.config.vocab_size * self.config.d_model * 4;
        total += self.config.d_model * 4;
        total += self.config.d_model * self.config.vocab_size * 4;
        total += self.config.n_layers * self.config.d_model * 4 * 2;
        total
    }

    /// Equivalent f32 model memory.
    pub fn f32_equivalent_bytes(&self) -> usize {
        let d = self.config.d_model;
        let ff = self.config.d_ff;
        let v = self.config.vocab_size;
        let n = self.config.n_layers;
        let per_layer = (4 * d * d + 3 * d * ff + 2 * d) * 4;
        let global = (v * d + d * v + d) * 4;
        n * per_layer + global
    }

    /// Compression ratio vs f32.
    pub fn compression_ratio(&self) -> f32 {
        self.f32_equivalent_bytes() as f32 / self.weight_memory_bytes() as f32
    }

    /// Model info string.
    pub fn info(&self) -> String {
        format!(
            "QuatNet 2-bit | d={} heads={} kv_heads={} layers={} ff={} vocab={} | \
             mem={:.1}MB (f32 equiv: {:.1}MB, {:.1}× compression)",
            self.config.d_model,
            self.config.n_heads,
            self.config.n_kv_heads,
            self.config.n_layers,
            self.config.d_ff,
            self.config.vocab_size,
            self.weight_memory_bytes() as f32 / (1024.0 * 1024.0),
            self.f32_equivalent_bytes() as f32 / (1024.0 * 1024.0),
            self.compression_ratio(),
        )
    }
}

fn extract_head(data: &[f32], seq_len: usize, d_model: usize, head: usize, d_head: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * d_head];
    for i in 0..seq_len {
        let src_start = i * d_model + head * d_head;
        let dst_start = i * d_head;
        out[dst_start..dst_start + d_head].copy_from_slice(&data[src_start..src_start + d_head]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Transformer, TransformerConfig};

    #[test]
    fn test_quatnet_from_transformer() {
        let config = TransformerConfig::tiny();
        let model = Transformer::new(config);
        let quatnet = QuatNetTransformer::from_transformer(&model);
        assert_eq!(quatnet.layers.len(), 2);
        assert!(quatnet.compression_ratio() > 1.0);
    }

    #[test]
    fn test_quatnet_forward() {
        let config = TransformerConfig::tiny();
        let model = Transformer::new(config);
        let mut quatnet = QuatNetTransformer::from_transformer(&model);

        let tokens = vec![0, 1, 2, 3];
        let logits = quatnet.forward(&tokens, false).unwrap();
        assert_eq!(logits.shape().dims(), &[4, 256]);

        let data = logits.as_f32_slice().unwrap();
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_quatnet_generate() {
        let config = TransformerConfig::tiny();
        let model = Transformer::new(config);
        let mut quatnet = QuatNetTransformer::from_transformer(&model);

        let prompt = vec![0, 1, 2];
        let output = quatnet.generate(&prompt, 5).unwrap();
        assert_eq!(output.len(), 3 + 5);
        for &t in &output {
            assert!(t < 256);
        }
    }

    #[test]
    fn test_quatnet_compression() {
        let config = TransformerConfig::tiny();
        let model = Transformer::new(config);
        let quatnet = QuatNetTransformer::from_transformer(&model);

        let ratio = quatnet.compression_ratio();
        // Quaternary: 2 bits/param → less compression than ternary but still significant
        assert!(ratio > 1.0, "compression ratio {} too low", ratio);
    }

    #[test]
    fn test_quatnet_info() {
        let config = TransformerConfig::tiny();
        let model = Transformer::new(config);
        let quatnet = QuatNetTransformer::from_transformer(&model);
        let info = quatnet.info();
        assert!(info.contains("QuatNet"));
        assert!(info.contains("compression"));
    }

    #[test]
    fn test_quatnet_kv_cache() {
        let config = TransformerConfig::tiny();
        let model = Transformer::new(config);
        let mut quatnet = QuatNetTransformer::from_transformer(&model);

        let prompt = vec![10, 20, 30];
        let out1 = quatnet.generate(&prompt, 3).unwrap();
        assert_eq!(out1.len(), 6);

        let out2 = quatnet.generate(&prompt, 3).unwrap();
        assert_eq!(out1, out2);
    }
}
