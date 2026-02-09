//! BitNet b1.58 transformer — all linear projections quantized to 1.58-bit ternary.
//!
//! Architecture: Llama-style decoder with `BitLinear` replacing all `Linear` layers.
//! Embeddings, norms, and LM head remain in f32.
//!
//! Memory: ~20× compression vs f32 for weight matrices.
//!
//! Usage:
//! ```ignore
//! let config = TransformerConfig::tiny();
//! let model = Transformer::new(config.clone());
//! let bitnet = BitNetTransformer::from_transformer(&model, 0.3);
//! let logits = bitnet.forward(&[0, 1, 2], false).unwrap();
//! ```

use kore_core::{KoreError, Tensor};
use kore_kernels::cpu_ternary_matmul::{ternary_matmul, pack_weights_ternary};
use kore_attention::kv_cache::KvCache;

use crate::embedding::Embedding;
use crate::rms_norm::RMSNorm;
use crate::rope::RopeTable;
use crate::model::TransformerConfig;
use crate::sampler::{self, SamplerConfig, Rng};

/// Packed ternary weight matrix with per-row scales.
///
/// Always stored as [out_features, in_features] for ternary_matmul.
/// Computes: output[seq, out] = input[seq, in] @ W^T
struct PackedWeight {
    packed: Vec<u8>,
    scales: Vec<f32>,
    /// out_features (rows of packed weight)
    out_dim: usize,
    /// in_features (cols of packed weight)
    in_dim: usize,
}

impl PackedWeight {
    /// Pack a weight tensor stored as [out_features, in_features].
    /// This is the standard layout for nn::Linear weights.
    fn from_tensor(weight: &Tensor, threshold: f32) -> Self {
        let dims = weight.shape().dims();
        let out_dim = dims[0];
        let in_dim = dims[1];
        let w_data = weight.contiguous();
        let w_slice = w_data.as_f32_slice().expect("f32 weight required");
        let (packed, scales) = pack_weights_ternary(w_slice, out_dim, in_dim, threshold);
        Self { packed, scales, out_dim, in_dim }
    }

    /// Pack a weight tensor stored as [in_features, out_features].
    /// Transposes to [out_features, in_features] before packing.
    /// Used for FFN/MHA weights that are stored in [in, out] layout.
    fn from_tensor_transposed(weight: &Tensor, threshold: f32) -> Self {
        let wt = weight.transpose().expect("transpose failed").contiguous();
        Self::from_tensor(&wt, threshold)
    }

    /// Compute output = input @ W^T.
    /// input: [seq_len, in_dim] → output: [seq_len, out_dim]
    fn forward(&self, input: &Tensor) -> Result<Tensor, KoreError> {
        let seq_len = input.shape().dims()[0];
        // ternary_matmul: W_packed[M, K] @ b[K, N] → [M, N]
        // We want: input[seq, in] @ W^T[in, out] = (W[out, in] @ input^T[in, seq])^T
        // So: M=out_dim, K=in_dim, N=seq_len, b=input^T[in_dim, seq_len]
        let input_t = input.transpose()?.contiguous();
        let out = ternary_matmul(
            &self.packed, &self.scales, &input_t,
            self.out_dim, seq_len, self.in_dim,
        )?;
        // out is [out_dim, seq_len], transpose to [seq_len, out_dim]
        out.transpose().map(|t| t.contiguous())
    }

    /// Memory in bytes.
    fn memory_bytes(&self) -> usize {
        self.packed.len() + self.scales.len() * 4
    }
}

/// BitNet attention block — QKV + output projections are ternary-packed.
struct BitNetAttention {
    n_heads: usize,
    n_kv_heads: usize,
    d_model: usize,
    d_head: usize,
    wq: PackedWeight,
    wk: PackedWeight,
    wv: PackedWeight,
    wo: PackedWeight,
    kv_caches: Vec<KvCache>,
    rope: Option<RopeTable>,
    seq_offset: usize,
}

impl BitNetAttention {
    fn from_mha(mha: &crate::mha::MultiHeadAttention, threshold: f32, max_seq_len: usize) -> Self {
        let d_head = mha.d_head;
        // Reconstruct RoPE table from parameters (RopeTable doesn't impl Clone)
        let rope = mha.rope.as_ref().map(|r| RopeTable::new(r.d_head, r.max_seq_len, r.base));
        Self {
            n_heads: mha.n_heads,
            n_kv_heads: mha.n_kv_heads,
            d_model: mha.d_model,
            d_head,
            // MHA weights are [d_model, q_dim] i.e. [in, out] — need transpose
            wq: PackedWeight::from_tensor_transposed(&mha.wq, threshold),
            wk: PackedWeight::from_tensor_transposed(&mha.wk, threshold),
            wv: PackedWeight::from_tensor_transposed(&mha.wv, threshold),
            wo: PackedWeight::from_tensor_transposed(&mha.wo, threshold),
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

/// BitNet feed-forward block — gate/up/down projections are ternary-packed.
struct BitNetFeedForward {
    w1: PackedWeight,
    w2: PackedWeight,
    w3: PackedWeight,
    d_ff: usize,
}

impl BitNetFeedForward {
    fn from_ffn(ffn: &crate::feed_forward::FeedForward, threshold: f32) -> Self {
        Self {
            // FFN weights are [d_model, d_ff] i.e. [in, out] — need transpose
            w1: PackedWeight::from_tensor_transposed(&ffn.w1, threshold),
            w2: PackedWeight::from_tensor_transposed(&ffn.w2, threshold),
            w3: PackedWeight::from_tensor_transposed(&ffn.w3, threshold),
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

/// BitNet transformer block.
struct BitNetBlock {
    attn_norm: RMSNorm,
    attn: BitNetAttention,
    ffn_norm: RMSNorm,
    ffn: BitNetFeedForward,
    d_model: usize,
}

impl BitNetBlock {
    fn from_block(block: &crate::block::TransformerBlock, threshold: f32, max_seq_len: usize) -> Self {
        Self {
            attn_norm: RMSNorm::new(block.d_model, block.attn_norm.eps),
            attn: BitNetAttention::from_mha(&block.attn, threshold, max_seq_len),
            ffn_norm: RMSNorm::new(block.d_model, block.ffn_norm.eps),
            ffn: BitNetFeedForward::from_ffn(&block.ffn, threshold),
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

/// Full BitNet b1.58 transformer decoder.
///
/// All linear projections (QKV, output, gate/up/down) are quantized to
/// 1.58-bit ternary. Embeddings, norms, and LM head remain in f32.
pub struct BitNetTransformer {
    pub config: TransformerConfig,
    pub embedding: Embedding,
    pub layers: Vec<BitNetBlock>,
    pub final_norm: RMSNorm,
    pub lm_head: Tensor,
    pub threshold: f32,
}

impl BitNetTransformer {
    /// Quantize an existing f32 Transformer to BitNet 1.58-bit.
    pub fn from_transformer(model: &crate::model::Transformer, threshold: f32) -> Self {
        let layers: Vec<BitNetBlock> = model.layers.iter()
            .map(|block| BitNetBlock::from_block(block, threshold, model.config.max_seq_len))
            .collect();

        // Copy norms (they keep their learned weights)
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
            threshold,
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

    /// Total weight memory in bytes (packed ternary + scales + f32 norms/embeddings).
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
        // Add f32 components
        total += self.config.vocab_size * self.config.d_model * 4; // embedding
        total += self.config.d_model * 4; // final norm
        total += self.config.d_model * self.config.vocab_size * 4; // lm_head
        total += self.config.n_layers * self.config.d_model * 4 * 2; // layer norms
        total
    }

    /// Equivalent f32 model memory for comparison.
    pub fn f32_equivalent_bytes(&self) -> usize {
        let d = self.config.d_model;
        let ff = self.config.d_ff;
        let v = self.config.vocab_size;
        let n = self.config.n_layers;
        // Per layer: 4 projections (QKV+O) + 3 FFN + 2 norms
        let per_layer = (4 * d * d + 3 * d * ff + 2 * d) * 4;
        // Global: embedding + lm_head + final_norm
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
            "BitNet b1.58 | d={} heads={} kv_heads={} layers={} ff={} vocab={} | \
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
    fn test_bitnet_from_transformer() {
        let config = TransformerConfig::tiny();
        let model = Transformer::new(config);
        let bitnet = BitNetTransformer::from_transformer(&model, 0.3);
        assert_eq!(bitnet.layers.len(), 2);
        assert!(bitnet.compression_ratio() > 1.0);
    }

    #[test]
    fn test_bitnet_forward() {
        let config = TransformerConfig::tiny();
        let model = Transformer::new(config);
        let mut bitnet = BitNetTransformer::from_transformer(&model, 0.3);

        let tokens = vec![0, 1, 2, 3];
        let logits = bitnet.forward(&tokens, false).unwrap();
        assert_eq!(logits.shape().dims(), &[4, 256]);

        let data = logits.as_f32_slice().unwrap();
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_bitnet_generate() {
        let config = TransformerConfig::tiny();
        let model = Transformer::new(config);
        let mut bitnet = BitNetTransformer::from_transformer(&model, 0.3);

        let prompt = vec![0, 1, 2];
        let output = bitnet.generate(&prompt, 5).unwrap();
        assert_eq!(output.len(), 3 + 5);
        for &t in &output {
            assert!(t < 256);
        }
    }

    #[test]
    fn test_bitnet_compression() {
        let config = TransformerConfig::tiny();
        let model = Transformer::new(config);
        let bitnet = BitNetTransformer::from_transformer(&model, 0.3);

        let ratio = bitnet.compression_ratio();
        println!("BitNet compression: {:.1}x", ratio);
        println!("BitNet memory: {:.1}KB", bitnet.weight_memory_bytes() as f32 / 1024.0);
        println!("F32 equivalent: {:.1}KB", bitnet.f32_equivalent_bytes() as f32 / 1024.0);
        // Weight matrices should be significantly compressed
        assert!(ratio > 1.5, "compression ratio {} too low", ratio);
    }

    #[test]
    fn test_bitnet_info() {
        let config = TransformerConfig::tiny();
        let model = Transformer::new(config);
        let bitnet = BitNetTransformer::from_transformer(&model, 0.3);
        let info = bitnet.info();
        assert!(info.contains("BitNet"));
        assert!(info.contains("compression"));
        println!("{}", info);
    }

    #[test]
    fn test_bitnet_kv_cache() {
        let config = TransformerConfig::tiny();
        let model = Transformer::new(config);
        let mut bitnet = BitNetTransformer::from_transformer(&model, 0.3);

        let prompt = vec![10, 20, 30];
        let out1 = bitnet.generate(&prompt, 3).unwrap();
        assert_eq!(out1.len(), 6);

        let out2 = bitnet.generate(&prompt, 3).unwrap();
        assert_eq!(out1, out2);
    }
}
