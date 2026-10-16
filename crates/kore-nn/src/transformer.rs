//! Transformer reference architecture (validation rig).
//!
//! A minimal transformer-style model proving the teacher-evaluation forward path
//! for the attention/transformer side, mirroring the SqueezeNet CNN smoke test.
//!
//! Architecture: embedding -> mixed-quant linear -> norm -> attention -> head.
//! Loads safetensors directly (per design doc §8 ponytail pass — no .kwe, no converter).

use std::collections::HashMap;
use std::path::Path;

use crate::embedding::Embedding;
use crate::linear::Linear;
use crate::module::Module;
use crate::rms_norm::RMSNorm;
use crate::sampler::SamplerConfig;
use kore_attention::scaled_dot::scaled_dot_product_attention;
use kore_core::Tensor;

/// Lightweight 1-layer Multi-Token Prediction (MTP) draft head (`Qwen3.5` / `DeepSeek-V3` style).
///
/// Combines the target model's pre-head hidden state `h_t` with the newly sampled token's
/// embedding `e(y_t)` to predict draft logits for position `t + 1` in a single matrix-vector product.
pub struct MtpDraftHead {
    proj: Linear,
    norm: RMSNorm,
}

impl MtpDraftHead {
    /// Create a new `MtpDraftHead` for `hidden_dim`.
    pub fn new(hidden_dim: usize) -> Self {
        Self {
            proj: Linear::new(hidden_dim, hidden_dim, false),
            norm: RMSNorm::new(hidden_dim, 1e-6),
        }
    }

    /// Create an `MtpDraftHead` from explicit projection and norm weights.
    pub fn from_weights(
        proj_weight: Tensor,
        proj_bias: Option<Tensor>,
        norm_gamma: Tensor,
    ) -> Self {
        Self {
            proj: Linear::from_weight(proj_weight, proj_bias),
            norm: RMSNorm::from_weight(norm_gamma, 1e-6),
        }
    }

    /// Move draft head parameters to the specified device.
    pub fn to(&mut self, device: kore_core::Device) -> Result<(), kore_core::KoreError> {
        self.proj.to(device)?;
        self.norm.to(device)?;
        Ok(())
    }

    /// Compute draft logits `[1, vocab_size]` from target hidden state `hidden_last` (`[1, hidden_dim]`)
    /// and the newly sampled `token_id`.
    pub fn forward_draft(
        &self,
        hidden_last: &Tensor,
        token_id: usize,
        embedding: &Embedding,
        lm_head: &Linear,
    ) -> kore_core::Result<Tensor> {
        let tok_t = Tensor::from_f32(&[token_id as f32], &[1]);
        let mut tok_emb = embedding.forward(&tok_t)?;
        if !hidden_last.device().is_cpu() {
            tok_emb = tok_emb.to(hidden_last.device())?;
        }
        let combined = hidden_last.add(&tok_emb)?;
        let h_draft = self.norm.forward(&self.proj.forward(&combined)?)?;
        lm_head.forward(&h_draft)
    }
}

/// Result of a single Multi-Token Prediction (MTP) speculative verification step.
#[derive(Debug, Clone)]
pub struct SpeculativeStepResult {
    /// Ground-truth token sampled from the target model at position `pos` (after `pending_token`).
    pub verified_token: usize,
    /// Bonus token sampled from the target model at position `pos + 1` when `verified_token == draft_token`.
    pub bonus_token: Option<usize>,
    /// Next draft token proposed by the `MtpDraftHead` for the subsequent speculative step.
    pub next_draft: Option<usize>,
    /// Whether the speculative `draft_token` was accepted by the target model.
    pub draft_accepted: bool,
}

/// A minimal transformer reference architecture (validation rig).
///
/// Architecture:
/// ```text
/// token_ids -> Embedding -> MixedQuantLinear -> RMSNorm -> SelfAttention -> RMSNorm -> HeadLinear -> logits
/// ```
///
/// This is a VALIDATION RIG, not a committed architecture. It proves the
/// teacher-evaluation forward path exists for a transformer-style arch next to
/// the teacher (SqueezeNet already proves the CNN path).
pub struct Transformer {
    /// Token embedding table.
    embedding: Embedding,
    /// Mixed-quant (ternary/quaternary) linear projection after embedding.
    mixed_quant_linear: Linear,
    /// Pre-attention RMSNorm.
    pre_attn_norm: RMSNorm,
    /// Post-attention RMSNorm.
    post_attn_norm: RMSNorm,
    /// Output head linear projecting to vocab size.
    head: Linear,
    /// Optional grafted 1-layer Multi-Token Prediction (MTP) draft head.
    mtp_head: Option<MtpDraftHead>,
    /// Vocabulary size.
    vocab_size: usize,
    /// Hidden dimension.
    hidden_dim: usize,
    /// Attention head dimension.
    head_dim: usize,
    /// Target execution device.
    device: kore_core::Device,
}

impl Transformer {
    /// Create a new Transformer reference architecture.
    ///
    /// # Arguments
    /// * `vocab_size` - Number of tokens in the vocabulary
    /// * `hidden_dim` - Hidden/embedding dimension
    /// * `head_dim` - Per-head attention dimension (must divide hidden_dim)
    pub fn new(vocab_size: usize, hidden_dim: usize, head_dim: usize) -> Self {
        assert_eq!(
            hidden_dim % head_dim,
            0,
            "hidden_dim must be divisible by head_dim"
        );

        Self {
            embedding: Embedding::new(vocab_size, hidden_dim),
            mixed_quant_linear: Linear::new(hidden_dim, hidden_dim, true),
            pre_attn_norm: RMSNorm::new(hidden_dim, 1e-6),
            post_attn_norm: RMSNorm::new(hidden_dim, 1e-6),
            head: Linear::new(hidden_dim, vocab_size, true),
            mtp_head: None,
            vocab_size,
            hidden_dim,
            head_dim,
            device: kore_core::Device::Cpu,
        }
    }

    /// Create from safetensors (state dict loaded from a .safetensors file).
    ///
    /// Expected keys (kore-native format, mirroring SqueezeNet pattern):
    /// - `embedding.weight`       — [vocab_size, hidden_dim]
    /// - `mixed_quant.linear.weight` — [hidden_dim, hidden_dim]
    /// - `mixed_quant.linear.bias`   — [hidden_dim]
    /// - `pre_attn_norm.gamma`   — [hidden_dim]
    /// - `post_attn_norm.gamma`  — [hidden_dim]
    /// - `head.weight`           — [vocab_size, hidden_dim]
    /// - `head.bias`             — [vocab_size]
    /// Optional MTP draft head keys:
    /// - `mtp.proj.weight`       — [hidden_dim, hidden_dim]
    /// - `mtp.proj.bias`         — [hidden_dim]
    /// - `mtp.norm.gamma`        — [hidden_dim]
    pub fn from_state_dict(
        sd: &HashMap<String, Tensor>,
        head_dim: usize,
    ) -> Result<Self, kore_core::KoreError> {
        let get = |key: &str| -> Result<Tensor, kore_core::KoreError> {
            sd.get(key)
                .cloned()
                .ok_or_else(|| kore_core::KoreError::StorageError(format!("missing key: {}", key)))
        };
        let get_opt = |key: &str| -> Option<Tensor> { sd.get(key).cloned() };

        let emb_w = get("embedding.weight")?;
        let hidden_dim = emb_w.shape().dims()[1];
        assert_eq!(
            hidden_dim % head_dim,
            0,
            "hidden_dim {} not divisible by head_dim {}",
            hidden_dim,
            head_dim
        );

        let embedding = Embedding::from_weight(emb_w);

        let mq_linear = Linear::from_weight(
            get("mixed_quant.linear.weight")?,
            get_opt("mixed_quant.linear.bias"),
        );

        let pre_attn_norm = RMSNorm::from_weight(get("pre_attn_norm.gamma")?, 1e-6);
        let post_attn_norm = RMSNorm::from_weight(get("post_attn_norm.gamma")?, 1e-6);

        let vocab_size = sd
            .get("head.weight")
            .map(|t| t.shape().dims()[0])
            .unwrap_or(hidden_dim);
        let head = Linear::from_weight(get("head.weight")?, get_opt("head.bias"));

        let mtp_head = if let (Some(proj_w), Some(norm_g)) =
            (get_opt("mtp.proj.weight"), get_opt("mtp.norm.gamma"))
        {
            Some(MtpDraftHead::from_weights(
                proj_w,
                get_opt("mtp.proj.bias"),
                norm_g,
            ))
        } else {
            None
        };

        Ok(Self {
            embedding,
            mixed_quant_linear: mq_linear,
            pre_attn_norm,
            post_attn_norm,
            head,
            mtp_head,
            vocab_size,
            hidden_dim,
            head_dim,
            device: kore_core::Device::Cpu,
        })
    }

    /// Load from a safetensors file directly (no .kwe, no converter).
    ///
    /// Uses `load_state_dict` from kore-nn serialization — identical to SqueezeNet path.
    pub fn load_safetensors(path: &Path, head_dim: usize) -> Result<Self, kore_core::KoreError> {
        let sd = crate::serialization::load_state_dict(path)?;
        Self::from_state_dict(&sd, head_dim)
    }

    /// Enable or attach a 1-layer Multi-Token Prediction (MTP) draft head for speculative decoding.
    pub fn enable_mtp_head(&mut self) -> Result<(), kore_core::KoreError> {
        if self.mtp_head.is_none() {
            let mut mtp = MtpDraftHead::new(self.hidden_dim);
            if !self.device.is_cpu() {
                mtp.to(self.device)?;
            }
            self.mtp_head = Some(mtp);
        }
        Ok(())
    }

    /// Attach a custom `MtpDraftHead`.
    pub fn set_mtp_head(&mut self, mut mtp: MtpDraftHead) -> Result<(), kore_core::KoreError> {
        if !self.device.is_cpu() {
            mtp.to(self.device)?;
        }
        self.mtp_head = Some(mtp);
        Ok(())
    }

    /// Whether an `MtpDraftHead` is attached to this model.
    pub fn has_mtp_head(&self) -> bool {
        self.mtp_head.is_some()
    }

    /// Move entire model parameters to the target device.
    pub fn to(&mut self, device: kore_core::Device) -> Result<(), kore_core::KoreError> {
        self.mixed_quant_linear.to(device)?;
        self.pre_attn_norm.to(device)?;
        self.post_attn_norm.to(device)?;
        self.head.to(device)?;
        if let Some(ref mut mtp) = self.mtp_head {
            mtp.to(device)?;
        }
        self.device = device;
        Ok(())
    }

    /// Current execution device.
    pub fn device(&self) -> kore_core::Device {
        self.device
    }

    /// Number of output classes (vocab size).
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Attention head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Count total trainable parameters.
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|t| t.numel()).sum()
    }

    /// Sample a token from the logits using the given config.
    pub fn sample_token(&self, logits: &[f32], config: &SamplerConfig, rng_seed: u64) -> usize {
        let mut rng = crate::sampler::Rng::new(rng_seed);
        crate::sampler::sample(logits, &[], config, &mut rng)
    }

    /// Incremental forward pass using a KV-cache, returning `(last_logits [1, vocab_size], last_hidden [1, hidden_dim])`.
    pub fn forward_with_cache_and_hidden(
        &self,
        input: &Tensor,
        kv_cache: &mut kore_attention::kv_cache::KvCache,
    ) -> kore_core::Result<(Tensor, Tensor)> {
        // 1. Embedding: [new_tokens] -> [new_tokens, hidden_dim]
        let mut embedded = self.embedding.forward(input)?;
        if !self.device.is_cpu() {
            embedded = embedded.to(self.device)?;
        }

        // 2. Mixed-quant linear projection: [new_tokens, hidden_dim] -> [new_tokens, hidden_dim]
        let x = self.mixed_quant_linear.forward(&embedded)?;

        // 3. Pre-attention RMSNorm
        let x = self.pre_attn_norm.forward(&x)?;

        // 4. Append new K, V to KV-cache: k_full, v_full have shape [total_seq_len, hidden_dim]
        let (k_full, v_full) = kv_cache.update(&x, &x)?;

        // 5. Extract only the last query row [1, hidden_dim].
        let new_tokens = x.shape().dims()[0];
        let q_last = x.slice_rows(new_tokens - 1, 1)?;
        let attn_out = scaled_dot_product_attention(&q_last, &k_full, &v_full, None, None)?;

        // 6. Post-attention RMSNorm on [1, hidden_dim]
        let x_post = self.post_attn_norm.forward(&attn_out)?;

        // 7. Project only the last token to vocabulary logits: [1, hidden_dim] -> [1, vocab_size]
        let logits = self.head.forward(&x_post)?;
        Ok((logits, x_post))
    }

    /// Incremental forward pass using a KV-cache, returning only the last token's logits `[1, vocab_size]`.
    ///
    /// - During prefill, pass `input` of shape `[prompt_len]`. All prompt K/V vectors are appended to `kv_cache`.
    /// - During decode, pass `input` of shape `[1]` (the newly sampled token). Only the new token's K/V is
    ///   appended to `kv_cache`, and attention + output projection run with `M = 1`.
    pub fn forward_with_cache(
        &self,
        input: &Tensor,
        kv_cache: &mut kore_attention::kv_cache::KvCache,
    ) -> kore_core::Result<Tensor> {
        let (logits, _hidden) = self.forward_with_cache_and_hidden(input, kv_cache)?;
        Ok(logits)
    }

    /// Propose a speculative draft token using the attached `MtpDraftHead` (if present).
    pub fn draft_token(
        &self,
        hidden_last: &Tensor,
        sampled_token: usize,
        config: &SamplerConfig,
        seed: u64,
        pos: usize,
    ) -> kore_core::Result<Option<usize>> {
        if let Some(ref mtp) = self.mtp_head {
            let draft_logits =
                mtp.forward_draft(hidden_last, sampled_token, &self.embedding, &self.head)?;
            let tok = crate::sampler::sample_from_tensor(&draft_logits, &[], config, seed, pos)?;
            Ok(Some(tok))
        } else {
            Ok(None)
        }
    }

    /// Execute one speculative decoding verification step (`M = 2` small-batch forward pass).
    ///
    /// Feeds `[pending_token, draft_token]` through the target model in a single batch-invariant
    /// pass (`M = 2`):
    /// - If `verified_token == draft_token`, both tokens are accepted in one step (`bonus_token = Some(...)`).
    /// - If `verified_token != draft_token`, `kv_cache` is rolled back by 1 token via `truncate` and
    ///   `verified_token` is returned as the ground-truth next token.
    pub fn step_speculative(
        &self,
        pending_token: usize,
        draft_token: usize,
        kv_cache: &mut kore_attention::kv_cache::KvCache,
        config: &SamplerConfig,
        seed: u64,
        pos: usize,
    ) -> kore_core::Result<SpeculativeStepResult> {
        let base_seq_len = kv_cache.len();
        let batch_input = Tensor::from_f32(&[pending_token as f32, draft_token as f32], &[2]);

        let mut embedded = self.embedding.forward(&batch_input)?;
        if !self.device.is_cpu() {
            embedded = embedded.to(self.device)?;
        }

        let x = self.mixed_quant_linear.forward(&embedded)?;
        let x = self.pre_attn_norm.forward(&x)?;
        let (k_full, v_full) = kv_cache.update(&x, &x)?;

        let total_seq = k_full.shape().dims()[0];
        let scale_factor = 1.0 / (self.hidden_dim as f32).sqrt();

        // Causal attention for the 2 queries at positions `base_seq_len` and `base_seq_len + 1`.
        let attn_out = if !x.device().is_cpu() && self.hidden_dim <= 256 {
            if let Some(backend) = kore_core::device::get_backend(x.device()) {
                // Negative scale signals causal horizon `max_t = (seq_k - seq_q) + q_row + 1` in flash_decode.wgsl
                backend.fused_attention_decode(&x, &k_full, &v_full, -scale_factor)?
            } else {
                let mut mask_data = vec![0.0f32; 2 * total_seq];
                mask_data[total_seq - 1] = f32::NEG_INFINITY;
                let mask = Tensor::from_f32(&mask_data, &[2, total_seq]).to(x.device())?;
                scaled_dot_product_attention(&x, &k_full, &v_full, Some(&mask), None)?
            }
        } else {
            let mut mask_data = vec![0.0f32; 2 * total_seq];
            mask_data[total_seq - 1] = f32::NEG_INFINITY;
            let mut mask = Tensor::from_f32(&mask_data, &[2, total_seq]);
            if !x.device().is_cpu() {
                mask = mask.to(x.device())?;
            }
            scaled_dot_product_attention(&x, &k_full, &v_full, Some(&mask), None)?
        };

        let x_post = self.post_attn_norm.forward(&attn_out)?;
        let logits_2 = self.head.forward(&x_post)?;

        // Sample target tokens from row 0 (position `pos`) and row 1 (position `pos + 1`)
        let (target_tok_0, target_tok_1) = if config.temperature <= 0.0
            && config.repetition_penalty == 1.0
            && !logits_2.device().is_cpu()
        {
            let indices = logits_2.argmax_tokens()?;
            (indices[0] as usize, indices[1] as usize)
        } else {
            let logits_cpu = logits_2.cpu()?;
            let slice = logits_cpu.as_f32_slice().unwrap();
            let row0 = &slice[..self.vocab_size];
            let row1 = &slice[self.vocab_size..2 * self.vocab_size];
            let t0 = crate::sampler::sample_with_pos(row0, &[], config, seed, pos);
            let t1 =
                crate::sampler::sample_with_pos(row1, &[], config, seed.wrapping_add(1), pos + 1);
            (t0, t1)
        };

        if target_tok_0 == draft_token {
            // Draft accepted! Keep both KV entries (`base_seq_len + 2`).
            let h1 = x_post.slice_rows(1, 1)?;
            let next_draft =
                self.draft_token(&h1, target_tok_1, config, seed.wrapping_add(2), pos + 2)?;
            Ok(SpeculativeStepResult {
                verified_token: target_tok_0,
                bonus_token: Some(target_tok_1),
                next_draft,
                draft_accepted: true,
            })
        } else {
            // Draft rejected! Roll back the draft token's KV entry (`base_seq_len + 1`).
            kv_cache.truncate(base_seq_len + 1);
            let h0 = x_post.slice_rows(0, 1)?;
            let next_draft =
                self.draft_token(&h0, target_tok_0, config, seed.wrapping_add(1), pos + 1)?;
            Ok(SpeculativeStepResult {
                verified_token: target_tok_0,
                bonus_token: None,
                next_draft,
                draft_accepted: false,
            })
        }
    }
}

impl Module for Transformer {
    /// Forward pass.
    ///
    /// Input: `[seq_len]` token IDs (f32 tensor of integer IDs).
    /// Output: `[seq_len, vocab_size]` logits.
    fn forward(&self, input: &Tensor) -> kore_core::Result<Tensor> {
        // 1. Embedding: [seq_len] -> [seq_len, hidden_dim]
        let mut embedded = self.embedding.forward(input)?;
        if !self.device.is_cpu() {
            embedded = embedded.to(self.device)?;
        }

        // 2. Mixed-quant linear projection: [seq_len, hidden_dim] -> [seq_len, hidden_dim]
        let x = self.mixed_quant_linear.forward(&embedded)?;

        // 3. Pre-attention RMSNorm
        let x = self.pre_attn_norm.forward(&x)?;

        // 4. Self-attention (causal). Q, K, V all come from the same hidden states.
        //    Input shape: [seq_len, hidden_dim] -> 2D attention (no batch dim).
        let seq_len = x.shape().dims()[0];
        let q = x.clone();
        let k = x.clone();
        let v = x.clone();

        // Causal mask: [seq_len, seq_len]
        let causal_mask: Vec<f32> = (0..seq_len)
            .flat_map(move |i| {
                (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 })
            })
            .collect();
        let mut mask = Tensor::from_f32(&causal_mask, &[seq_len, seq_len]);
        if !self.device.is_cpu() {
            mask = mask.to(self.device)?;
        }

        let attn_out = scaled_dot_product_attention(&q, &k, &v, Some(&mask), None)?;

        // 5. Post-attention RMSNorm
        let x = self.post_attn_norm.forward(&attn_out)?;

        // 6. Head: [seq_len, hidden_dim] -> [seq_len, vocab_size]
        self.head.forward(&x)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.embedding.parameters();
        params.extend(self.mixed_quant_linear.parameters());
        params.extend(self.pre_attn_norm.parameters());
        params.extend(self.post_attn_norm.parameters());
        params.extend(self.head.parameters());
        params
    }

    fn named_parameters(&self) -> Vec<(&str, &Tensor)> {
        let mut params = Vec::new();
        for (_name, t) in self.embedding.named_parameters() {
            params.push(("embedding.weight", t));
        }
        for (name, t) in self.mixed_quant_linear.named_parameters() {
            let name: &str = Box::leak(format!("mixed_quant.linear.{}", name).into_boxed_str());
            params.push((name, t));
        }
        for (name, t) in self.pre_attn_norm.named_parameters() {
            let name: &str = Box::leak(format!("pre_attn_norm.{}", name).into_boxed_str());
            params.push((name, t));
        }
        for (name, t) in self.post_attn_norm.named_parameters() {
            let name: &str = Box::leak(format!("post_attn_norm.{}", name).into_boxed_str());
            params.push((name, t));
        }
        for (name, t) in self.head.named_parameters() {
            let name: &str = Box::leak(format!("head.{}", name).into_boxed_str());
            params.push((name, t));
        }
        params
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();
        for (name, t) in self.embedding.named_parameters() {
            sd.insert(format!("embedding.{}", name), t.clone());
        }
        for (name, t) in self.mixed_quant_linear.named_parameters() {
            sd.insert(format!("mixed_quant.linear.{}", name), t.clone());
        }
        for (name, t) in self.pre_attn_norm.named_parameters() {
            sd.insert(format!("pre_attn_norm.{}", name), t.clone());
        }
        for (name, t) in self.post_attn_norm.named_parameters() {
            sd.insert(format!("post_attn_norm.{}", name), t.clone());
        }
        for (name, t) in self.head.named_parameters() {
            sd.insert(format!("head.{}", name), t.clone());
        }
        sd
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a random tensor of given shape.
    fn rand_tensor(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01) % 1.0).collect();
        Tensor::from_f32(&data, shape)
    }

    #[test]
    fn test_transformer_construction() {
        let model = Transformer::new(100, 64, 16);
        assert_eq!(model.vocab_size(), 100);
        assert_eq!(model.hidden_dim(), 64);
        let params = model.parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_transformer_forward_shape() {
        let model = Transformer::new(50, 32, 8);
        // [seq_len=4] token IDs
        let input = Tensor::from_f32(&[0.0, 1.0, 2.0, 3.0], &[4]);
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[4, 50]);
        let data = output.as_f32_slice().unwrap();
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_transformer_forward_longer_seq() {
        let model = Transformer::new(100, 64, 16);
        let input = Tensor::from_f32(&[0.0, 5.0, 10.0, 15.0, 20.0], &[5]);
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[5, 100]);
    }

    #[test]
    fn test_transformer_from_state_dict_and_forward() {
        let mut sd = HashMap::new();
        let vocab = 50;
        let hidden = 32;
        let head_dim = 8;

        sd.insert("embedding.weight".into(), rand_tensor(&[vocab, hidden]));
        sd.insert(
            "mixed_quant.linear.weight".into(),
            rand_tensor(&[hidden, hidden]),
        );
        sd.insert("mixed_quant.linear.bias".into(), rand_tensor(&[hidden]));
        sd.insert("pre_attn_norm.gamma".into(), rand_tensor(&[hidden]));
        sd.insert("post_attn_norm.gamma".into(), rand_tensor(&[hidden]));
        sd.insert("head.weight".into(), rand_tensor(&[vocab, hidden]));
        sd.insert("head.bias".into(), rand_tensor(&[vocab]));

        let model = Transformer::from_state_dict(&sd, head_dim).unwrap();
        assert_eq!(model.vocab_size(), vocab);

        let input = Tensor::from_f32(&[0.0, 1.0, 2.0], &[3]);
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[3, vocab]);
        let data = output.as_f32_slice().unwrap();
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_transformer_sample_token() {
        let model = Transformer::new(10, 16, 8);
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1];
        let config = SamplerConfig::greedy();
        let token = model.sample_token(&logits, &config, 42);
        assert_eq!(token, 3); // argmax of logits
    }

    #[test]
    fn test_transformer_state_dict_keys() {
        let model = Transformer::new(100, 32, 8);
        let sd = model.state_dict();
        assert!(sd.contains_key("embedding.weight"));
        assert!(sd.contains_key("mixed_quant.linear.weight"));
        assert!(sd.contains_key("mixed_quant.linear.bias"));
        assert!(sd.contains_key("pre_attn_norm.gamma"));
        assert!(sd.contains_key("post_attn_norm.gamma"));
        assert!(sd.contains_key("head.weight"));
        assert!(sd.contains_key("head.bias"));
    }

    #[test]
    fn test_transformer_head_dim_divisibility() {
        let model = Transformer::new(100, 64, 16);
        assert_eq!(model.hidden_dim() % 16, 0);
    }

    #[test]
    fn test_transformer_forward_with_cache_matches_full_forward() {
        let mut sd = HashMap::new();
        let vocab = 50;
        let hidden = 32;
        let head_dim = 8;

        sd.insert("embedding.weight".into(), rand_tensor(&[vocab, hidden]));
        sd.insert(
            "mixed_quant.linear.weight".into(),
            rand_tensor(&[hidden, hidden]),
        );
        sd.insert("mixed_quant.linear.bias".into(), rand_tensor(&[hidden]));
        sd.insert("pre_attn_norm.gamma".into(), rand_tensor(&[hidden]));
        sd.insert("post_attn_norm.gamma".into(), rand_tensor(&[hidden]));
        sd.insert("head.weight".into(), rand_tensor(&[vocab, hidden]));
        sd.insert("head.bias".into(), rand_tensor(&[vocab]));

        let model = Transformer::from_state_dict(&sd, head_dim).unwrap();
        let mut kv_cache = kore_attention::kv_cache::KvCache::new(hidden, hidden, 32);

        // Prefill with 3 tokens
        let prompt = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        let cached_prefill = model.forward_with_cache(&prompt, &mut kv_cache).unwrap();
        let full_prefill = model.forward(&prompt).unwrap();
        let cached_prefill_slice = cached_prefill.as_f32_slice().unwrap();
        let full_prefill_last = &full_prefill.as_f32_slice().unwrap()[2 * vocab..3 * vocab];
        for (a, b) in cached_prefill_slice.iter().zip(full_prefill_last.iter()) {
            assert!((a - b).abs() < 1e-5, "prefill mismatch: {} vs {}", a, b);
        }

        // Decode step 1: append token 4.0
        let next_tok = Tensor::from_f32(&[4.0], &[1]);
        let cached_step1 = model.forward_with_cache(&next_tok, &mut kv_cache).unwrap();
        let full_seq1 = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let full_out1 = model.forward(&full_seq1).unwrap();
        let cached_step1_slice = cached_step1.as_f32_slice().unwrap();
        let full_out1_last = &full_out1.as_f32_slice().unwrap()[3 * vocab..4 * vocab];
        for (a, b) in cached_step1_slice.iter().zip(full_out1_last.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "decode step 1 mismatch: {} vs {}",
                a,
                b
            );
        }

        // Decode step 2: append token 7.0
        let next_tok2 = Tensor::from_f32(&[7.0], &[1]);
        let cached_step2 = model.forward_with_cache(&next_tok2, &mut kv_cache).unwrap();
        let full_seq2 = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 7.0], &[5]);
        let full_out2 = model.forward(&full_seq2).unwrap();
        let cached_step2_slice = cached_step2.as_f32_slice().unwrap();
        let full_out2_last = &full_out2.as_f32_slice().unwrap()[4 * vocab..5 * vocab];
        for (a, b) in cached_step2_slice.iter().zip(full_out2_last.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "decode step 2 mismatch: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_transformer_speculative_decoding_exact_equivalence() {
        let mut model = Transformer::new(32, 32, 8);
        model.enable_mtp_head().unwrap();
        let config = SamplerConfig::greedy();

        // 1. Standard autoregressive reference decode for 6 new tokens
        let prompt = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        let mut ref_cache = kore_attention::kv_cache::KvCache::new(32, 32, 32);
        let ref_logits0 = model.forward_with_cache(&prompt, &mut ref_cache).unwrap();
        let mut ref_tok =
            crate::sampler::sample_from_tensor(&ref_logits0, &[], &config, 42, 0).unwrap();
        let mut ref_tokens = vec![ref_tok];
        for step in 1..6 {
            let t = Tensor::from_f32(&[ref_tok as f32], &[1]);
            let logits = model.forward_with_cache(&t, &mut ref_cache).unwrap();
            ref_tok =
                crate::sampler::sample_from_tensor(&logits, &[], &config, 42 + step as u64, step)
                    .unwrap();
            ref_tokens.push(ref_tok);
        }

        // 2. Speculative decoding with a deliberate mix of accepted (matching ref_tokens) and rejected drafts
        let mut spec_cache = kore_attention::kv_cache::KvCache::new(32, 32, 32);
        let (spec_logits0, _h0) = model
            .forward_with_cache_and_hidden(&prompt, &mut spec_cache)
            .unwrap();
        let mut cur_tok =
            crate::sampler::sample_from_tensor(&spec_logits0, &[], &config, 42, 0).unwrap();
        let mut spec_tokens = vec![cur_tok];

        // Step 1: pass exact ground-truth token `ref_tokens[1]` as draft -> MUST be accepted and yield bonus token `ref_tokens[2]`!
        let step1 = model
            .step_speculative(cur_tok, ref_tokens[1], &mut spec_cache, &config, 43, 1)
            .unwrap();
        assert!(step1.draft_accepted, "Exact draft should be accepted");
        assert_eq!(step1.verified_token, ref_tokens[1]);
        assert_eq!(step1.bonus_token, Some(ref_tokens[2]));
        spec_tokens.push(step1.verified_token);
        spec_tokens.push(step1.bonus_token.unwrap());
        cur_tok = step1.bonus_token.unwrap();

        // Step 2: pass wrong draft `(ref_tokens[3] + 1) % 32` -> MUST be rejected, truncate cache, and still recover `ref_tokens[3]`!
        let wrong_draft = (ref_tokens[3] + 1) % 32;
        let step2 = model
            .step_speculative(cur_tok, wrong_draft, &mut spec_cache, &config, 45, 3)
            .unwrap();
        assert!(!step2.draft_accepted, "Wrong draft should be rejected");
        assert_eq!(step2.verified_token, ref_tokens[3]);
        assert_eq!(step2.bonus_token, None);
        spec_tokens.push(step2.verified_token);
        cur_tok = step2.verified_token;

        // Step 3: pass exact ground-truth token `ref_tokens[4]` as draft -> MUST be accepted and yield `ref_tokens[5]`!
        let step3 = model
            .step_speculative(cur_tok, ref_tokens[4], &mut spec_cache, &config, 46, 4)
            .unwrap();
        assert!(
            step3.draft_accepted,
            "Exact draft after rollback should be accepted"
        );
        assert_eq!(step3.verified_token, ref_tokens[4]);
        assert_eq!(step3.bonus_token, Some(ref_tokens[5]));
        spec_tokens.push(step3.verified_token);
        spec_tokens.push(step3.bonus_token.unwrap());

        assert_eq!(spec_tokens, ref_tokens);
    }
}
