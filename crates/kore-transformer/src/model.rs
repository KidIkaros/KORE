//! Full transformer decoder model.
//!
//! Transformer = Embedding → N × TransformerBlock → RMSNorm → Linear(vocab)

use kore_core::{KoreError, Tensor};
use crate::embedding::Embedding;
use crate::rms_norm::RMSNorm;
use crate::block::TransformerBlock;
use crate::sampler::{self, SamplerConfig, Rng};

/// Configuration for a transformer model.
#[derive(Clone, Debug)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    /// Number of KV heads for Grouped-Query Attention. Equal to n_heads for MHA.
    pub n_kv_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub norm_eps: f32,
    /// Enable Rotary Position Embeddings (RoPE). Default: true.
    pub use_rope: bool,
    /// RoPE base frequency. Default: 10000.0.
    pub rope_base: f32,
}

impl TransformerConfig {
    /// A tiny config for testing (≈50K params).
    pub fn tiny() -> Self {
        Self {
            vocab_size: 256,
            d_model: 64,
            n_heads: 4,
            n_kv_heads: 4,
            n_layers: 2,
            d_ff: 128,
            max_seq_len: 128,
            norm_eps: 1e-5,
            use_rope: true,
            rope_base: 10000.0,
        }
    }

    /// A small config (~2M params).
    pub fn small() -> Self {
        Self {
            vocab_size: 32000,
            d_model: 256,
            n_heads: 8,
            n_kv_heads: 4,
            n_layers: 4,
            d_ff: 512,
            max_seq_len: 512,
            norm_eps: 1e-5,
            use_rope: true,
            rope_base: 10000.0,
        }
    }
}

/// Full transformer decoder.
pub struct Transformer {
    pub config: TransformerConfig,
    pub embedding: Embedding,
    pub layers: Vec<TransformerBlock>,
    pub final_norm: RMSNorm,
    /// Output projection: [d_model, vocab_size] (tied with embedding weight transposed)
    pub lm_head: Tensor,
}

impl Transformer {
    /// Build a new transformer with random weights.
    pub fn new(config: TransformerConfig) -> Self {
        let embedding = Embedding::new(config.vocab_size, config.d_model, Some(config.max_seq_len));

        let layers: Vec<TransformerBlock> = (0..config.n_layers)
            .map(|_| TransformerBlock::new_full(
                config.d_model, config.n_heads, config.n_kv_heads, config.d_ff, config.norm_eps,
                config.use_rope, config.max_seq_len, config.rope_base,
            ))
            .collect();

        let final_norm = RMSNorm::new(config.d_model, config.norm_eps);

        // LM head: [d_model, vocab_size]
        let scale = (1.0 / config.d_model as f64).sqrt() as f32;
        let lm_data: Vec<f32> = (0..config.d_model * config.vocab_size)
            .map(|i| {
                let x = ((i * 2654435761 + 7) & 0xFFFFFF) as f32 / 0xFFFFFF as f32;
                (x * 2.0 - 1.0) * scale
            })
            .collect();
        let lm_head = Tensor::from_f32(&lm_data, &[config.d_model, config.vocab_size]);

        Self { config, embedding, layers, final_norm, lm_head }
    }

    /// Forward pass: token_ids → logits.
    ///
    /// `token_ids`: &[usize] of length seq_len
    /// `use_cache`: if true, uses KV cache for autoregressive generation
    ///
    /// Returns: [seq_len, vocab_size] logits tensor
    pub fn forward(&mut self, token_ids: &[usize], use_cache: bool) -> Result<Tensor, KoreError> {
        let seq_len = token_ids.len();

        // Embedding lookup
        let mut x = self.embedding.forward(token_ids)?;

        // Causal mask
        let mask = kore_attention::mask::causal_mask(seq_len);

        // Transformer blocks
        for layer in &mut self.layers {
            x = layer.forward(&x, Some(&mask), use_cache)?;
        }

        // Final norm
        x = self.final_norm.forward(&x)?;

        // LM head: x @ lm_head → [seq_len, vocab_size]
        let x_data = x.as_f32_slice().ok_or(KoreError::StorageError("expected f32 tensor".into()))?;
        let lm_data = self.lm_head.as_f32_slice()
            .ok_or(KoreError::StorageError("lm_head: expected f32 tensor".into()))?;
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

    /// Generate tokens autoregressively using greedy decoding.
    ///
    /// `prompt`: initial token IDs
    /// `max_new_tokens`: how many tokens to generate
    ///
    /// Returns: full sequence including prompt
    pub fn generate(&mut self, prompt: &[usize], max_new_tokens: usize) -> Result<Vec<usize>, KoreError> {
        self.generate_with_config(prompt, max_new_tokens, &SamplerConfig::greedy(), &mut Rng::new(42))
    }

    /// Generate tokens autoregressively with configurable sampling.
    ///
    /// `prompt`: initial token IDs
    /// `max_new_tokens`: how many tokens to generate
    /// `sampler_config`: sampling parameters (temperature, top-k, top-p, etc.)
    /// `rng`: random number generator for stochastic sampling
    ///
    /// Returns: full sequence including prompt
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

        // Prefill: process entire prompt
        let logits = self.forward(&tokens, true)?;
        let logits_data = logits.as_f32_slice().ok_or(KoreError::StorageError("expected f32 logits".into()))?;

        // Sample next token from last position
        let last_row = &logits_data[(tokens.len() - 1) * v..tokens.len() * v];
        let next = sampler::sample(last_row, &tokens, sampler_config, rng);
        tokens.push(next);

        // Check EOS
        if sampler_config.eos_token_id == Some(next) {
            return Ok(tokens);
        }

        // Decode: one token at a time
        for _ in 1..max_new_tokens {
            let last_token = *tokens.last()
                .expect("generate: tokens should never be empty during generation");
            let logits = self.forward(&[last_token], true)?;
            let logits_data = logits.as_f32_slice().ok_or(KoreError::StorageError("expected f32 logits".into()))?;
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

    /// Count total parameters.
    pub fn param_count(&self) -> usize {
        let emb = self.config.vocab_size * self.config.d_model;
        let pos = self.config.max_seq_len * self.config.d_model;
        let per_layer = {
            let d = self.config.d_model;
            let ff = self.config.d_ff;
            // QKV + O projections + 2 norms + SwiGLU (W1, W2, W3)
            4 * d * d + 2 * d + 3 * d * ff
        };
        let lm_head = self.config.d_model * self.config.vocab_size;
        let final_norm = self.config.d_model;
        emb + pos + self.config.n_layers * per_layer + lm_head + final_norm
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_forward() {
        let config = TransformerConfig::tiny();
        let mut model = Transformer::new(config);
        let tokens = vec![0, 1, 2, 3];
        let logits = model.forward(&tokens, false).unwrap();
        assert_eq!(logits.shape().dims(), &[4, 256]);
    }

    #[test]
    fn test_transformer_generate() {
        let config = TransformerConfig::tiny();
        let mut model = Transformer::new(config);
        let prompt = vec![0, 1, 2];
        let output = model.generate(&prompt, 5).unwrap();
        assert_eq!(output.len(), 3 + 5); // prompt + generated
        // All tokens should be valid
        for &t in &output {
            assert!(t < 256);
        }
    }

    #[test]
    fn test_transformer_param_count() {
        let config = TransformerConfig::tiny();
        let model = Transformer::new(config);
        let params = model.param_count();
        // Should be reasonable for tiny config
        assert!(params > 10_000);
        assert!(params < 500_000);
        println!("Tiny model params: {}", params);
    }

    #[test]
    fn test_transformer_kv_cache_generation() {
        let config = TransformerConfig::tiny();
        let mut model = Transformer::new(config);

        // Generate with cache
        let prompt = vec![10, 20, 30];
        let out1 = model.generate(&prompt, 3).unwrap();
        assert_eq!(out1.len(), 6);

        // Generate again — should produce same result with same weights
        let out2 = model.generate(&prompt, 3).unwrap();
        assert_eq!(out1, out2);
    }
}
