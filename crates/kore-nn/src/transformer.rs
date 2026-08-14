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
    /// Vocabulary size.
    vocab_size: usize,
    /// Hidden dimension.
    hidden_dim: usize,
    /// Attention head dimension.
    head_dim: usize,
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
            vocab_size,
            hidden_dim,
            head_dim,
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

        Ok(Self {
            embedding,
            mixed_quant_linear: mq_linear,
            pre_attn_norm,
            post_attn_norm,
            head,
            vocab_size,
            hidden_dim,
            head_dim,
        })
    }

    /// Load from a safetensors file directly (no .kwe, no converter).
    ///
    /// Uses `load_state_dict` from kore-nn serialization — identical to SqueezeNet path.
    pub fn load_safetensors(path: &Path, head_dim: usize) -> Result<Self, kore_core::KoreError> {
        let sd = crate::serialization::load_state_dict(path)?;
        Self::from_state_dict(&sd, head_dim)
    }

    /// Number of output classes (vocab size).
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
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
}

impl Module for Transformer {
    /// Forward pass.
    ///
    /// Input: `[seq_len]` token IDs (f32 tensor of integer IDs).
    /// Output: `[seq_len, vocab_size]` logits.
    fn forward(&self, input: &Tensor) -> kore_core::Result<Tensor> {
        // 1. Embedding: [seq_len] -> [seq_len, hidden_dim]
        let embedded = self.embedding.forward(input)?;

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
                (0..seq_len).map(move |j| {
                    if j > i {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        let mask = Tensor::from_f32(&causal_mask, &[seq_len, seq_len]);

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
        for (name, t) in self.embedding.named_parameters() {
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
        sd.insert("mixed_quant.linear.weight".into(), rand_tensor(&[hidden, hidden]));
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
}
