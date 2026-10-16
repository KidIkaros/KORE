//! Shared application state holding the loaded model.

use kore_nn::sampler::{Rng, SamplerConfig};
use parking_lot::Mutex;
use std::sync::Arc;

/// Trait for any model that can perform autoregressive generation.
///
/// Implement this for your model type (Transformer, Mamba, etc.) to use
/// with the Kore inference server.
pub trait InferenceModel: Send + 'static {
    /// Generate tokens autoregressively from a prompt.
    ///
    /// Returns the full sequence (prompt + generated tokens).
    fn generate_with_config(
        &mut self,
        prompt_tokens: &[usize],
        max_tokens: usize,
        config: &SamplerConfig,
        rng: &mut Rng,
    ) -> Result<Vec<usize>, String>;
}

impl InferenceModel for kore_nn::Transformer {
    fn generate_with_config(
        &mut self,
        prompt_tokens: &[usize],
        max_tokens: usize,
        config: &SamplerConfig,
        rng: &mut Rng,
    ) -> Result<Vec<usize>, String> {
        let actual_hidden = self.hidden_dim();
        let max_cache_len = prompt_tokens.len() + max_tokens + 32;
        let mut kv_cache =
            kore_attention::kv_cache::KvCache::new(actual_hidden, actual_hidden, max_cache_len);

        let input_f32: Vec<f32> = prompt_tokens.iter().map(|&id| id as f32).collect();
        let input_tensor = kore_core::Tensor::from_f32(&input_f32, &[input_f32.len()]);
        let (logits, _hidden) = self
            .forward_with_cache_and_hidden(&input_tensor, &mut kv_cache)
            .map_err(|e| e.to_string())?;

        let next_tok = kore_nn::sampler::sample_from_tensor(
            &logits,
            &[],
            config,
            rng.next_u64(),
            prompt_tokens.len(),
        )
        .map_err(|e| e.to_string())?;

        let mut output = prompt_tokens.to_vec();
        output.push(next_tok);

        let mut pending_token = next_tok;
        for step in 1..max_tokens {
            let step_input = kore_core::Tensor::from_f32(&[pending_token as f32], &[1]);
            let (step_logits, _h) = self
                .forward_with_cache_and_hidden(&step_input, &mut kv_cache)
                .map_err(|e| e.to_string())?;
            let cur_tok = kore_nn::sampler::sample_from_tensor(
                &step_logits,
                &output,
                config,
                rng.next_u64(),
                prompt_tokens.len() + step,
            )
            .map_err(|e| e.to_string())?;
            output.push(cur_tok);
            pending_token = cur_tok;
        }

        Ok(output)
    }
}

/// Shared state for the inference server.
#[derive(Clone)]
pub struct AppState {
    /// The loaded model (behind a mutex for mutable generate()).
    pub model: Arc<Mutex<Option<Box<dyn InferenceModel>>>>,
    /// Model name / path for API responses.
    pub model_name: String,
}

impl AppState {
    /// Create state with no model loaded (placeholder mode).
    pub fn empty() -> Self {
        Self {
            model: Arc::new(Mutex::new(None)),
            model_name: "none".to_string(),
        }
    }

    /// Create state with a loaded model.
    pub fn with_model(model: impl InferenceModel, name: String) -> Self {
        Self {
            model: Arc::new(Mutex::new(Some(Box::new(model)))),
            model_name: name,
        }
    }

    /// Check if a model is loaded.
    pub fn has_model(&self) -> bool {
        self.model.lock().is_some()
    }
}
