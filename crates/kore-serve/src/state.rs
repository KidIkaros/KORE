//! Shared application state holding the loaded model.

use std::sync::Arc;
use parking_lot::Mutex;
use kore_nn::sampler::{SamplerConfig, Rng};

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
