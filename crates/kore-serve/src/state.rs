//! Shared application state holding the loaded model.

use std::sync::Arc;
use parking_lot::Mutex;
use kore_transformer::Transformer;

/// Shared state for the inference server.
#[derive(Clone)]
pub struct AppState {
    /// The loaded transformer model (behind a mutex for mutable generate()).
    pub model: Arc<Mutex<Option<Transformer>>>,
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
    pub fn with_model(model: Transformer, name: String) -> Self {
        Self {
            model: Arc::new(Mutex::new(Some(model))),
            model_name: name,
        }
    }

    /// Check if a model is loaded.
    pub fn has_model(&self) -> bool {
        self.model.lock().is_some()
    }
}
