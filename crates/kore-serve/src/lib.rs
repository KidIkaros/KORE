//! # kore-serve
//!
//! Model-agnostic inference server for Kore with OpenAI-compatible REST API.
//!
//! Any model implementing [`InferenceModel`] can be served. Example:
//! ```rust,no_run
//! use kore_serve::state::{AppState, InferenceModel};
//! // let model = MyModel::load("path/to/weights");
//! // let state = AppState::with_model(model, "my-model".into());
//! // kore_serve::server::serve_with_state("0.0.0.0:8080", state).await;
//! ```
//!
//! Provides:
//! - `/v1/completions` — text completion endpoint
//! - `/v1/chat/completions` — chat completion endpoint
//! - `/health` — health check
//! - SSE streaming for token-by-token generation

pub mod api;
pub mod server;
pub mod health;
pub mod state;

pub use state::InferenceModel;
