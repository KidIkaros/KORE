//! # kore-serve
//!
//! Inference server for Kore with OpenAI-compatible REST API.
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
