//! # kore-transformer
//!
//! End-to-end transformer building blocks:
//! - `Embedding` — token + positional embedding lookup
//! - `RMSNorm` — Root Mean Square Layer Normalization
//! - `MultiHeadAttention` — multi-head self-attention with KV-cache
//! - `FeedForward` — SwiGLU feed-forward network
//! - `TransformerBlock` — single decoder layer (pre-norm)
//! - `Transformer` — full decoder stack with final projection

pub mod embedding;
pub mod rms_norm;
pub mod mha;
pub mod feed_forward;
pub mod block;
pub mod model;
pub mod bitnet;
pub mod quatnet;
pub mod loader;
pub mod rope;
pub mod sampler;
pub mod tokenizer;

pub use embedding::Embedding;
pub use rms_norm::RMSNorm;
pub use mha::MultiHeadAttention;
pub use feed_forward::FeedForward;
pub use block::TransformerBlock;
pub use model::{Transformer, TransformerConfig};
pub use bitnet::BitNetTransformer;
pub use quatnet::QuatNetTransformer;
pub use sampler::{SamplerConfig, Rng};
