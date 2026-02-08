//! # kore-attention
//!
//! Attention mechanisms and inference primitives for Kore.
//!
//! Provides:
//! - Scaled dot-product attention (baseline)
//! - Flash Attention v2 (tiled, O(n) memory)
//! - KV-cache for autoregressive generation
//! - Causal and sliding window masks
//! - Grouped-query attention (GQA/MQA)

pub mod scaled_dot;
pub mod flash;
pub mod kv_cache;
pub mod mask;
