//! Layered inference engine for running large models on limited VRAM.
//!
//! Inspired by [AirLLM](https://github.com/lyogavin/airllm), this module loads
//! one transformer layer at a time into GPU/CPU memory, runs the forward pass,
//! and frees memory before moving to the next layer. This enables running 70B+
//! parameter models on hardware with as little as 4GB VRAM.
//!
//! # Architecture
//!
//! ```text
//! Time →
//! Layer N:   [====COMPUTE====]
//! Layer N+1:      [LOAD][DECOMPRESS][GPU_XFER]
//! Layer N+2:                  [LOAD][DECOMPRESS][GPU_XFER]
//! ```
//!
//! # Key components
//!
//! - [`LayeredConfig`] — configuration for sharded inference
//! - [`LayerCache`] — LRU RAM cache for layer weights
//! - [`LayerPrefetcher`] — async double-buffered layer loading
//! - [`shard_model`](sharder::shard_model) — split safetensors into per-layer shards
//! - [`LayeredEngine`] — orchestrates layer-by-layer forward passes

pub mod cache;
pub mod config;
pub mod engine;
pub mod prefetcher;
pub mod sharder;

pub use cache::LayerCache;
pub use config::LayeredConfig;
pub use engine::{KvCache, LayeredEngine};
pub use prefetcher::LayerPrefetcher;
