//! # kore-data
//!
//! Data loading and batching utilities for Kore.
//!
//! Provides:
//! - `MultipackSampler` — pack multiple sequences into one batch to eliminate padding waste
//! - `StreamingDataset` — memory-mapped dataset for large corpora
//! - `TokenBatcher` — efficient token-level batching for LLM training

pub mod batcher;
pub mod dataset;
pub mod multipack;

pub use batcher::TokenBatcher;
pub use dataset::StreamingDataset;
pub use multipack::MultipackSampler;
