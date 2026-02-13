//! # kore-data
//!
//! Data loading and batching utilities for Kore.
//!
//! Provides:
//! - `MultipackSampler` — pack multiple sequences into one batch to eliminate padding waste
//! - `StreamingDataset` — memory-mapped dataset for large corpora
//! - `TokenBatcher` — efficient token-level batching for LLM training

pub mod multipack;
pub mod dataset;
pub mod batcher;
pub mod dataloader;

pub use multipack::MultipackSampler;
pub use dataset::StreamingDataset;
pub use batcher::TokenBatcher;
pub use dataloader::{Dataset, Sample, TensorDataset, DataLoader, DataLoaderIter, Batch};
