//! # kore-autograd
//!
//! Automatic differentiation engine for Kore.
//!
//! Provides a tape-based autograd system with:
//! - `GradFn` trait for differentiable operations
//! - `GradNode` computation graph nodes with weak references
//! - Topological sort (Kahn's algorithm) for backward pass
//! - `no_grad` scope for inference
//! - Gradient checkpointing support

pub mod graph;
pub mod backward;
pub mod grad_fn;
pub mod scope;

pub use graph::{GradNode, GradGraph};
pub use backward::backward;
pub use scope::NoGradGuard;
