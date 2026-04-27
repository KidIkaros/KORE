//! # kore-autograd
//!
//! Automatic differentiation engine for Kore.
//!
//! The canonical autograd types (`GradFn`, `GradNode`, `NoGradGuard`, `backward`)
//! now live in `kore_core::autograd` so that `Tensor` can carry gradient tracking
//! without circular dependencies. This crate re-exports them for backward
//! compatibility and provides the original module structure.

pub mod backward;
pub mod checkpoint;
pub mod grad_fn;
pub mod graph;
pub mod scope;

// Re-export canonical types from kore-core
pub use kore_core::autograd::{
    backward, is_grad_enabled, AbsBackward, AddBackward, AddScalarBackward, ClampBackward,
    DivBackward, ExpBackward, GradFn, GradNode, LogBackward, MatmulBackward, MeanBackward,
    MulBackward, MulScalarBackward, NegBackward, NoGradGuard, PowScalarBackward, SqrtBackward,
    SubBackward, SumBackward,
};

// Legacy re-exports from submodules
pub use graph::GradGraph;
