//! # kore-autograd
//!
//! Automatic differentiation engine for Kore.
//!
//! The canonical autograd types (`GradFn`, `GradNode`, `NoGradGuard`, `backward`)
//! now live in `kore_core::autograd` so that `Tensor` can carry gradient tracking
//! without circular dependencies. This crate re-exports them for backward
//! compatibility and provides the original module structure.

pub mod graph;
pub mod backward;
pub mod grad_fn;
pub mod scope;
pub mod checkpoint;

// Re-export canonical types from kore-core
pub use kore_core::autograd::{
    GradFn, GradNode, NoGradGuard,
    backward, is_grad_enabled,
    AddBackward, SubBackward, MulBackward, DivBackward,
    MatmulBackward, NegBackward, ExpBackward, LogBackward,
    SqrtBackward, AbsBackward, SumBackward, MeanBackward,
    AddScalarBackward, MulScalarBackward, PowScalarBackward,
    ClampBackward,
};

// Legacy re-exports from submodules
pub use graph::GradGraph;
