//! # kore-core
//!
//! Core tensor engine for the Kore ML framework.
//!
//! Provides the foundational `Tensor` type with:
//! - Multiple dtypes (F16, BF16, F32, F64, I8, Ternary, Quaternary)
//! - CPU and CUDA device support
//! - Zero-copy views (reshape, transpose, slice)
//! - SIMD-accelerated operations
//! - Arena-friendly memory layout

pub mod autograd;
pub mod device;
pub mod dtype;
pub mod error;
pub mod ops;
pub mod prelude;
pub mod shape;
pub mod storage;
pub mod tensor;

pub use autograd::{backward, is_grad_enabled, GradFn, GradNode, NoGradGuard};
pub use device::Device;
pub use dtype::DType;
pub use error::KoreError;
pub use shape::Shape;
pub use storage::Storage;
pub use tensor::Tensor;

pub type Result<T> = std::result::Result<T, KoreError>;
