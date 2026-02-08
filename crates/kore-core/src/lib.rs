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

pub mod dtype;
pub mod device;
pub mod storage;
pub mod shape;
pub mod tensor;
pub mod ops;
pub mod error;

pub use dtype::DType;
pub use device::Device;
pub use storage::Storage;
pub use shape::Shape;
pub use tensor::Tensor;
pub use error::KoreError;

pub type Result<T> = std::result::Result<T, KoreError>;
