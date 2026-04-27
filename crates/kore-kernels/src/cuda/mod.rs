//! CUDA GPU backend for Kore.
//!
//! Provides:
//! - Device context management (lazy singleton per GPU)
//! - GPU memory allocation and host↔device transfers
//! - Kernel launcher with PTX caching
//! - Optimized CUDA kernels (GEMM, element-wise, reductions, etc.)

pub mod context;
pub mod launch;
pub mod memory;
pub mod ops;
