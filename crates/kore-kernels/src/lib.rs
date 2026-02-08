//! # kore-kernels
//!
//! CUDA (cudarc + PTX) and CPU SIMD kernel dispatch for Kore.
//!
//! Provides:
//! - Runtime SIMD capability detection (AVX2, AVX-512, NEON)
//! - Tiled CPU matmul with SIMD inner loops
//! - Quaternary (2-bit) matmul with SIMD unpacking
//! - Fused operations (linear+ReLU, LayerNorm, softmax)
//! - CUDA dispatch (behind `cuda` feature flag)

pub mod simd;
pub mod cpu_matmul;
pub mod cpu_quat_matmul;
pub mod cpu_fused;

pub use simd::SimdCapability;
