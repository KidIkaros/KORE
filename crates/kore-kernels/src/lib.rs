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

pub mod cpu_binary_dequant;
pub mod cpu_binary_matmul;
pub mod cpu_fused;
pub mod cpu_fused_backward;
pub mod cpu_matmul;
pub mod cpu_matmul_f16;
pub mod cpu_quat_matmul;
pub mod cpu_ternary_dequant;
pub mod cpu_ternary_matmul;
pub mod hadamard;
pub mod k_quant;
pub mod q2_0;
pub mod simd;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use simd::SimdCapability;
