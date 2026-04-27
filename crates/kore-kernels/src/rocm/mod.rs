//! ROCm/HIP GPU backend for Kore.
//!
//! Parallel to `cuda/` — shares the same kernel source files (`.cu`),
//! compiled via hiprtc at runtime. Uses `libloading` for runtime-loaded
//! HIP/hiprtc function pointers (no build-time ROCm dependency).
//!
//! Provides:
//! - Runtime HIP detection and device management
//! - GPU memory allocation and host↔device transfers
//! - hiprtc kernel compilation with caching
//! - Mamba-3 scan forward/backward dispatch

pub mod context;
pub mod ffi;
pub mod launch;
pub mod memory;
pub mod ops;
