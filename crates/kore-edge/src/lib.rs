//! # kore-edge
//!
//! Lightweight on-device inference runtime for mobile (Android/iOS),
//! WASM (browser/Node), and embedded targets.
//!
//! Competes with ExecuTorch by leveraging Kore's native ternary/quaternary
//! quantization (1.6-bit and 2-bit) for extreme model compression.
//!
//! ## Key Features
//! - `.koref` model format: single-file, mmap-friendly, mixed quantization
//! - Arena allocator: zero-alloc steady-state inference
//! - Portable operator library with SIMD backends (NEON + WASM SIMD128)
//! - C FFI for Swift/Kotlin + wasm-bindgen JS/TS API

pub mod arena;
pub mod format;
pub mod loader;
pub mod ops;
pub mod plan;
pub mod runtime;
pub mod simd_dispatch;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "wasm32")]
pub mod wasm_simd;

pub mod wasm_api;

#[cfg(feature = "ffi")]
pub mod ffi;

pub use arena::Arena;
pub use format::{KorefHeader, KorefModel, TensorEntry};
pub use plan::ExecutionPlan;
pub use runtime::Session;
