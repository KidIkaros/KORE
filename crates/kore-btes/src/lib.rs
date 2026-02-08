//! # kore-btes
//!
//! Binary/Ternary/Quaternary encoding and compute engine.
//!
//! Consolidates all BTES functionality:
//! - Binary ↔ Ternary ↔ Quaternary conversion
//! - 64-trit parallel ALU (VT-ALU)
//! - Base-243 ternary packing + 2-bit quaternary packing
//! - SIMD-accelerated matmul for quantized weights
//! - Float → ternary/quaternary quantization

pub mod encoder;
pub mod packing;
pub mod quantize;
