//! # kore-btes
//!
//! Binary/Ternary/Quaternary encoding and compute engine.
//!
//! Consolidates all BTES functionality:
//! - Binary ↔ Ternary ↔ Quaternary conversion
//! - 64-trit parallel ALU (VT-ALU)
//! - Base-243 ternary packing + 2-bit quaternary packing
//! - SIMD-accelerated matmul for quantized weights
//! - Float → ternary/quaternary/binary quantization

pub mod encoder;
pub mod memory;
pub mod packing;
pub mod quantize;
pub mod vtalu;

pub use encoder::{
    decode_trits, encode_trits, pack_quats, unpack_quats, Quat, Trit, TERNARY_DECODE_LUT,
};
pub use memory::TernaryFrame;
pub use quantize::{
    dequantize_binary, dequantize_quaternary, dequantize_ternary, pack_binary_block,
    quantize_binary, quantize_quaternary, quantize_ternary, unpack_binary_block,
};
pub use vtalu::TernaryWord64;
