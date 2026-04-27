//! Portable operator library for edge inference.
//!
//! All ops work on raw `&[f32]` / `&[u8]` slices — no Tensor overhead.
//! These are scalar fallbacks; SIMD backends override via `simd_dispatch`.

pub mod activation;
pub mod attention;
pub mod elementwise;
pub mod embedding;
pub mod matmul;
pub mod norm;
pub mod rope;
