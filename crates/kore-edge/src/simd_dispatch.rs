//! Unified SIMD dispatch layer.
//!
//! Provides a single function signature per op that dispatches to the best
//! available backend at compile time: NEON → WASM SIMD → scalar fallback.
//! Zero runtime overhead — monomorphized per target.

use crate::ops;

/// Matmul: C[M,N] = A[M,K] @ B[K,N]
pub fn matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        crate::neon::neon_matmul_f32(a, b, c, m, n, k);
        return;
    }

    #[cfg(target_arch = "wasm32")]
    {
        crate::wasm_simd::wasm_matmul_f32(a, b, c, m, n, k);
        return;
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
    {
        ops::matmul::matmul_f32(a, b, c, m, n, k);
    }
}

/// Ternary matmul: C[M,N] = A_ternary[M,K] @ B_f32[K,N] * scales[M]
pub fn matmul_ternary(
    a_packed: &[u8], a_scales: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        crate::neon::neon_matmul_ternary(a_packed, a_scales, b, c, m, n, k);
        return;
    }

    #[cfg(target_arch = "wasm32")]
    {
        crate::wasm_simd::wasm_matmul_ternary(a_packed, a_scales, b, c, m, n, k);
        return;
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
    {
        ops::matmul::matmul_ternary(a_packed, a_scales, b, c, m, n, k);
    }
}

/// Quaternary matmul: C[M,N] = A_quat[M,K] @ B_f32[K,N] * scales[M]
pub fn matmul_quaternary(
    a_packed: &[u8], a_scales: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        crate::neon::neon_matmul_quaternary(a_packed, a_scales, b, c, m, n, k);
        return;
    }

    #[cfg(target_arch = "wasm32")]
    {
        crate::wasm_simd::wasm_matmul_quaternary(a_packed, a_scales, b, c, m, n, k);
        return;
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
    {
        ops::matmul::matmul_quaternary(a_packed, a_scales, b, c, m, n, k);
    }
}

/// RMS normalization (in-place).
pub fn rms_norm(data: &mut [f32], gamma: &[f32], dim: usize, eps: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        crate::neon::neon_rms_norm(data, gamma, dim, eps);
        return;
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        ops::norm::rms_norm(data, gamma, dim, eps);
    }
}

/// Softmax over rows (in-place).
pub fn softmax(data: &mut [f32], dim: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        crate::neon::neon_softmax(data, dim);
        return;
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        ops::activation::softmax(data, dim);
    }
}

/// Returns the name of the active SIMD backend.
pub fn backend_name() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    { return "NEON"; }

    #[cfg(target_arch = "wasm32")]
    { return "WASM-SIMD128"; }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
    { "scalar" }
}
