//! Tiled CPU matrix multiplication with SIMD acceleration.
//!
//! Uses cache-friendly tiling and dispatches to AVX2/AVX-512/NEON
//! inner loops when available, falling back to scalar.

use kore_core::{DType, KoreError, Tensor};

use crate::simd::SimdCapability;

/// Tile size for cache-friendly blocking.
/// 64×64 tiles fit comfortably in L1 cache (~32KB for f32).
const TILE_M: usize = 64;
const TILE_N: usize = 64;
const TILE_K: usize = 64;

/// Optimized 2D matrix multiplication: C = A @ B
/// [M, K] @ [K, N] → [M, N]
///
/// Uses tiled algorithm with SIMD inner loop dispatch.
pub fn matmul_f32(a: &Tensor, b: &Tensor) -> Result<Tensor, KoreError> {
    if a.dtype() != DType::F32 || b.dtype() != DType::F32 {
        return Err(KoreError::UnsupportedDType(a.dtype()));
    }

    let a = a.contiguous();
    let b = b.contiguous();
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();

    if a_dims.len() != 2 || b_dims.len() != 2 {
        return Err(KoreError::ShapeMismatch {
            expected: vec![0, 0],
            got: a_dims.to_vec(),
        });
    }

    let m = a_dims[0];
    let k = a_dims[1];
    let k2 = b_dims[0];
    let n = b_dims[1];

    if k != k2 {
        return Err(KoreError::MatmulDimMismatch { m, k1: k, k2, n });
    }

    let a_data = a.as_f32_slice().unwrap();
    let b_data = b.as_f32_slice().unwrap();
    let mut c_data = vec![0.0f32; m * n];

    let cap = SimdCapability::detect();

    if cap.avx2 {
        tiled_matmul_avx2(a_data, b_data, &mut c_data, m, n, k);
    } else {
        tiled_matmul_scalar(a_data, b_data, &mut c_data, m, n, k);
    }

    Ok(Tensor::from_f32(&c_data, &[m, n]))
}

/// Scalar tiled matmul (fallback).
fn tiled_matmul_scalar(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i0 in (0..m).step_by(TILE_M) {
        let i_end = (i0 + TILE_M).min(m);
        for j0 in (0..n).step_by(TILE_N) {
            let j_end = (j0 + TILE_N).min(n);
            for p0 in (0..k).step_by(TILE_K) {
                let p_end = (p0 + TILE_K).min(k);

                for i in i0..i_end {
                    for p in p0..p_end {
                        let a_val = a[i * k + p];
                        for j in j0..j_end {
                            c[i * n + j] += a_val * b[p * n + j];
                        }
                    }
                }
            }
        }
    }
}

/// AVX2-accelerated tiled matmul.
///
/// Processes 8 floats at a time in the inner loop using 256-bit SIMD.
#[cfg(target_arch = "x86_64")]
fn tiled_matmul_avx2(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return tiled_matmul_scalar(a, b, c, m, n, k);
    }

    // Safety: we checked AVX2+FMA above
    unsafe { tiled_matmul_avx2_inner(a, b, c, m, n, k) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tiled_matmul_avx2_inner(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    use std::arch::x86_64::*;

    for i0 in (0..m).step_by(TILE_M) {
        let i_end = (i0 + TILE_M).min(m);
        for j0 in (0..n).step_by(TILE_N) {
            let j_end = (j0 + TILE_N).min(n);
            for p0 in (0..k).step_by(TILE_K) {
                let p_end = (p0 + TILE_K).min(k);

                for i in i0..i_end {
                    for p in p0..p_end {
                        let a_val = _mm256_set1_ps(a[i * k + p]);

                        // Process 8 elements at a time
                        let mut j = j0;
                        while j + 8 <= j_end {
                            let c_ptr = c.as_mut_ptr().add(i * n + j);
                            let b_ptr = b.as_ptr().add(p * n + j);

                            let c_vec = _mm256_loadu_ps(c_ptr);
                            let b_vec = _mm256_loadu_ps(b_ptr);
                            let result = _mm256_fmadd_ps(a_val, b_vec, c_vec);
                            _mm256_storeu_ps(c_ptr, result);

                            j += 8;
                        }

                        // Scalar tail
                        while j < j_end {
                            c[i * n + j] += a[i * k + p] * b[p * n + j];
                            j += 1;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn tiled_matmul_avx2(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    tiled_matmul_scalar(a, b, c, m, n, k);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_basic() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::from_f32(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        let c = matmul_f32(&a, &b).unwrap();
        assert_eq!(c.shape().dims(), &[2, 2]);
        assert_eq!(c.as_f32_slice().unwrap(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_matmul_identity() {
        let a = Tensor::from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let b = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = matmul_f32(&a, &b).unwrap();
        assert_eq!(c.as_f32_slice().unwrap(), &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_matmul_large() {
        // Test with sizes that exercise tiling
        let m = 128;
        let k = 64;
        let n = 96;
        let a_data: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 * 0.1).collect();

        let a = Tensor::from_f32(&a_data, &[m, k]);
        let b = Tensor::from_f32(&b_data, &[k, n]);

        let c_fast = matmul_f32(&a, &b).unwrap();
        let c_ref = a.matmul(&b).unwrap();

        let fast_data = c_fast.as_f32_slice().unwrap();
        let ref_data = c_ref.as_f32_slice().unwrap();

        for (i, (&f, &r)) in fast_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (f - r).abs() < 1e-3,
                "Mismatch at index {}: fast={}, ref={}",
                i, f, r
            );
        }
    }

    #[test]
    fn test_matmul_dim_mismatch() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3, 1]);
        assert!(matmul_f32(&a, &b).is_err());
    }
}
