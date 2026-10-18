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
unsafe fn tiled_matmul_avx2_inner(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
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

/// Zero-allocation in-place CPU Q4_K GEMV / GEMM (`y = x @ W^T`):
/// - `x`: `[M, K]` f32 tensor
/// - `w_raw`: `[N * (K / 256) * 144]` packed Q4_K bytes
/// - `n`: output features (rows of W)
/// - `k`: input features (columns of W, multiple of 256)
pub fn gemv_q4_k_cpu(x: &Tensor, w_raw: &[u8], n: usize, k: usize) -> Result<Tensor, KoreError> {
    use rayon::prelude::*;

    let x_cont = x.contiguous();
    let x_shape = x_cont.shape().dims();
    let m = if x_shape.is_empty() { 1 } else { x_shape[0] };
    let k_in = if x_shape.len() > 1 {
        x_shape[1]
    } else {
        x_shape[0]
    };
    if k_in != k || k % 256 != 0 {
        return Err(KoreError::ShapeMismatch {
            expected: vec![m, k],
            got: x_shape.to_vec(),
        });
    }
    let blocks_per_row = k / 256;
    let row_bytes = blocks_per_row * 144;
    if w_raw.len() < n * row_bytes {
        return Err(KoreError::StorageError(format!(
            "gemv_q4_k_cpu: w_raw too small ({} < {})",
            w_raw.len(),
            n * row_bytes
        )));
    }
    let x_data = x_cont
        .as_f32_slice()
        .ok_or_else(|| KoreError::UnsupportedDType(x.dtype()))?;

    let mut out = vec![0.0f32; m * n];
    // Parallelize across output rows `r in 0..n`
    let row_results: Vec<[f32; 4]> = if m <= 4 {
        (0..n)
            .into_par_iter()
            .map(|r| {
                let w_row = &w_raw[r * row_bytes..(r + 1) * row_bytes];
                dot_q4_k_row_batch(w_row, x_data, blocks_per_row, k, m)
            })
            .collect()
    } else {
        Vec::new()
    };

    if m <= 4 {
        for (r, sums) in row_results.iter().enumerate() {
            for mi in 0..m {
                out[mi * n + r] = sums[mi];
            }
        }
    } else {
        for mi in 0..m {
            let x_row = &x_data[mi * k..(mi + 1) * k];
            let out_row = &mut out[mi * n..(mi + 1) * n];
            out_row.par_iter_mut().enumerate().for_each(|(r, slot)| {
                let w_row = &w_raw[r * row_bytes..(r + 1) * row_bytes];
                *slot = dot_q4_k_row_single(w_row, x_row, blocks_per_row);
            });
        }
    }

    Ok(Tensor::from_f32(&out, &[m, n]))
}

#[inline]
fn unpack_q4_k_scales(sc_bytes: &[u8]) -> ([f32; 8], [f32; 8]) {
    let mut sc = [0.0f32; 8];
    let mut mn = [0.0f32; 8];
    for j in 0..4 {
        sc[j] = (sc_bytes[j] & 63) as f32;
        mn[j] = (sc_bytes[j + 4] & 63) as f32;
    }
    for j in 4..8 {
        let lo_sc = sc_bytes[j + 4] & 0x0F;
        let hi_sc = (sc_bytes[j - 4] >> 6) & 0x03;
        sc[j] = (lo_sc | (hi_sc << 4)) as f32;

        let lo_mn = (sc_bytes[j + 4] >> 4) & 0x0F;
        let hi_mn = (sc_bytes[j] >> 6) & 0x03;
        mn[j] = (lo_mn | (hi_mn << 4)) as f32;
    }
    (sc, mn)
}

#[inline]
fn dot_q4_k_row_single(w_row: &[u8], x_row: &[f32], blocks_per_row: usize) -> f32 {
    dot_q4_k_row_batch(w_row, x_row, blocks_per_row, x_row.len(), 1)[0]
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)))
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn cvt_u8x32_f32x8(v: std::arch::x86_64::__m256i, out: &mut [std::arch::x86_64::__m256; 4]) {
    use std::arch::x86_64::*;
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    out[0] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo));
    out[1] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8)));
    out[2] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi));
    out[3] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8)));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_q4_k_row_batch_avx2(
    w_row: &[u8],
    x_data: &[f32],
    blocks_per_row: usize,
    k: usize,
    m: usize,
) -> [f32; 4] {
    use std::arch::x86_64::*;
    let mut sums = [0.0f32; 4];
    let mask0f = _mm256_set1_epi8(0x0F);
    for b in 0..blocks_per_row {
        let block = w_row.as_ptr().add(b * 144);
        let d = half::f16::from_le_bytes([*block, *block.add(1)]).to_f32();
        let dmin = half::f16::from_le_bytes([*block.add(2), *block.add(3)]).to_f32();
        let (sc, mn) = unpack_q4_k_scales(std::slice::from_raw_parts(block.add(4), 12));
        let qs = block.add(16);
        let k_base = b * 256;
        for g in 0..4 {
            let v = _mm256_loadu_si256(qs.add(g * 32) as *const __m256i);
            let mut ql = [_mm256_setzero_ps(); 4];
            let mut qh = [_mm256_setzero_ps(); 4];
            cvt_u8x32_f32x8(_mm256_and_si256(v, mask0f), &mut ql);
            cvt_u8x32_f32x8(_mm256_and_si256(_mm256_srli_epi16(v, 4), mask0f), &mut qh);
            let gk = k_base + g * 64;
            let d_sc_lo = d * sc[2 * g];
            let dm_mn_lo = dmin * mn[2 * g];
            let d_sc_hi = d * sc[2 * g + 1];
            let dm_mn_hi = dmin * mn[2 * g + 1];
            for mi in 0..m {
                let xp = x_data.as_ptr().add(mi * k + gk);
                let mut dlo = _mm256_setzero_ps();
                let mut dhi = _mm256_setzero_ps();
                let mut slo = _mm256_setzero_ps();
                let mut shi = _mm256_setzero_ps();
                for c in 0..4 {
                    let xl = _mm256_loadu_ps(xp.add(c * 8));
                    let xh = _mm256_loadu_ps(xp.add(32 + c * 8));
                    dlo = _mm256_fmadd_ps(ql[c], xl, dlo);
                    slo = _mm256_add_ps(slo, xl);
                    dhi = _mm256_fmadd_ps(qh[c], xh, dhi);
                    shi = _mm256_add_ps(shi, xh);
                }
                sums[mi] += d_sc_lo * hsum256(dlo) - dm_mn_lo * hsum256(slo)
                    + d_sc_hi * hsum256(dhi)
                    - dm_mn_hi * hsum256(shi);
            }
        }
    }
    sums
}

#[inline]
fn dot_q4_k_row_batch(
    w_row: &[u8],
    x_data: &[f32],
    blocks_per_row: usize,
    k: usize,
    m: usize,
) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    if SimdCapability::detect().avx2 {
        return unsafe { dot_q4_k_row_batch_avx2(w_row, x_data, blocks_per_row, k, m) };
    }
    let mut sums = [0.0f32; 4];
    for b in 0..blocks_per_row {
        let bo = b * 144;
        let block = &w_row[bo..bo + 144];
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let (sc, mn) = unpack_q4_k_scales(&block[4..16]);
        let qs = &block[16..144];
        let k_base = b * 256;

        for g in 0..4 {
            let d_sc_lo = d * sc[2 * g];
            let dm_mn_lo = dmin * mn[2 * g];
            let d_sc_hi = d * sc[2 * g + 1];
            let dm_mn_hi = dmin * mn[2 * g + 1];
            let q_group = &qs[g * 32..(g + 1) * 32];
            let gk = k_base + g * 64;

            for mi in 0..m {
                let x_lo = &x_data[mi * k + gk..mi * k + gk + 32];
                let x_hi = &x_data[mi * k + gk + 32..mi * k + gk + 64];
                let mut dot_lo = 0.0f32;
                let mut sum_x_lo = 0.0f32;
                let mut dot_hi = 0.0f32;
                let mut sum_x_hi = 0.0f32;
                for l in 0..32 {
                    let qb = q_group[l];
                    let xl = x_lo[l];
                    let xh = x_hi[l];
                    dot_lo += ((qb & 0x0F) as f32) * xl;
                    sum_x_lo += xl;
                    dot_hi += ((qb >> 4) as f32) * xh;
                    sum_x_hi += xh;
                }
                sums[mi] +=
                    d_sc_lo * dot_lo - dm_mn_lo * sum_x_lo + d_sc_hi * dot_hi - dm_mn_hi * sum_x_hi;
            }
        }
    }
    sums
}

/// Zero-allocation in-place CPU Q6_K GEMV / GEMM (`y = x @ W^T`):
/// - `x`: `[M, K]` f32 tensor
/// - `w_raw`: `[N * (K / 256) * 210]` packed Q6_K bytes
/// - `n`: output features (rows of W)
/// - `k`: input features (columns of W, multiple of 256)
pub fn gemv_q6_k_cpu(x: &Tensor, w_raw: &[u8], n: usize, k: usize) -> Result<Tensor, KoreError> {
    use rayon::prelude::*;

    let x_cont = x.contiguous();
    let x_shape = x_cont.shape().dims();
    let m = if x_shape.is_empty() { 1 } else { x_shape[0] };
    let k_in = if x_shape.len() > 1 {
        x_shape[1]
    } else {
        x_shape[0]
    };
    if k_in != k || k % 256 != 0 {
        return Err(KoreError::ShapeMismatch {
            expected: vec![m, k],
            got: x_shape.to_vec(),
        });
    }
    let blocks_per_row = k / 256;
    let row_bytes = blocks_per_row * 210;
    if w_raw.len() < n * row_bytes {
        return Err(KoreError::StorageError(format!(
            "gemv_q6_k_cpu: w_raw too small ({} < {})",
            w_raw.len(),
            n * row_bytes
        )));
    }
    let x_data = x_cont
        .as_f32_slice()
        .ok_or_else(|| KoreError::UnsupportedDType(x.dtype()))?;

    let mut out = vec![0.0f32; m * n];
    if m <= 4 {
        let row_results: Vec<[f32; 4]> = (0..n)
            .into_par_iter()
            .map(|r| {
                let w_row = &w_raw[r * row_bytes..(r + 1) * row_bytes];
                dot_q6_k_row_batch(w_row, x_data, blocks_per_row, k, m)
            })
            .collect();
        for (r, sums) in row_results.iter().enumerate() {
            for mi in 0..m {
                out[mi * n + r] = sums[mi];
            }
        }
    } else {
        for mi in 0..m {
            let x_row = &x_data[mi * k..(mi + 1) * k];
            let out_row = &mut out[mi * n..(mi + 1) * n];
            out_row.par_iter_mut().enumerate().for_each(|(r, slot)| {
                let w_row = &w_raw[r * row_bytes..(r + 1) * row_bytes];
                *slot = dot_q6_k_row_batch(w_row, x_row, blocks_per_row, k, 1)[0];
            });
        }
    }

    Ok(Tensor::from_f32(&out, &[m, n]))
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn cvt_i8x32_f32x8(v: std::arch::x86_64::__m256i, out: &mut [std::arch::x86_64::__m256; 4]) {
    use std::arch::x86_64::*;
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    out[0] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
    out[1] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
    out[2] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
    out[3] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_q6_k_row_batch_avx2(
    w_row: &[u8],
    x_data: &[f32],
    blocks_per_row: usize,
    k: usize,
    m: usize,
) -> [f32; 4] {
    use std::arch::x86_64::*;
    let mut sums = [0.0f32; 4];
    let mask0f = _mm256_set1_epi8(0x0F);
    let mask03 = _mm256_set1_epi8(0x03);
    let c32 = _mm256_set1_epi8(32);
    for b in 0..blocks_per_row {
        let block = w_row.as_ptr().add(b * 210);
        let ql = block;
        let qh = block.add(128);
        let scales = block.add(192);
        let d = half::f16::from_le_bytes([*block.add(208), *block.add(209)]).to_f32();
        let k_base = b * 256;

        for g in 0..2 {
            let ql_a = _mm256_loadu_si256(ql.add(g * 64) as *const __m256i);
            let ql_b = _mm256_loadu_si256(ql.add(g * 64 + 32) as *const __m256i);
            let qhb = _mm256_loadu_si256(qh.add(g * 32) as *const __m256i);
            let sc_g = scales.add(g * 8);
            let gk = k_base + g * 128;

            // sub-column c (x offsets c*32..c*32+32): low/high nibble of ql_{a|b} | 2-bit field of qhb << 4, -32
            let q = [
                _mm256_sub_epi8(
                    _mm256_or_si256(
                        _mm256_and_si256(ql_a, mask0f),
                        _mm256_slli_epi16(_mm256_and_si256(qhb, mask03), 4),
                    ),
                    c32,
                ),
                _mm256_sub_epi8(
                    _mm256_or_si256(
                        _mm256_and_si256(ql_b, mask0f),
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhb, 2), mask03), 4),
                    ),
                    c32,
                ),
                _mm256_sub_epi8(
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(ql_a, 4), mask0f),
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhb, 4), mask03), 4),
                    ),
                    c32,
                ),
                _mm256_sub_epi8(
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(ql_b, 4), mask0f),
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhb, 6), mask03), 4),
                    ),
                    c32,
                ),
            ];
            let mut w = [[_mm256_setzero_ps(); 4]; 4];
            for c in 0..4 {
                cvt_i8x32_f32x8(q[c], &mut w[c]);
            }

            for mi in 0..m {
                let xp = x_data.as_ptr().add(mi * k + gk);
                let mut acc16 = [[_mm256_setzero_ps(); 2]; 4];
                for c in 0..4 {
                    let xc = xp.add(c * 32);
                    acc16[c][0] = _mm256_fmadd_ps(w[c][0], _mm256_loadu_ps(xc), acc16[c][0]);
                    acc16[c][0] = _mm256_fmadd_ps(w[c][1], _mm256_loadu_ps(xc.add(8)), acc16[c][0]);
                    acc16[c][1] =
                        _mm256_fmadd_ps(w[c][2], _mm256_loadu_ps(xc.add(16)), acc16[c][1]);
                    acc16[c][1] =
                        _mm256_fmadd_ps(w[c][3], _mm256_loadu_ps(xc.add(24)), acc16[c][1]);
                }
                for c in 0..4 {
                    let s_a = d * (*sc_g.add(2 * c) as i8) as f32;
                    let s_b = d * (*sc_g.add(2 * c + 1) as i8) as f32;
                    sums[mi] += s_a * hsum256(acc16[c][0]) + s_b * hsum256(acc16[c][1]);
                }
            }
        }
    }
    sums
}

#[inline]
fn dot_q6_k_row_batch(
    w_row: &[u8],
    x_data: &[f32],
    blocks_per_row: usize,
    k: usize,
    m: usize,
) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    if SimdCapability::detect().avx2 {
        return unsafe { dot_q6_k_row_batch_avx2(w_row, x_data, blocks_per_row, k, m) };
    }
    let mut sums = [0.0f32; 4];
    for b in 0..blocks_per_row {
        let bo = b * 210;
        let block = &w_row[bo..bo + 210];
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = half::f16::from_le_bytes([block[208], block[209]]).to_f32();
        let k_base = b * 256;

        for g in 0..2 {
            let ql_g = &ql[g * 64..(g + 1) * 64];
            let qh_g = &qh[g * 32..(g + 1) * 32];
            let sc_g = &scales[g * 8..(g + 1) * 8];
            let gk = k_base + g * 128;

            for is in 0..2 {
                let s0 = d * (sc_g[is] as i8 as f32);
                let s1 = d * (sc_g[is + 2] as i8 as f32);
                let s2 = d * (sc_g[is + 4] as i8 as f32);
                let s3 = d * (sc_g[is + 6] as i8 as f32);
                let l_start = is * 16;

                for l in l_start..l_start + 16 {
                    let ql1 = ql_g[l];
                    let ql2 = ql_g[l + 32];
                    let qhb = qh_g[l];

                    let w0 = s0 * (((ql1 & 0x0F) | ((qhb & 3) << 4)) as i32 - 32) as f32;
                    let w1 = s1 * (((ql2 & 0x0F) | (((qhb >> 2) & 3) << 4)) as i32 - 32) as f32;
                    let w2 = s2 * (((ql1 >> 4) | (((qhb >> 4) & 3) << 4)) as i32 - 32) as f32;
                    let w3 = s3 * (((ql2 >> 4) | (((qhb >> 6) & 3) << 4)) as i32 - 32) as f32;

                    for mi in 0..m {
                        let base_x = mi * k + gk + l;
                        sums[mi] += w0 * x_data[base_x]
                            + w1 * x_data[base_x + 32]
                            + w2 * x_data[base_x + 64]
                            + w3 * x_data[base_x + 96];
                    }
                }
            }
        }
    }
    sums
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
                i,
                f,
                r
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
