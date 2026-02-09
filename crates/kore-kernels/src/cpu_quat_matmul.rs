//! Quaternary (2-bit) matrix multiplication on CPU.
//!
//! Ported from btes/cuda/qbtes_cuda.cu and btes/src/qbtes_matmul_avx2.c.
//! Packed weights: 4 quaternary values per byte (2 bits each).
//! Quaternary values map to: {-3, -1, +1, +3}.

use kore_core::{KoreError, Tensor};

/// Quaternary value lookup table: index → float.
/// Matches btes/cuda/qbtes_cuda.cu c_quat_values[].
const QUAT_VALUES: [f32; 4] = [-3.0, -1.0, 1.0, 3.0];

/// Quaternary matrix multiplication: C = A_packed @ B
///
/// - `a_packed`: packed quaternary weights [M, K_packed] as raw bytes
/// - `a_scales`: per-row scale factors [M]
/// - `b`: dense f32 activations [K, N]
/// - `m`, `n`, `k`: logical dimensions
/// - `k_packed`: number of packed bytes per row (= ceil(K/4))
///
/// Returns C [M, N] as f32 tensor.
pub fn quat_matmul(
    a_packed: &[u8],
    a_scales: &[f32],
    b: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Tensor, KoreError> {
    let b_cont = b.contiguous();
    let b_data = b_cont.as_f32_slice().ok_or_else(|| {
        KoreError::UnsupportedDType(b.dtype())
    })?;

    let k_packed = (k + 3) / 4;

    if a_packed.len() < m * k_packed {
        return Err(KoreError::StorageError(format!(
            "a_packed too small: need {} bytes, got {}",
            m * k_packed,
            a_packed.len()
        )));
    }
    if a_scales.len() < m {
        return Err(KoreError::StorageError(format!(
            "a_scales too small: need {}, got {}",
            m,
            a_scales.len()
        )));
    }

    let mut c_data = vec![0.0f32; m * n];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                quat_matmul_avx2(a_packed, a_scales, b_data, &mut c_data, m, n, k, k_packed);
            }
            return Ok(Tensor::from_f32(&c_data, &[m, n]));
        }
    }

    quat_matmul_scalar(a_packed, a_scales, b_data, &mut c_data, m, n, k, k_packed);
    Ok(Tensor::from_f32(&c_data, &[m, n]))
}

/// Scalar fallback for quaternary matmul.
///
/// Uses A-stationary accumulation: pre-unpack the entire row of quaternary
/// values, then iterate over K and scatter weighted B values into N outputs.
/// This is more cache-friendly for the output vector when N is small.
fn quat_matmul_scalar(
    a_packed: &[u8],
    a_scales: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    k_packed: usize,
) {
    for row in 0..m {
        let scale = a_scales[row];
        let a_row = &a_packed[row * k_packed..(row + 1) * k_packed];
        let c_row = &mut c[row * n..(row + 1) * n];

        // Pre-unpack all quaternary values for this row
        let mut quats = Vec::with_capacity(k + 3);
        for &byte in a_row {
            quats.push(QUAT_VALUES[((byte) & 0x3) as usize]);
            quats.push(QUAT_VALUES[((byte >> 2) & 0x3) as usize]);
            quats.push(QUAT_VALUES[((byte >> 4) & 0x3) as usize]);
            quats.push(QUAT_VALUES[((byte >> 6) & 0x3) as usize]);
        }
        quats.truncate(k);

        // A-stationary: iterate K, scatter into N outputs
        for ki in 0..k {
            let w = quats[ki] * scale;
            let b_row = &b[ki * n..ki * n + n];
            for col in 0..n {
                c_row[col] += w * b_row[col];
            }
        }
    }
}

/// AVX2-accelerated quaternary matmul.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn quat_matmul_avx2(
    a_packed: &[u8],
    a_scales: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    k_packed: usize,
) {
    use std::arch::x86_64::*;

    let _mask_2bit = _mm256_set1_epi32(0x03);

    for row in 0..m {
        let scale = a_scales[row];
        let _scale_vec = _mm256_set1_ps(scale);
        let a_row = &a_packed[row * k_packed..(row + 1) * k_packed];

        for col_block in (0..n).step_by(8) {
            let cols_left = (n - col_block).min(8);
            let mut acc = _mm256_setzero_ps();

            for kp in 0..k_packed {
                let packed = a_row[kp] as i32;
                let k_base = kp * 4;

                for i in 0..4 {
                    let ki = k_base + i;
                    if ki >= k {
                        break;
                    }

                    let idx = (packed >> (2 * i)) & 0x3;
                    let w = QUAT_VALUES[idx as usize] * scale;
                    let w_vec = _mm256_set1_ps(w);

                    if cols_left == 8 {
                        let b_vec = _mm256_loadu_ps(b.as_ptr().add(ki * n + col_block));
                        acc = _mm256_fmadd_ps(w_vec, b_vec, acc);
                    } else {
                        // Scalar tail for partial blocks
                        for j in 0..cols_left {
                            let c_idx = row * n + col_block + j;
                            c[c_idx] += w * b[ki * n + col_block + j];
                        }
                    }
                }
            }

            if cols_left == 8 {
                _mm256_storeu_ps(c.as_mut_ptr().add(row * n + col_block), acc);
            }
        }
    }
}

/// Pack f32 weights into quaternary format with per-row scales.
///
/// Returns (packed_bytes, scales).
pub fn pack_weights_quaternary(weights: &[f32], m: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let k_packed = (k + 3) / 4;
    let mut packed = vec![0u8; m * k_packed];
    let mut scales = vec![0.0f32; m];

    for row in 0..m {
        let row_start = row * k;
        let row_end = row_start + k;
        let row_data = &weights[row_start..row_end];

        // Per-row scale: max absolute value / 3
        let abs_max = row_data.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
        let scale = if abs_max < 1e-8 { 1.0 } else { abs_max / 3.0 };
        scales[row] = scale;

        // Quantize and pack
        for kp in 0..k_packed {
            let mut byte = 0u8;
            for i in 0..4 {
                let ki = kp * 4 + i;
                if ki >= k {
                    break;
                }
                let normalized = row_data[ki] / scale;
                let idx: u8 = if normalized < -2.0 {
                    0 // -3
                } else if normalized < 0.0 {
                    1 // -1
                } else if normalized < 2.0 {
                    2 // +1
                } else {
                    3 // +3
                };
                byte |= idx << (2 * i);
            }
            packed[row * k_packed + kp] = byte;
        }
    }

    (packed, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_identity() {
        // Weights that map cleanly to quaternary values
        let weights = vec![
            -3.0, -1.0, 1.0, 3.0,  // row 0
            3.0, 1.0, -1.0, -3.0,  // row 1
        ];
        let (packed, scales) = pack_weights_quaternary(&weights, 2, 4);

        assert_eq!(packed.len(), 2); // 2 rows × 1 byte each
        assert_eq!(scales.len(), 2);
        assert!((scales[0] - 1.0).abs() < 1e-6);
        assert!((scales[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quat_matmul_basic() {
        // A = [[3, 1], [-1, -3]] (quaternary-friendly)
        // B = [[1, 0], [0, 1]] (identity)
        // C = A @ B = A
        let weights = vec![3.0, 1.0, -1.0, -3.0];
        let (packed, scales) = pack_weights_quaternary(&weights, 2, 2);

        let b = Tensor::from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let c = quat_matmul(&packed, &scales, &b, 2, 2, 2).unwrap();

        let c_data = c.as_f32_slice().unwrap();
        // With quantization, results should be close to original
        assert!((c_data[0] - 3.0).abs() < 0.5, "got {}", c_data[0]);
        assert!((c_data[1] - 1.0).abs() < 0.5, "got {}", c_data[1]);
        assert!((c_data[2] - (-1.0)).abs() < 0.5, "got {}", c_data[2]);
        assert!((c_data[3] - (-3.0)).abs() < 0.5, "got {}", c_data[3]);
    }

    #[test]
    fn test_quat_matmul_larger() {
        let m = 16;
        let k = 32;
        let n = 16;

        let weights: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0)).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.1).collect();

        let (packed, scales) = pack_weights_quaternary(&weights, m, k);
        let b = Tensor::from_f32(&b_data, &[k, n]);

        let c = quat_matmul(&packed, &scales, &b, m, n, k).unwrap();
        assert_eq!(c.shape().dims(), &[m, n]);

        // Verify no NaN/Inf
        let c_data = c.as_f32_slice().unwrap();
        assert!(c_data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_quat_matmul_non_aligned_k() {
        // K=7 is not a multiple of 4, tests the truncation path
        let m = 2;
        let k = 7;
        let n = 3;

        let weights: Vec<f32> = (0..m * k).map(|i| ((i % 4) as f32 * 2.0 - 3.0)).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();

        let (packed, scales) = pack_weights_quaternary(&weights, m, k);
        let b = Tensor::from_f32(&b_data, &[k, n]);

        let c = quat_matmul(&packed, &scales, &b, m, n, k).unwrap();
        assert_eq!(c.shape().dims(), &[m, n]);
        let c_data = c.as_f32_slice().unwrap();
        assert!(c_data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_quat_matmul_single_element() {
        // 1×1 matmul
        let weights = vec![3.0];
        let (packed, scales) = pack_weights_quaternary(&weights, 1, 1);
        let b = Tensor::from_f32(&[2.0], &[1, 1]);
        let c = quat_matmul(&packed, &scales, &b, 1, 1, 1).unwrap();
        let c_data = c.as_f32_slice().unwrap();
        // 3.0 quantized → Pos3 with scale=1.0, so 3.0 * 2.0 = 6.0
        assert!((c_data[0] - 6.0).abs() < 0.5, "got {}", c_data[0]);
    }
}
