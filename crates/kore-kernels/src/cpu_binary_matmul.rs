//! Binary (1-bit) matrix multiplication on CPU with in-memory dequantization.
//!
//! Ported from the quantize.rs `dequantize_binary` primitive and wired into a
//! full matmul path. Weights are stored packed (1 bit per weight) with a single
//! f32 scale per row, then dequantized to f32 in-memory before the matmul —
//! NOT on-disk packed format (deferred per §8).
//!
//! Uses rayon parallelism across output rows.

use rayon::prelude::*;

use kore_core::{KoreError, Tensor};

/// Binary matrix multiplication: C = A_binary_dequant @ B
///
/// - `a_packed`: packed binary weights [M, K_packed] (1 bit per weight)
/// - `a_scales`: per-row scale factors [M]
/// - `b`: dense f32 activations [K, N]
/// - `m`, `n`, `k`: logical dimensions
///
/// Returns C [M, N] as f32 tensor.
pub fn binary_matmul(
    a_packed: &[u8],
    a_scales: &[f32],
    b: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Tensor, KoreError> {
    let b_cont = b.contiguous();
    let b_data = b_cont
        .as_f32_slice()
        .ok_or_else(|| KoreError::UnsupportedDType(b.dtype()))?;

    let k_packed = k.div_ceil(8);

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

    // Pre-dequantize all rows into f32 (in-memory, not on-disk)
    let all_dequant: Vec<Vec<f32>> = (0..m)
        .map(|row| {
            let row_packed = &a_packed[row * k_packed..(row + 1) * k_packed];
            let scale = a_scales[row];
            dequantize_row_binary(row_packed, k, scale)
        })
        .collect();

    let mut c_data = vec![0.0f32; m * n];

    if m >= 16 {
        c_data
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(row, c_row)| {
                compute_binary_row(&all_dequant[row], b_data, c_row, n, k);
            });
    } else {
        for row in 0..m {
            let c_row = &mut c_data[row * n..(row + 1) * n];
            compute_binary_row(&all_dequant[row], b_data, c_row, n, k);
        }
    }

    Ok(Tensor::from_f32(&c_data, &[m, n]))
}

fn dequantize_row_binary(packed_row: &[u8], k: usize, scale: f32) -> Vec<f32> {
    let n_weights = packed_row.len() * 8;
    let mut result = Vec::with_capacity(k.min(n_weights));
    for &byte in packed_row {
        for bit in 0..8 {
            if result.len() >= k {
                break;
            }
            let val = if (byte >> bit) & 1 == 1 { 1.0 } else { -1.0 };
            result.push(val * scale);
        }
        if result.len() >= k {
            break;
        }
    }
    result
}

#[inline]
fn compute_binary_row(dequant: &[f32], b_data: &[f32], c_row: &mut [f32], n: usize, k: usize) {
    for ki in 0..k {
        let w = dequant[ki];
        if w == 0.0 {
            continue;
        }
        let b_row = &b_data[ki * n..ki * n + n];
        for (c_val, &b_val) in c_row.iter_mut().zip(b_row.iter()) {
            *c_val += w * b_val;
        }
    }
}

/// Pack f32 weights into binary format with per-row scales.
///
/// Returns (packed_bytes, scales). Each row is ceil(K/8) bytes.
pub fn pack_weights_binary(weights: &[f32], m: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let k_packed = k.div_ceil(8);
    let mut packed = vec![0u8; m * k_packed];
    let mut scales = vec![0.0f32; m];

    for row in 0..m {
        let row_start = row * k;
        let row_data = &weights[row_start..row_start + k];

        let abs_max = row_data.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
        let scale = if abs_max < 1e-8 { 1.0 } else { abs_max };
        scales[row] = scale;

        for i in 0..k {
            if row_data[i] >= 0.0 {
                packed[row * k_packed + i / 8] |= 1 << (i % 8);
            }
        }
    }

    (packed, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_binary() {
        let weights = vec![1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0];
        let (packed, scales) = pack_weights_binary(&weights, 2, 4);
        assert_eq!(packed.len(), 2);
        assert_eq!(scales.len(), 2);
    }

    #[test]
    fn test_binary_matmul_identity() {
        let weights = vec![1.0, -0.1, -0.1, 1.0];
        let (packed, scales) = pack_weights_binary(&weights, 2, 2);

        let b = Tensor::from_f32(&[2.0, 0.0, 0.0, 3.0], &[2, 2]);
        let c = binary_matmul(&packed, &scales, &b, 2, 2, 2).unwrap();

        let c_data = c.as_f32_slice().unwrap();
        assert!((c_data[0] - 2.0).abs() < 1.0, "got {}", c_data[0]);
        assert!((c_data[3] - 3.0).abs() < 1.0, "got {}", c_data[3]);
    }

    #[test]
    fn test_binary_matmul_larger() {
        let m = 8;
        let k = 20;
        let n = 4;

        let weights: Vec<f32> = (0..m * k).map(|i| ((i % 3) as f32 - 1.0)).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.2).collect();

        let (packed, scales) = pack_weights_binary(&weights, m, k);
        let b = Tensor::from_f32(&b_data, &[k, n]);

        let c = binary_matmul(&packed, &scales, &b, m, n, k).unwrap();
        assert_eq!(c.shape().dims(), &[m, n]);

        let c_data = c.as_f32_slice().unwrap();
        assert!(c_data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_binary_matmul_non_aligned_k() {
        let m = 2;
        let k = 13;
        let n = 3;

        let weights: Vec<f32> = (0..m * k).map(|i| ((i % 4) as f32 - 1.5)).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();

        let (packed, scales) = pack_weights_binary(&weights, m, k);
        let b = Tensor::from_f32(&b_data, &[k, n]);

        let c = binary_matmul(&packed, &scales, &b, m, n, k).unwrap();
        assert_eq!(c.shape().dims(), &[m, n]);
        let c_data = c.as_f32_slice().unwrap();
        assert!(c_data.iter().all(|v| v.is_finite()));
    }
}
