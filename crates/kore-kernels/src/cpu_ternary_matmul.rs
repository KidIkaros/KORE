//! Ternary (balanced) matrix multiplication on CPU.
//!
//! Uses kore-btes `TernaryWord64` for 64-trit parallel dot products.
//! Packed weights: 5 trits per byte (base-243 encoding).
//! Ternary values: {-1, 0, +1}.
//!
//! Optimizations:
//! - Rayon parallelism across output rows
//! - Pre-unpacked i8 trits with branchless multiply (trit * activation)
//! - Column-blocked accumulation for cache locality

use rayon::prelude::*;

use kore_btes::vtalu::TernaryWord64;
use kore_btes::encoder::{Trit, decode_trits, encode_trits};
use kore_core::{KoreError, Tensor};

/// Minimum rows before we use rayon parallelism.
const PAR_ROW_THRESHOLD: usize = 16;

/// Ternary matrix multiplication: C = A_ternary @ B
///
/// - `a_packed`: base-243 packed ternary weights [M, K_packed_bytes]
/// - `a_scales`: per-row scale factors [M]
/// - `b`: dense f32 activations [K, N]
/// - `m`, `n`, `k`: logical dimensions
///
/// Returns C [M, N] as f32 tensor.
pub fn ternary_matmul(
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

    let k_packed = (k + 4) / 5; // base-243: 5 trits per byte

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

    // Pre-unpack all rows into i8 trits (avoids repeated decode in inner loop)
    let all_trits: Vec<Vec<i8>> = (0..m)
        .map(|row| unpack_row_trits(&a_packed[row * k_packed..(row + 1) * k_packed], k))
        .collect();

    let mut c_data = vec![0.0f32; m * n];

    if m >= PAR_ROW_THRESHOLD {
        // Parallel: each row computed independently
        c_data
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(row, c_row)| {
                let scale = a_scales[row];
                let trits = &all_trits[row];
                compute_row(trits, b_data, c_row, scale, n, k);
            });
    } else {
        // Sequential for small M
        for row in 0..m {
            let scale = a_scales[row];
            let trits = &all_trits[row];
            let c_row = &mut c_data[row * n..(row + 1) * n];
            compute_row(trits, b_data, c_row, scale, n, k);
        }
    }

    Ok(Tensor::from_f32(&c_data, &[m, n]))
}

/// Unpack a packed ternary row into i8 trits.
#[inline]
fn unpack_row_trits(packed_row: &[u8], k: usize) -> Vec<i8> {
    let mut trits = Vec::with_capacity(k + 4);
    for &byte in packed_row {
        let decoded = decode_trits(byte);
        for &t in &decoded {
            trits.push(t as i8);
        }
    }
    trits.truncate(k);
    trits
}

/// Compute one output row: c_row[col] = scale * sum(trits[ki] * b[ki, col])
#[inline]
fn compute_row(trits: &[i8], b_data: &[f32], c_row: &mut [f32], scale: f32, n: usize, k: usize) {
    // Accumulate: for each trit position, scatter trit*activation into output columns.
    // This is "A-stationary": iterate over K dimension, accumulate into N outputs.
    // Better cache behavior when N is small (typical for decode: N=1 or small batch).
    for ki in 0..k {
        let t = trits[ki];
        if t == 0 { continue; }
        let t_f32 = t as f32;
        let b_row = &b_data[ki * n..ki * n + n];
        for col in 0..n {
            c_row[col] += t_f32 * b_row[col];
        }
    }
    // Apply scale
    for col in 0..n {
        c_row[col] *= scale;
    }
}

/// Ternary matmul using TernaryWord64 dot products for integer accumulation.
///
/// This variant transposes B so each column is contiguous, then uses
/// the VT-ALU dot product for the inner loop. Best when B can also be
/// quantized to ternary (e.g., ternary-ternary matmul).
pub fn ternary_ternary_matmul(
    a_packed: &[u8],
    a_scales: &[f32],
    b_packed: &[u8],
    b_scales: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Tensor, KoreError> {
    let k_packed = (k + 4) / 5;

    if a_packed.len() < m * k_packed || b_packed.len() < n * k_packed {
        return Err(KoreError::StorageError(
            "packed arrays too small".to_string(),
        ));
    }

    // Unpack rows into TernaryWord64 chunks
    let k_words = (k + 63) / 64;

    let unpack_row = |packed_row: &[u8]| -> Vec<TernaryWord64> {
        let mut trits = Vec::with_capacity(k_words * 64);
        for &byte in packed_row {
            let decoded = decode_trits(byte);
            for &t in &decoded {
                trits.push(t as i8);
            }
        }
        trits.resize(k_words * 64, 0);
        trits
            .chunks_exact(64)
            .map(|chunk| TernaryWord64::from_trits(chunk))
            .collect()
    };

    // Pre-unpack all rows
    let a_words: Vec<Vec<TernaryWord64>> = (0..m)
        .map(|row| unpack_row(&a_packed[row * k_packed..(row + 1) * k_packed]))
        .collect();

    let b_words: Vec<Vec<TernaryWord64>> = (0..n)
        .map(|col| unpack_row(&b_packed[col * k_packed..(col + 1) * k_packed]))
        .collect();

    let mut c_data = vec![0.0f32; m * n];

    for row in 0..m {
        let a_scale = a_scales[row];
        for col in 0..n {
            let b_scale = b_scales[col];

            // Integer dot product via VT-ALU
            let mut int_acc: i64 = 0;
            for w in 0..k_words {
                int_acc += a_words[row][w].dot(b_words[col][w]);
            }

            c_data[row * n + col] = int_acc as f32 * a_scale * b_scale;
        }
    }

    Ok(Tensor::from_f32(&c_data, &[m, n]))
}

/// Pack f32 weights into ternary format with per-row scales.
///
/// Returns (packed_bytes, scales).
pub fn pack_weights_ternary(weights: &[f32], m: usize, k: usize, threshold: f32) -> (Vec<u8>, Vec<f32>) {
    let k_packed = (k + 4) / 5;
    let mut packed = vec![0u8; m * k_packed];
    let mut scales = vec![0.0f32; m];

    for row in 0..m {
        let row_data = &weights[row * k..(row + 1) * k];

        let abs_max = row_data.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
        let scale = if abs_max < 1e-8 { 1.0 } else { abs_max };
        scales[row] = scale;

        // Quantize to trits
        let mut trits = Vec::with_capacity(k);
        for &w in row_data {
            let normalized = w / scale;
            let trit = if normalized > threshold {
                Trit::Pos
            } else if normalized < -threshold {
                Trit::Neg
            } else {
                Trit::Zero
            };
            trits.push(trit);
        }

        // Pack into base-243 bytes
        for (kp, chunk) in trits.chunks(5).enumerate() {
            let mut block = [Trit::Zero; 5];
            for (i, &t) in chunk.iter().enumerate() {
                block[i] = t;
            }
            packed[row * k_packed + kp] = encode_trits(&block);
        }
    }

    (packed, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_ternary() {
        let weights = vec![1.0, -1.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0];
        let (packed, scales) = pack_weights_ternary(&weights, 2, 4, 0.3);
        assert_eq!(packed.len(), 2); // 2 rows, ceil(4/5)=1 byte each
        assert_eq!(scales.len(), 2);
        assert!(scales[0] > 0.0);
    }

    #[test]
    fn test_ternary_matmul_identity() {
        // A = [[1, 0], [0, 1]] (ternary-friendly)
        // B = [[2, 0], [0, 3]]
        // C = A @ B = [[2, 0], [0, 3]]
        let weights = vec![1.0, 0.0, 0.0, 1.0];
        let (packed, scales) = pack_weights_ternary(&weights, 2, 2, 0.3);

        let b = Tensor::from_f32(&[2.0, 0.0, 0.0, 3.0], &[2, 2]);
        let c = ternary_matmul(&packed, &scales, &b, 2, 2, 2).unwrap();

        let c_data = c.as_f32_slice().unwrap();
        assert!((c_data[0] - 2.0).abs() < 0.1, "got {}", c_data[0]);
        assert!((c_data[1]).abs() < 0.1, "got {}", c_data[1]);
        assert!((c_data[2]).abs() < 0.1, "got {}", c_data[2]);
        assert!((c_data[3] - 3.0).abs() < 0.1, "got {}", c_data[3]);
    }

    #[test]
    fn test_ternary_matmul_larger() {
        let m = 8;
        let k = 20;
        let n = 4;

        let weights: Vec<f32> = (0..m * k).map(|i| ((i % 3) as f32 - 1.0)).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.2).collect();

        let (packed, scales) = pack_weights_ternary(&weights, m, k, 0.3);
        let b = Tensor::from_f32(&b_data, &[k, n]);

        let c = ternary_matmul(&packed, &scales, &b, m, n, k).unwrap();
        assert_eq!(c.shape().dims(), &[m, n]);

        let c_data = c.as_f32_slice().unwrap();
        assert!(c_data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_ternary_ternary_matmul() {
        // Both A and B are ternary
        let a_weights = vec![1.0, -1.0, 0.0, 1.0];
        let b_weights = vec![1.0, 0.0, -1.0, 1.0]; // B is [N, K] (transposed)

        let (a_packed, a_scales) = pack_weights_ternary(&a_weights, 2, 2, 0.3);
        let (b_packed, b_scales) = pack_weights_ternary(&b_weights, 2, 2, 0.3);

        let c = ternary_ternary_matmul(&a_packed, &a_scales, &b_packed, &b_scales, 2, 2, 2).unwrap();
        assert_eq!(c.shape().dims(), &[2, 2]);

        let c_data = c.as_f32_slice().unwrap();
        assert!(c_data.iter().all(|v| v.is_finite()));
    }
}
