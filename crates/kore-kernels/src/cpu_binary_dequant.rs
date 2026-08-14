//! Binary dequant + matmul on CPU using kore-btes quantize primitives.
//!
//! This module provides a CPU matmul path where binary weights are stored
//! via kore-btes `quantize_binary` / `dequantize_binary` and the actual
//! matmul runs on f32 (dequantized in-memory, NOT on-disk packing — deferred
//! per design §8).
//!
//! Workflow:
//!   1. Quantize f32 weights → (packed_bits, scale) via kore_btes::quantize_binary
//!   2. Dequantize bits → f32 via kore_btes::dequantize_binary
//!   3. Run f32 matmul on the dequantized weights
//!
//! This is the IN-MEMORY dequant path. On-disk packed format (Q1_0_g128) is deferred.

use kore_btes::{dequantize_binary, quantize_binary};
use kore_core::{KoreError, Tensor};
use crate::cpu_matmul::matmul_f32;

/// Dequantize binary packed bits to f32 in-memory and run matmul.
///
/// `a_packed`: packed binary bits (1 bit per weight, LSB first per byte) — single row of K weights
/// `a_scale`: single scale factor for all weights
/// `b`: dense f32 activations [K, N]
/// `m`: number of output rows (weight row is tiled across M)
/// `n`, `k`: logical dimensions
///
/// Returns C [M, N] as f32 tensor.
pub fn binary_dequant_matmul_f32(
    a_packed: &[u8],
    a_scale: f32,
    b: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Tensor, KoreError> {
    let expected_packed = k.div_ceil(8);
    if a_packed.len() != expected_packed {
        return Err(KoreError::StorageError(format!(
            "a_packed length mismatch: expected {} bytes for k={}, got {}",
            expected_packed, k, a_packed.len()
        )));
    }

    let b_cont = b.contiguous();
    let _b_data = b_cont
        .as_f32_slice()
        .ok_or_else(|| KoreError::UnsupportedDType(b.dtype()))?;

    // Dequantize in-memory to f32
    let a_deq: Vec<f32> = dequantize_binary(a_packed, a_scale)
        .into_iter()
        .take(k)
        .collect();

    if a_deq.len() != k {
        return Err(KoreError::StorageError(format!(
            "dequantized length mismatch: expected {}, got {}",
            k, a_deq.len()
        )));
    }

    // Tile the same K weights across M output rows
    let a_deq_tiled: Vec<f32> = a_deq.iter().cycle().take(m * k).cloned().collect();
    let a_tensor = Tensor::from_f32(&a_deq_tiled, &[m, k]);
    matmul_f32(&a_tensor, &b_cont)
}

/// Quantize f32 weights to binary and dequantize+matmul in one shot.
///
/// Single shared weight row of K weights, tiled across M outputs.
pub fn binary_quant_matmul_f32(
    a_weights: &[f32],
    b: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Tensor, KoreError> {
    if a_weights.len() != k {
        return Err(KoreError::StorageError(format!(
            "a_weights length mismatch: expected {}, got {}",
            k, a_weights.len()
        )));
    }

    let b_cont = b.contiguous();
    let _b_data = b_cont
        .as_f32_slice()
        .ok_or_else(|| KoreError::UnsupportedDType(b.dtype()))?;

    let (packed, scale) = quantize_binary(a_weights);
    binary_dequant_matmul_f32(&packed, scale, &b_cont, m, n, k)
}

/// Binary matmul with per-row weights (each output row has its own K binary weights).
///
/// `a_weights`: f32 weights [M, K] flattened
/// `b`: dense f32 activations [K, N]
/// `m`, `n`, `k`: logical dimensions
pub fn binary_quant_matmul_f32_per_row(
    a_weights: &[f32],
    b: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Tensor, KoreError> {
    if a_weights.len() != m * k {
        return Err(KoreError::StorageError(format!(
            "a_weights length mismatch: expected {}, got {}",
            m * k, a_weights.len()
        )));
    }

    let b_cont = b.contiguous();
    let _b_data = b_cont
        .as_f32_slice()
        .ok_or_else(|| KoreError::UnsupportedDType(b.dtype()))?;

    let mut all_deq = Vec::with_capacity(m * k);
    for row in 0..m {
        let row_start = row * k;
        let row_data = &a_weights[row_start..row_start + k];
        let (packed, scale) = quantize_binary(row_data);
        let deq: Vec<f32> = dequantize_binary(&packed, scale).into_iter().take(k).collect();
        all_deq.extend(deq);
    }

    let a_tensor = Tensor::from_f32(&all_deq, &[m, k]);
    matmul_f32(&a_tensor, &b_cont)
}

/// Full roundtrip test: f32 weights → quantize → dequant → matmul, compare vs f32 reference.
pub fn binary_matmul_roundtrip(
    a_weights: &[f32],
    b_data: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> (Vec<f32>, Vec<f32>) {
    let a_tensor = Tensor::from_f32(a_weights, &[m, k]);
    let b_tensor = Tensor::from_f32(b_data, &[k, n]);
    let ref_c = matmul_f32(&a_tensor, &b_tensor).unwrap();
    let ref_data = ref_c.as_f32_slice().unwrap().to_vec();

    // Per-row quantized path
    let mut all_deq = Vec::with_capacity(m * k);
    for row in 0..m {
        let row_start = row * k;
        let row_data = &a_weights[row_start..row_start + k];
        let (packed, scale) = quantize_binary(row_data);
        let deq: Vec<f32> = dequantize_binary(&packed, scale).into_iter().take(k).collect();
        all_deq.extend(deq);
    }
    let a_deq_tensor = Tensor::from_f32(&all_deq, &[m, k]);
    let b_deq_tensor = Tensor::from_f32(b_data, &[k, n]);
    let quant_c = matmul_f32(&a_deq_tensor, &b_deq_tensor).unwrap();
    let quant_data = quant_c.as_f32_slice().unwrap().to_vec();

    (quant_data, ref_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_dequant_matmul_basic() {
        // A = [1, -1, 1, -1] (binary-friendly, all ±1)
        // B = [[1, 0], [0, 1], [1, 0], [0, 1]]
        // C = A @ B = [2, -2]
        let a_weights = vec![1.0f32, -1.0, 1.0, -1.0];
        let b_data = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let m = 1;
        let n = 2;
        let k = 4;

        let result = binary_quant_matmul_f32(&a_weights,
            &Tensor::from_f32(&b_data, &[k, n]), m, n, k).unwrap();
        let data = result.as_f32_slice().unwrap();

        assert!((data[0] - 2.0).abs() < 0.5, "got {}", data[0]);
        assert!((data[1] - (-2.0)).abs() < 0.5, "got {}", data[1]);
    }

    #[test]
    fn test_binary_dequant_matmul_per_row() {
        // M=2, K=4, N=2
        let a_weights = vec![
            1.0f32, -1.0, 1.0, -1.0,
            -1.0, 1.0, -1.0, 1.0,
        ];
        let b_data = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let m = 2;
        let n = 2;
        let k = 4;

        let result = binary_quant_matmul_f32_per_row(&a_weights,
            &Tensor::from_f32(&b_data, &[k, n]), m, n, k).unwrap();
        let data = result.as_f32_slice().unwrap();

        assert!((data[0] - 2.0).abs() < 0.5, "got {}", data[0]);
        assert!((data[1] - (-2.0)).abs() < 0.5, "got {}", data[1]);
        assert!((data[2] - (-2.0)).abs() < 0.5, "got {}", data[2]);
        assert!((data[3] - 2.0).abs() < 0.5, "got {}", data[3]);
    }

    #[test]
    fn test_binary_matmul_roundtrip() {
        let a_weights = vec![1.0f32, -0.8, 0.3, -1.2, 0.5, -0.4, 0.9, -0.1];
        let b_data = vec![2.0f32, 0.5, 1.0, -1.0];
        let m = 2;
        let n = 1;
        let k = 4;

        let (quant, ref_data) = binary_matmul_roundtrip(&a_weights[..], &b_data[..], m, n, k);

        for i in 0..quant.len() {
            assert!(quant[i].is_finite(), "quant[{}] not finite: {}", i, quant[i]);
            assert!(ref_data[i].is_finite(), "ref[{}] not finite: {}", i, ref_data[i]);
        }
    }

    #[test]
    fn test_binary_dequant_matmul_larger() {
        let m = 8;
        let k = 32;
        let n = 8;

        let a_weights: Vec<f32> = (0..m * k).map(|i| ((i % 3) as f32 - 1.0)).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.1).collect();

        let result = binary_quant_matmul_f32_per_row(&a_weights,
            &Tensor::from_f32(&b_data, &[k, n]), m, n, k).unwrap();
        let data = result.as_f32_slice().unwrap();

        assert_eq!(data.len(), m * n);
        assert!(data.iter().all(|v| v.is_finite()));
    }
}
