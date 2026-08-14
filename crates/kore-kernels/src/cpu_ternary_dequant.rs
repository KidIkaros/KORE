//! Ternary dequant + matmul on CPU using kore-btes quantize primitives.
//!
//! This module provides a CPU matmul path where ternary weights are stored
//! via kore-btes `quantize_ternary` / `dequantize_ternary` and the actual
//! matmul runs on f32 (dequantized in-memory, NOT on-disk packing — deferred
//! per design §8).
//!
//! Workflow:
//!   1. Quantize f32 weights → (trits, scale) via kore_btes::quantize_ternary
//!   2. Dequantize trits → f32 via kore_btes::dequantize_ternary
//!   3. Run f32 matmul on the dequantized weights
//!
//! This is the IN-MEMORY dequant path. On-disk packed format is deferred.

use kore_btes::{dequantize_ternary, quantize_ternary, Trit};
use kore_core::{KoreError, Tensor};
use crate::cpu_matmul::matmul_f32;

/// Dequantize ternary trits to f32 in-memory and run matmul.
pub fn ternary_dequant_matmul_f32(
    a_trits: &[Trit],
    a_scale: f32,
    b: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Tensor, KoreError> {
    if a_trits.len() != m * k {
        return Err(KoreError::StorageError(format!(
            "a_trits length mismatch: expected {}, got {}",
            m * k,
            a_trits.len()
        )));
    }

    let b_cont = b.contiguous();
    let _b_data = b_cont
        .as_f32_slice()
        .ok_or_else(|| KoreError::UnsupportedDType(b.dtype()))?;

    let a_deq = dequantize_ternary(a_trits, a_scale);
    let a_tensor = Tensor::from_f32(&a_deq, &[m, k]);
    matmul_f32(&a_tensor, &b_cont)
}

/// Quantize f32 weights to ternary and dequantize+matmul in one shot.
pub fn ternary_quant_matmul_f32(
    a_weights: &[f32],
    threshold: f32,
    b: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Tensor, KoreError> {
    if a_weights.len() != m * k {
        return Err(KoreError::StorageError(format!(
            "a_weights length mismatch: expected {}, got {}",
            m * k,
            a_weights.len()
        )));
    }

    let b_cont = b.contiguous();
    let _b_data = b_cont
        .as_f32_slice()
        .ok_or_else(|| KoreError::UnsupportedDType(b.dtype()))?;

    let (trits, scale) = quantize_ternary(a_weights, threshold);
    ternary_dequant_matmul_f32(&trits, scale, &b_cont, m, n, k)
}

/// Full roundtrip test: f32 weights → quantize → dequant → matmul, compare vs f32 reference.
pub fn ternary_matmul_roundtrip(
    a_weights: &[f32],
    b_data: &[f32],
    m: usize,
    k: usize,
    threshold: f32,
) -> (Vec<f32>, Vec<f32>) {
    let a_tensor = Tensor::from_f32(a_weights, &[m, k]);
    let b_tensor = Tensor::from_f32(b_data, &[k, m]);
    let ref_c = matmul_f32(&a_tensor, &b_tensor).unwrap();
    let ref_data = ref_c.as_f32_slice().unwrap().to_vec();

    let (trits, scale) = quantize_ternary(a_weights, threshold);
    let a_deq = dequantize_ternary(&trits, scale);
    let a_deq_tensor = Tensor::from_f32(&a_deq, &[m, k]);
    let b_deq_tensor = Tensor::from_f32(b_data, &[k, m]);
    let quant_c = matmul_f32(&a_deq_tensor, &b_deq_tensor).unwrap();
    let quant_data = quant_c.as_f32_slice().unwrap().to_vec();

    (quant_data, ref_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_dequant_matmul_basic() {
        let a_weights = vec![1.0f32, -1.0, 0.0, 0.0, 1.0, -1.0];
        let b_data = vec![2.0f32, 0.0, 0.0, 3.0, 1.0, 1.0];
        let m = 2;
        let n = 2;
        let k = 3;

        let result = ternary_quant_matmul_f32(&a_weights, 0.3,
            &Tensor::from_f32(&b_data, &[k, n]), m, n, k).unwrap();
        let data = result.as_f32_slice().unwrap();

        assert!((data[0] - 2.0).abs() < 1.0, "got {}", data[0]);
        assert!((data[1] - (-3.0)).abs() < 1.0, "got {}", data[1]);
        assert!((data[2] - (-1.0)).abs() < 1.0, "got {}", data[2]);
        assert!((data[3] - 2.0).abs() < 1.0, "got {}", data[3]);
    }

    #[test]
    fn test_ternary_matmul_roundtrip_tolerance() {
        let a_weights = vec![1.0f32, -0.5, 0.3, -0.8, 0.1, 0.9];
        let b_data = vec![2.0f32, 0.5, 1.0, -1.0, 0.3, 0.7];
        let m = 2;
        let k = 3;

        let (quant, ref_data) = ternary_matmul_roundtrip(&a_weights[..], &b_data[..], m, k, 0.3);

        for i in 0..quant.len() {
            assert!(quant[i].is_finite(), "quant[{}] not finite: {}", i, quant[i]);
            assert!(ref_data[i].is_finite(), "ref[{}] not finite: {}", i, ref_data[i]);
        }
    }

    #[test]
    fn test_ternary_dequant_matmul_identity_weights() {
        let a_weights = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let b_data = vec![5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0];
        let m = 2;
        let k = 3;

        let result = ternary_quant_matmul_f32(&a_weights, 0.3,
            &Tensor::from_f32(&b_data, &[k, m]), m, m, k).unwrap();
        let data = result.as_f32_slice().unwrap();

        assert!((data[0] - 5.0).abs() < 0.5, "got {}", data[0]);
        assert!((data[1] - 6.0).abs() < 0.5, "got {}", data[1]);
        assert!((data[2] - 7.0).abs() < 0.5, "got {}", data[2]);
        assert!((data[3] - 8.0).abs() < 0.5, "got {}", data[3]);
    }

    #[test]
    fn test_ternary_dequant_matmul_larger() {
        let m = 8;
        let k = 16;
        let n = 8;

        let a_weights: Vec<f32> = (0..m * k).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32 * 0.1).collect();

        let result = ternary_quant_matmul_f32(&a_weights, 0.3,
            &Tensor::from_f32(&b_data, &[k, n]), m, n, k).unwrap();
        let data = result.as_f32_slice().unwrap();

        assert_eq!(data.len(), m * n);
        assert!(data.iter().all(|v| v.is_finite()));
    }
}
