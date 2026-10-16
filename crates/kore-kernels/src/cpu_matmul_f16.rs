//! Half-precision (f16) matrix multiplication on CPU.
//!
//! Converts f16 inputs to f32, runs tiled matmul, optionally converts back to f16.

use crate::cpu_matmul::matmul_f32;
use half::f16;
use kore_core::{KoreError, Tensor};

/// 2D matrix multiplication: C = A @ B, where A and B are f16 tensors.
/// Returns f32 result by default. Use `output_f16=true` to get f16 back.
pub fn matmul_f16(a: &Tensor, b: &Tensor, output_f16: bool) -> Result<Tensor, KoreError> {
    if a.dtype() != kore_core::DType::F16 || b.dtype() != kore_core::DType::F16 {
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

    // Convert f16 -> f32
    let a_bytes = a.storage_ref().as_bytes();
    let b_bytes = b.storage_ref().as_bytes();

    let a_f32: Vec<f32> = a_bytes
        .chunks(2)
        .take(m * k)
        .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect();

    let b_f32: Vec<f32> = b_bytes
        .chunks(2)
        .take(k * n)
        .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect();

    let a_tensor = Tensor::from_f32(&a_f32, &[m, k]);
    let b_tensor = Tensor::from_f32(&b_f32, &[k, n]);

    // Run f32 matmul
    let c_f32 = matmul_f32(&a_tensor, &b_tensor)?;

    if output_f16 {
        // Convert f32 -> f16
        let c_data = c_f32.as_f32_slice().unwrap();
        let c_f16: Vec<u8> = c_data
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect();

        let storage = kore_core::Storage::from_bytes(kore_core::DType::F16, m * n, c_f16)
            .expect("f16 storage creation failed");
        Ok(Tensor::from_storage(storage, &[m, n]))
    } else {
        Ok(c_f32)
    }
}

/// F16 matmul with explicit dimension inputs (for dequant paths).
pub fn matmul_f16_raw(
    a_f16: &[u8],
    b_f16: &[u8],
    m: usize,
    n: usize,
    k: usize,
    output_f16: bool,
) -> Result<Tensor, KoreError> {
    let a_f32: Vec<f32> = a_f16
        .chunks(2)
        .take(m * k)
        .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect();

    let b_f32: Vec<f32> = b_f16
        .chunks(2)
        .take(k * n)
        .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect();

    let a_tensor = Tensor::from_f32(&a_f32, &[m, k]);
    let b_tensor = Tensor::from_f32(&b_f32, &[k, n]);
    let c_f32 = matmul_f32(&a_tensor, &b_tensor)?;

    if output_f16 {
        let c_data = c_f32.as_f32_slice().unwrap();
        let c_f16: Vec<u8> = c_data
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect();

        let storage = kore_core::Storage::from_bytes(kore_core::DType::F16, m * n, c_f16)
            .expect("f16 storage creation failed");
        Ok(Tensor::from_storage(storage, &[m, n]))
    } else {
        Ok(c_f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_matmul_basic() {
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let a_f16: Vec<u8> = a_data
            .iter()
            .map(|&v| f16::from_f32(v).to_le_bytes())
            .flatten()
            .collect();

        let b_data = vec![5.0f32, 6.0, 7.0, 8.0];
        let b_f16: Vec<u8> = b_data
            .iter()
            .map(|&v| f16::from_f32(v).to_le_bytes())
            .flatten()
            .collect();

        let a = Tensor::from_storage(
            kore_core::Storage::from_bytes(kore_core::DType::F16, 4, a_f16).expect("storage"),
            &[2, 2],
        );
        let b = Tensor::from_storage(
            kore_core::Storage::from_bytes(kore_core::DType::F16, 4, b_f16).expect("storage"),
            &[2, 2],
        );

        let c = matmul_f16(&a, &b, false).unwrap();
        let c_data = c.as_f32_slice().unwrap();
        assert!((c_data[0] - 19.0).abs() < 0.1, "got {}", c_data[0]);
        assert!((c_data[1] - 22.0).abs() < 0.1, "got {}", c_data[1]);
        assert!((c_data[2] - 43.0).abs() < 0.1, "got {}", c_data[2]);
        assert!((c_data[3] - 50.0).abs() < 0.1, "got {}", c_data[3]);
    }

    #[test]
    fn test_f16_matmul_output_f16() {
        let a_data = vec![1.0f32, 0.0, 0.0, 1.0];
        let a_f16: Vec<u8> = a_data
            .iter()
            .map(|&v| f16::from_f32(v).to_le_bytes())
            .flatten()
            .collect();

        let b_data = vec![2.0f32, 3.0, 4.0, 5.0];
        let b_f16: Vec<u8> = b_data
            .iter()
            .map(|&v| f16::from_f32(v).to_le_bytes())
            .flatten()
            .collect();

        let a = Tensor::from_storage(
            kore_core::Storage::from_bytes(kore_core::DType::F16, 4, a_f16).expect("storage"),
            &[2, 2],
        );
        let b = Tensor::from_storage(
            kore_core::Storage::from_bytes(kore_core::DType::F16, 4, b_f16).expect("storage"),
            &[2, 2],
        );

        let c = matmul_f16(&a, &b, true).unwrap();
        assert_eq!(c.dtype(), kore_core::DType::F16);
        assert_eq!(c.shape().dims(), &[2, 2]);
    }
}
