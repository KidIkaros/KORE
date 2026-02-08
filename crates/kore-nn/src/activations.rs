//! Activation functions.

use kore_core::{DType, Tensor, KoreError};

/// ReLU activation: max(0, x)
pub fn relu(input: &Tensor) -> kore_core::Result<Tensor> {
    if input.dtype() != DType::F32 {
        return Err(KoreError::UnsupportedDType(input.dtype()));
    }
    input.clamp(0.0, f32::INFINITY)
}

/// GELU activation: x * Φ(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
pub fn gelu(input: &Tensor) -> kore_core::Result<Tensor> {
    if input.dtype() != DType::F32 {
        return Err(KoreError::UnsupportedDType(input.dtype()));
    }
    let data = input.contiguous();
    let slice = data.as_f32_slice().unwrap();
    let result: Vec<f32> = slice
        .iter()
        .map(|&x| {
            let inner = std::f32::consts::FRAC_2_SQRT_PI * std::f32::consts::FRAC_1_SQRT_2
                * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        })
        .collect();
    Ok(Tensor::from_f32(&result, input.shape().dims()))
}

/// SiLU (Swish) activation: x * sigmoid(x)
pub fn silu(input: &Tensor) -> kore_core::Result<Tensor> {
    if input.dtype() != DType::F32 {
        return Err(KoreError::UnsupportedDType(input.dtype()));
    }
    let data = input.contiguous();
    let slice = data.as_f32_slice().unwrap();
    let result: Vec<f32> = slice
        .iter()
        .map(|&x| x / (1.0 + (-x).exp()))
        .collect();
    Ok(Tensor::from_f32(&result, input.shape().dims()))
}

/// Sigmoid activation: 1 / (1 + exp(-x))
pub fn sigmoid(input: &Tensor) -> kore_core::Result<Tensor> {
    if input.dtype() != DType::F32 {
        return Err(KoreError::UnsupportedDType(input.dtype()));
    }
    let data = input.contiguous();
    let slice = data.as_f32_slice().unwrap();
    let result: Vec<f32> = slice
        .iter()
        .map(|&x| 1.0 / (1.0 + (-x).exp()))
        .collect();
    Ok(Tensor::from_f32(&result, input.shape().dims()))
}

/// Softmax along the last axis.
pub fn softmax(input: &Tensor) -> kore_core::Result<Tensor> {
    if input.dtype() != DType::F32 {
        return Err(KoreError::UnsupportedDType(input.dtype()));
    }
    let data = input.contiguous();
    let dims = data.shape().dims();
    let slice = data.as_f32_slice().unwrap();

    if dims.is_empty() {
        return Ok(Tensor::scalar(1.0));
    }

    let last_dim = *dims.last().unwrap();
    let batch_size = data.numel() / last_dim;
    let mut result = vec![0.0f32; data.numel()];

    for b in 0..batch_size {
        let start = b * last_dim;
        let end = start + last_dim;
        let row = &slice[start..end];

        // Numerical stability: subtract max
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();

        for (i, &x) in row.iter().enumerate() {
            result[start + i] = (x - max_val).exp() / exp_sum;
        }
    }

    Ok(Tensor::from_f32(&result, dims))
}

/// Tanh activation.
pub fn tanh(input: &Tensor) -> kore_core::Result<Tensor> {
    if input.dtype() != DType::F32 {
        return Err(KoreError::UnsupportedDType(input.dtype()));
    }
    let data = input.contiguous();
    let slice = data.as_f32_slice().unwrap();
    let result: Vec<f32> = slice.iter().map(|&x| x.tanh()).collect();
    Ok(Tensor::from_f32(&result, input.shape().dims()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let t = Tensor::from_f32(&[-1.0, 0.0, 1.0, 2.0], &[4]);
        let r = relu(&t).unwrap();
        assert_eq!(r.as_f32_slice().unwrap(), &[0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid() {
        let t = Tensor::from_f32(&[0.0], &[1]);
        let s = sigmoid(&t).unwrap();
        assert!((s.get_f32(0).unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_softmax() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        let s = softmax(&t).unwrap();
        let data = s.as_f32_slice().unwrap();
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(data[2] > data[1]);
        assert!(data[1] > data[0]);
    }

    #[test]
    fn test_softmax_batched() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
        let s = softmax(&t).unwrap();
        let data = s.as_f32_slice().unwrap();
        let sum1: f32 = data[0..3].iter().sum();
        let sum2: f32 = data[3..6].iter().sum();
        assert!((sum1 - 1.0).abs() < 1e-6);
        assert!((sum2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gelu() {
        let t = Tensor::from_f32(&[0.0, 1.0, -1.0], &[3]);
        let g = gelu(&t).unwrap();
        let data = g.as_f32_slice().unwrap();
        assert!((data[0] - 0.0).abs() < 1e-4);
        assert!(data[1] > 0.8); // GELU(1) ≈ 0.841
        assert!(data[2] < 0.0); // GELU(-1) ≈ -0.159
    }

    #[test]
    fn test_tanh() {
        let t = Tensor::from_f32(&[0.0, 1.0, -1.0], &[3]);
        let r = tanh(&t).unwrap();
        let data = r.as_f32_slice().unwrap();
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[1] - 1.0f32.tanh()).abs() < 1e-6);
    }
}
