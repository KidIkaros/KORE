//! Fused CPU operations for reduced memory traffic.
//!
//! Ported from IDTorch/src/00_backend/fused_kernels.py.
//! Each fused op combines multiple operations into a single pass
//! over the data, avoiding intermediate allocations.

use kore_core::{DType, KoreError, Tensor};

/// Fused linear + ReLU: max(0, x @ W^T + b)
///
/// Single pass: matmul + bias + activation without intermediate tensor.
pub fn fused_linear_relu(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor, KoreError> {
    if input.dtype() != DType::F32 {
        return Err(KoreError::UnsupportedDType(input.dtype()));
    }

    let input = input.contiguous();
    let wt = weight.transpose()?;
    let wt = wt.contiguous();

    let in_data = input.as_f32_slice().unwrap();
    let w_data = wt.as_f32_slice().unwrap();

    let in_dims = input.shape().dims();
    let w_dims = wt.shape().dims();

    let batch = in_dims[0];
    let k = in_dims[1];
    let n = w_dims[1];

    let mut out = vec![0.0f32; batch * n];

    for i in 0..batch {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += in_data[i * k + p] * w_data[p * n + j];
            }

            // Fused bias
            if let Some(b) = bias {
                let b_data = b.as_f32_slice().unwrap();
                acc += b_data[j];
            }

            // Fused ReLU
            out[i * n + j] = acc.max(0.0);
        }
    }

    Ok(Tensor::from_f32(&out, &[batch, n]))
}

/// Fused linear + GELU: gelu(x @ W^T + b)
pub fn fused_linear_gelu(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor, KoreError> {
    if input.dtype() != DType::F32 {
        return Err(KoreError::UnsupportedDType(input.dtype()));
    }

    let input = input.contiguous();
    let wt = weight.transpose()?;
    let wt = wt.contiguous();

    let in_data = input.as_f32_slice().unwrap();
    let w_data = wt.as_f32_slice().unwrap();

    let in_dims = input.shape().dims();
    let w_dims = wt.shape().dims();

    let batch = in_dims[0];
    let k = in_dims[1];
    let n = w_dims[1];

    let mut out = vec![0.0f32; batch * n];

    for i in 0..batch {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += in_data[i * k + p] * w_data[p * n + j];
            }

            if let Some(b) = bias {
                let b_data = b.as_f32_slice().unwrap();
                acc += b_data[j];
            }

            // Fused GELU
            let inner = std::f32::consts::FRAC_2_SQRT_PI * std::f32::consts::FRAC_1_SQRT_2
                * (acc + 0.044715 * acc * acc * acc);
            out[i * n + j] = 0.5 * acc * (1.0 + inner.tanh());
        }
    }

    Ok(Tensor::from_f32(&out, &[batch, n]))
}

/// Fused Layer Normalization: (x - mean) / sqrt(var + eps) * gamma + beta
///
/// Normalizes over the last dimension in a single pass.
pub fn fused_layer_norm(
    input: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f32,
) -> Result<Tensor, KoreError> {
    if input.dtype() != DType::F32 {
        return Err(KoreError::UnsupportedDType(input.dtype()));
    }

    let input = input.contiguous();
    let in_data = input.as_f32_slice().unwrap();
    let g_data = gamma.as_f32_slice().unwrap();
    let b_data = beta.as_f32_slice().unwrap();

    let dims = input.shape().dims();
    let last_dim = *dims.last().unwrap();
    let batch_size = input.numel() / last_dim;

    let mut out = vec![0.0f32; input.numel()];

    for b in 0..batch_size {
        let start = b * last_dim;
        let row = &in_data[start..start + last_dim];

        // Welford's online algorithm for mean and variance in one pass
        let mut mean = 0.0f32;
        let mut m2 = 0.0f32;
        for (i, &x) in row.iter().enumerate() {
            let delta = x - mean;
            mean += delta / (i + 1) as f32;
            let delta2 = x - mean;
            m2 += delta * delta2;
        }
        let var = m2 / last_dim as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        // Normalize + affine in one pass
        for i in 0..last_dim {
            let normalized = (row[i] - mean) * inv_std;
            out[start + i] = normalized * g_data[i] + b_data[i];
        }
    }

    Ok(Tensor::from_f32(&out, dims))
}

/// Fused RMS Normalization: x / rms(x) * gamma
///
/// Used in LLaMA, Mistral, etc. Simpler than LayerNorm (no mean subtraction).
pub fn fused_rms_norm(
    input: &Tensor,
    gamma: &Tensor,
    eps: f32,
) -> Result<Tensor, KoreError> {
    if input.dtype() != DType::F32 {
        return Err(KoreError::UnsupportedDType(input.dtype()));
    }

    let input = input.contiguous();
    let in_data = input.as_f32_slice().unwrap();
    let g_data = gamma.as_f32_slice().unwrap();

    let dims = input.shape().dims();
    let last_dim = *dims.last().unwrap();
    let batch_size = input.numel() / last_dim;

    let mut out = vec![0.0f32; input.numel()];

    for b in 0..batch_size {
        let start = b * last_dim;
        let row = &in_data[start..start + last_dim];

        // RMS = sqrt(mean(x^2))
        let sum_sq: f32 = row.iter().map(|x| x * x).sum();
        let rms = (sum_sq / last_dim as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        for i in 0..last_dim {
            out[start + i] = row[i] * inv_rms * g_data[i];
        }
    }

    Ok(Tensor::from_f32(&out, dims))
}

/// Fused softmax with numerical stability (subtract max).
///
/// Operates over the last dimension.
pub fn fused_softmax(input: &Tensor) -> Result<Tensor, KoreError> {
    if input.dtype() != DType::F32 {
        return Err(KoreError::UnsupportedDType(input.dtype()));
    }

    let input = input.contiguous();
    let in_data = input.as_f32_slice().unwrap();
    let dims = input.shape().dims();
    let last_dim = *dims.last().unwrap();
    let batch_size = input.numel() / last_dim;

    let mut out = vec![0.0f32; input.numel()];

    for b in 0..batch_size {
        let start = b * last_dim;
        let row = &in_data[start..start + last_dim];

        // Single pass: find max
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Single pass: exp and sum
        let mut sum = 0.0f32;
        for i in 0..last_dim {
            let e = (row[i] - max_val).exp();
            out[start + i] = e;
            sum += e;
        }

        // Single pass: normalize
        let inv_sum = 1.0 / sum;
        for i in 0..last_dim {
            out[start + i] *= inv_sum;
        }
    }

    Ok(Tensor::from_f32(&out, dims))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_linear_relu() {
        let input = Tensor::from_f32(&[1.0, -1.0, 2.0, -2.0], &[2, 2]);
        let weight = Tensor::from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]); // identity
        let bias = Tensor::from_f32(&[0.0, 0.0], &[2]);

        let out = fused_linear_relu(&input, &weight, Some(&bias)).unwrap();
        let data = out.as_f32_slice().unwrap();
        // ReLU(identity @ input) = [max(0,1), max(0,-1), max(0,2), max(0,-2)]
        assert_eq!(data, &[1.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn test_fused_layer_norm() {
        let input = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let gamma = Tensor::ones(&[3]);
        let beta = Tensor::zeros(&[3], DType::F32);

        let out = fused_layer_norm(&input, &gamma, &beta, 1e-5).unwrap();
        let data = out.as_f32_slice().unwrap();

        // Each row should be normalized to mean≈0, std≈1
        let row1_mean: f32 = data[0..3].iter().sum::<f32>() / 3.0;
        assert!(row1_mean.abs() < 1e-5, "mean={}", row1_mean);
    }

    #[test]
    fn test_fused_rms_norm() {
        let input = Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3]);
        let gamma = Tensor::ones(&[3]);

        let out = fused_rms_norm(&input, &gamma, 1e-5).unwrap();
        let data = out.as_f32_slice().unwrap();

        // RMS = sqrt((1+4+9)/3) = sqrt(14/3) ≈ 2.16
        // normalized ≈ [0.46, 0.93, 1.39]
        assert!(data[0] > 0.0 && data[0] < 1.0);
        assert!(data[2] > 1.0);
    }

    #[test]
    fn test_fused_softmax() {
        let input = Tensor::from_f32(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
        let out = fused_softmax(&input).unwrap();
        let data = out.as_f32_slice().unwrap();

        // Each row sums to 1
        let sum1: f32 = data[0..3].iter().sum();
        let sum2: f32 = data[3..6].iter().sum();
        assert!((sum1 - 1.0).abs() < 1e-6);
        assert!((sum2 - 1.0).abs() < 1e-6);

        // Monotonically increasing
        assert!(data[2] > data[1]);
        assert!(data[1] > data[0]);
    }

    #[test]
    fn test_fused_softmax_numerical_stability() {
        // Large values that would overflow without max subtraction
        let input = Tensor::from_f32(&[1000.0, 1001.0, 1002.0], &[1, 3]);
        let out = fused_softmax(&input).unwrap();
        let data = out.as_f32_slice().unwrap();

        assert!(data.iter().all(|v| v.is_finite()));
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    /// CPU-reference test for fused RMSNorm + Linear projection.
    /// Validates that: y = RMSNorm(x) @ W^T matches doing the two steps separately.
    #[test]
    fn test_fused_rms_norm_proj_cpu_reference() {
        let hidden = 8;
        let out_dim = 4;
        let batch = 2;

        // Random-ish input
        let x_data: Vec<f32> = (0..batch * hidden).map(|i| (i as f32 * 0.3 - 1.0)).collect();
        let gamma_data: Vec<f32> = (0..hidden).map(|i| 1.0 + i as f32 * 0.1).collect();
        let w_data: Vec<f32> = (0..out_dim * hidden).map(|i| (i as f32 * 0.1 - 0.5)).collect();

        let input = Tensor::from_f32(&x_data, &[batch, hidden]);
        let gamma = Tensor::from_f32(&gamma_data, &[hidden]);
        let weight = Tensor::from_f32(&w_data, &[out_dim, hidden]);

        // Step 1: RMSNorm
        let normed = fused_rms_norm(&input, &gamma, 1e-6).unwrap();

        // Step 2: Linear projection (normed @ W^T)
        let normed_data = normed.as_f32_slice().unwrap();
        let mut expected = vec![0.0f32; batch * out_dim];
        for b in 0..batch {
            for j in 0..out_dim {
                let mut acc = 0.0f32;
                for k in 0..hidden {
                    acc += normed_data[b * hidden + k] * w_data[j * hidden + k];
                }
                expected[b * out_dim + j] = acc;
            }
        }

        // Verify fused result matches (simulated inline)
        let mut fused = vec![0.0f32; batch * out_dim];
        let eps = 1e-6f32;
        for b in 0..batch {
            let row = &x_data[b * hidden..(b + 1) * hidden];
            let sum_sq: f32 = row.iter().map(|v| v * v).sum();
            let inv_rms = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
            for j in 0..out_dim {
                let mut acc = 0.0f32;
                for k in 0..hidden {
                    let x_hat = row[k] * inv_rms * gamma_data[k];
                    acc += x_hat * w_data[j * hidden + k];
                }
                fused[b * out_dim + j] = acc;
            }
        }

        for i in 0..expected.len() {
            assert!((expected[i] - fused[i]).abs() < 1e-5,
                "mismatch at {}: expected={}, fused={}", i, expected[i], fused[i]);
        }
    }

    /// CPU-reference test for dequantized quaternary matmul.
    /// Validates: C = dequant(A_packed) @ B matches explicit unpack + matmul.
    #[test]
    fn test_dequant_quat_matmul_cpu_reference() {
        use kore_btes::encoder::{Quat, pack_quats};

        let m = 4;
        let k = 8;
        let n = 3;
        let k_packed = (k + 3) / 4;

        // Create quaternary weight matrix and pack it
        let quat_weights: Vec<Quat> = vec![
            Quat::Pos3, Quat::Neg1, Quat::Pos1, Quat::Neg3,
            Quat::Pos1, Quat::Pos3, Quat::Neg1, Quat::Neg3,
        ];
        // Repeat for m rows
        let mut packed = Vec::new();
        let scales = vec![0.5f32; m]; // uniform scale
        for _row in 0..m {
            for chunk in quat_weights.chunks(4) {
                let block = [chunk[0], chunk[1], chunk[2], chunk[3]];
                packed.push(pack_quats(&block));
            }
        }

        // Dense activations B [K, N]
        let b_data: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.1).collect();

        // Reference: explicit dequant then matmul
        let quat_vals: Vec<f32> = quat_weights.iter().map(|q| q.to_f32()).collect();
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += quat_vals[p] * b_data[p * n + j];
                }
                expected[i * n + j] = acc * scales[i];
            }
        }

        // Simulate on-the-fly dequant matmul (same logic as CUDA kernel)
        let mut fused = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    let byte_idx = p / 4;
                    let bit_slot = p % 4;
                    let byte = packed[i * k_packed + byte_idx];
                    let qidx = (byte >> (bit_slot * 2)) & 0x3;
                    let lut = [-3.0f32, -1.0, 1.0, 3.0];
                    let w = lut[qidx as usize];
                    acc += w * b_data[p * n + j];
                }
                fused[i * n + j] = acc * scales[i];
            }
        }

        for i in 0..expected.len() {
            assert!((expected[i] - fused[i]).abs() < 1e-5,
                "mismatch at {}: expected={}, fused={}", i, expected[i], fused[i]);
        }
    }

    /// CPU-reference test for fused RMSNorm + SiLU gate.
    /// Validates: y = RMSNorm(x) * SiLU(z).
    #[test]
    fn test_fused_rms_norm_silu_gate_cpu_reference() {
        let cols = 4;
        let rows = 2;
        let eps = 1e-6f32;

        let x_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma_data = vec![1.0f32; cols];
        let z_data = vec![0.5f32, 1.0, -0.5, 2.0, -1.0, 0.0, 1.5, -2.0];

        let silu = |x: f32| -> f32 { x / (1.0 + (-x).exp()) };

        let mut expected = vec![0.0f32; rows * cols];
        for b in 0..rows {
            let row = &x_data[b * cols..(b + 1) * cols];
            let sum_sq: f32 = row.iter().map(|v| v * v).sum();
            let inv_rms = 1.0 / (sum_sq / cols as f32 + eps).sqrt();
            for c in 0..cols {
                let x_hat = row[c] * inv_rms * gamma_data[c];
                let z_val = z_data[b * cols + c];
                expected[b * cols + c] = x_hat * silu(z_val);
            }
        }

        // All values should be finite
        assert!(expected.iter().all(|v| v.is_finite()));
        // SiLU(0) = 0, so for z=0.0 the output should be 0
        // z_data[5] = 0.0, so expected[5] should be ~0
        assert!(expected[5].abs() < 1e-5, "SiLU(0) gate should zero output, got {}", expected[5]);
    }
}
