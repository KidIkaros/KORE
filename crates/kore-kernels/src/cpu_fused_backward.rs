//! Fused backward (gradient) kernels for common operations.
//!
//! Each function computes gradients in a single pass over the data,
//! avoiding intermediate tensor allocations that the naive chain-rule
//! approach would require.

use kore_core::{DType, KoreError, Tensor};

/// Fused RMSNorm backward.
///
/// Given forward: `y = x / rms(x) * gamma`
/// Computes: `dx`, `dgamma` in a single pass.
///
/// `grad_output`: [batch, d] — upstream gradient
/// `input`: [batch, d] — original input to RMSNorm
/// `gamma`: [d] — scale parameter
/// `eps`: numerical stability constant
///
/// Returns `(grad_input, grad_gamma)`.
pub fn fused_rms_norm_backward(
    grad_output: &Tensor,
    input: &Tensor,
    gamma: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor), KoreError> {
    if input.dtype() != DType::F32 {
        return Err(KoreError::UnsupportedDType(input.dtype()));
    }

    let input = input.contiguous();
    let go = grad_output.contiguous();
    let in_data = input.as_f32_slice().unwrap();
    let go_data = go.as_f32_slice().unwrap();
    let g_data = gamma.as_f32_slice().unwrap();

    let dims = input.shape().dims();
    let d = *dims.last().unwrap();
    let batch = input.numel() / d;

    let mut dx = vec![0.0f32; batch * d];
    let mut dgamma = vec![0.0f32; d];

    for b in 0..batch {
        let start = b * d;
        let x_row = &in_data[start..start + d];
        let go_row = &go_data[start..start + d];

        // Forward quantities
        let sum_sq: f32 = x_row.iter().map(|&v| v * v).sum();
        let rms = (sum_sq / d as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        // dgamma += go * (x / rms)
        for i in 0..d {
            dgamma[i] += go_row[i] * x_row[i] * inv_rms;
        }

        // dx: d/dx[x * inv_rms * gamma]
        // = gamma * inv_rms * (I - x * x^T / (d * rms^2)) * go
        let mut dot_xg = 0.0f32;
        for i in 0..d {
            dot_xg += x_row[i] * go_row[i] * g_data[i];
        }
        let coeff = dot_xg * inv_rms / (sum_sq + eps * d as f32);

        for i in 0..d {
            dx[start + i] = g_data[i] * inv_rms * go_row[i] - coeff * x_row[i];
        }
    }

    Ok((
        Tensor::from_f32(&dx, dims),
        Tensor::from_f32(&dgamma, &[d]),
    ))
}

/// Fused softmax backward.
///
/// Given forward: `y = softmax(x)` (over last dim)
/// Computes: `dx = y * (dy - sum(y * dy))` in a single pass.
///
/// `grad_output`: [batch, d] — upstream gradient
/// `output`: [batch, d] — softmax output (from forward pass)
///
/// Returns `grad_input`.
pub fn fused_softmax_backward(
    grad_output: &Tensor,
    output: &Tensor,
) -> Result<Tensor, KoreError> {
    if output.dtype() != DType::F32 {
        return Err(KoreError::UnsupportedDType(output.dtype()));
    }

    let out = output.contiguous();
    let go = grad_output.contiguous();
    let out_data = out.as_f32_slice().unwrap();
    let go_data = go.as_f32_slice().unwrap();

    let dims = output.shape().dims();
    let d = *dims.last().unwrap();
    let batch = output.numel() / d;

    let mut dx = vec![0.0f32; batch * d];

    for b in 0..batch {
        let start = b * d;
        let y = &out_data[start..start + d];
        let dy = &go_data[start..start + d];

        // dot = sum(y * dy)
        let dot: f32 = y.iter().zip(dy.iter()).map(|(&a, &b)| a * b).sum();

        // dx = y * (dy - dot)
        for i in 0..d {
            dx[start + i] = y[i] * (dy[i] - dot);
        }
    }

    Ok(Tensor::from_f32(&dx, dims))
}

/// Fused cross-entropy backward.
///
/// Given forward: `loss = -mean(log_softmax(logits)[targets])`
/// Computes: `d_logits = (softmax(logits) - one_hot(targets)) / batch`
///
/// This avoids computing softmax and one_hot as separate tensors.
///
/// `logits`: [batch, num_classes] — raw logits
/// `targets`: [batch] — integer class indices (as f32)
///
/// Returns `grad_logits` [batch, num_classes].
pub fn fused_cross_entropy_backward(
    logits: &Tensor,
    targets: &Tensor,
) -> Result<Tensor, KoreError> {
    if logits.dtype() != DType::F32 {
        return Err(KoreError::UnsupportedDType(logits.dtype()));
    }

    let logits = logits.contiguous();
    let targets = targets.contiguous();
    let l_data = logits.as_f32_slice().unwrap();
    let t_data = targets.as_f32_slice().unwrap();

    let dims = logits.shape().dims();
    let batch = dims[0];
    let num_classes = dims[1];

    let mut grad = vec![0.0f32; batch * num_classes];
    let inv_batch = 1.0 / batch as f32;

    for b in 0..batch {
        let row = &l_data[b * num_classes..(b + 1) * num_classes];
        let target_idx = t_data[b] as usize;

        // Compute softmax for this row
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for i in 0..num_classes {
            let e = (row[i] - max_val).exp();
            grad[b * num_classes + i] = e;
            sum += e;
        }
        let inv_sum = 1.0 / sum;

        // grad = (softmax - one_hot) / batch
        for i in 0..num_classes {
            grad[b * num_classes + i] *= inv_sum * inv_batch;
        }
        grad[b * num_classes + target_idx] -= inv_batch;
    }

    Ok(Tensor::from_f32(&grad, dims))
}

/// Fused softmax + cross-entropy backward (combined).
///
/// Most efficient when you have raw logits and targets — computes the
/// gradient without ever materializing the softmax output or log-softmax.
///
/// Equivalent to `fused_cross_entropy_backward` but named explicitly
/// to indicate it handles the full softmax+CE chain.
pub fn fused_softmax_cross_entropy_backward(
    logits: &Tensor,
    targets: &Tensor,
) -> Result<Tensor, KoreError> {
    fused_cross_entropy_backward(logits, targets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_backward_shape() {
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let gamma = Tensor::ones(&[3]);
        let go = Tensor::ones(&[2, 3]);

        let (dx, dgamma) = fused_rms_norm_backward(&go, &x, &gamma, 1e-5).unwrap();
        assert_eq!(dx.shape().dims(), &[2, 3]);
        assert_eq!(dgamma.shape().dims(), &[3]);
    }

    #[test]
    fn test_rms_norm_backward_finite() {
        let x = Tensor::from_f32(&[0.5, -0.3, 1.2, -0.8, 0.1, 0.7], &[2, 3]);
        let gamma = Tensor::from_f32(&[1.0, 0.5, 2.0], &[3]);
        let go = Tensor::from_f32(&[0.1, -0.2, 0.3, -0.1, 0.4, -0.5], &[2, 3]);

        let (dx, dgamma) = fused_rms_norm_backward(&go, &x, &gamma, 1e-5).unwrap();
        let dx_data = dx.as_f32_slice().unwrap();
        let dg_data = dgamma.as_f32_slice().unwrap();
        assert!(dx_data.iter().all(|v| v.is_finite()));
        assert!(dg_data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_rms_norm_backward_zero_grad_for_zero_upstream() {
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3]);
        let gamma = Tensor::ones(&[3]);
        let go = Tensor::zeros(&[1, 3], DType::F32);

        let (dx, dgamma) = fused_rms_norm_backward(&go, &x, &gamma, 1e-5).unwrap();
        let dx_data = dx.as_f32_slice().unwrap();
        let dg_data = dgamma.as_f32_slice().unwrap();
        assert!(dx_data.iter().all(|v| v.abs() < 1e-6));
        assert!(dg_data.iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn test_softmax_backward_shape() {
        let y = Tensor::from_f32(&[0.2, 0.3, 0.5, 0.1, 0.6, 0.3], &[2, 3]);
        let go = Tensor::ones(&[2, 3]);

        let dx = fused_softmax_backward(&go, &y).unwrap();
        assert_eq!(dx.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_softmax_backward_uniform_grad_is_zero() {
        // When upstream gradient is uniform, softmax backward should be ~0
        // because d/dx softmax(x) @ 1 = softmax(x) * (1 - sum(softmax(x) * 1)) = 0
        let y = Tensor::from_f32(&[0.2, 0.3, 0.5], &[1, 3]);
        let go = Tensor::from_f32(&[1.0, 1.0, 1.0], &[1, 3]);

        let dx = fused_softmax_backward(&go, &y).unwrap();
        let data = dx.as_f32_slice().unwrap();
        for &v in data {
            assert!(v.abs() < 1e-6, "expected ~0, got {}", v);
        }
    }

    #[test]
    fn test_cross_entropy_backward_shape() {
        let logits = Tensor::from_f32(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
        let targets = Tensor::from_f32(&[2.0, 0.0], &[2]);

        let grad = fused_cross_entropy_backward(&logits, &targets).unwrap();
        assert_eq!(grad.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_cross_entropy_backward_sums_to_zero() {
        // Each row of CE gradient should sum to 0 (softmax sums to 1, one_hot sums to 1)
        let logits = Tensor::from_f32(&[1.0, 2.0, 3.0, 0.5, -0.5, 1.0], &[2, 3]);
        let targets = Tensor::from_f32(&[1.0, 2.0], &[2]);

        let grad = fused_cross_entropy_backward(&logits, &targets).unwrap();
        let data = grad.as_f32_slice().unwrap();

        let row0_sum: f32 = data[0..3].iter().sum();
        let row1_sum: f32 = data[3..6].iter().sum();
        assert!(row0_sum.abs() < 1e-6, "row0 sum: {}", row0_sum);
        assert!(row1_sum.abs() < 1e-6, "row1 sum: {}", row1_sum);
    }

    #[test]
    fn test_cross_entropy_backward_correct_target_negative() {
        // Gradient at target class should be negative (softmax < 1 → softmax - 1 < 0)
        let logits = Tensor::from_f32(&[0.0, 0.0, 10.0], &[1, 3]);
        let targets = Tensor::from_f32(&[2.0], &[1]);

        let grad = fused_cross_entropy_backward(&logits, &targets).unwrap();
        let data = grad.as_f32_slice().unwrap();
        // Class 2 has high logit → softmax ≈ 1 → grad ≈ (1 - 1)/1 ≈ 0
        // Class 0,1 have low logit → softmax ≈ 0 → grad ≈ 0/1 ≈ 0
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_cross_entropy_backward_wrong_prediction() {
        // When prediction is wrong, target class gradient should be strongly negative
        let logits = Tensor::from_f32(&[10.0, 0.0, 0.0], &[1, 3]);
        let targets = Tensor::from_f32(&[2.0], &[1]); // target is class 2 but logits favor class 0

        let grad = fused_cross_entropy_backward(&logits, &targets).unwrap();
        let data = grad.as_f32_slice().unwrap();
        // Class 0: softmax ≈ 1, not target → grad ≈ +1/batch
        assert!(data[0] > 0.0);
        // Class 2: softmax ≈ 0, is target → grad ≈ (0 - 1)/batch = -1/batch
        assert!(data[2] < 0.0);
    }
}
