//! Gradient clipping utilities.
//!
//! Provides `clip_grad_norm_` and `clip_grad_value_` for preventing
//! gradient explosion during training.

use kore_core::{Tensor, KoreError};

/// Clip gradients by global L2 norm.
///
/// Scales all gradients so that the total L2 norm does not exceed `max_norm`.
/// Returns the original total norm before clipping.
///
/// # Arguments
/// * `grads` - Mutable slice of gradient tensors (must all be F32)
/// * `max_norm` - Maximum allowed L2 norm
///
/// # Errors
/// Returns `KoreError::UnsupportedDType` if any gradient tensor is not F32.
pub fn clip_grad_norm_(grads: &mut [Tensor], max_norm: f32) -> kore_core::Result<f32> {
    // Force contiguous and compute total L2 norm across all gradients
    let mut total_norm_sq = 0.0f32;
    for g in grads.iter_mut() {
        let c = g.contiguous();
        let data = c.as_f32_slice()
            .ok_or_else(|| KoreError::UnsupportedDType(g.dtype()))?;
        total_norm_sq += data.iter().map(|v| v * v).sum::<f32>();
    }
    let total_norm = total_norm_sq.sqrt();

    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        for g in grads.iter_mut() {
            let c = g.contiguous();
            let data = c.as_f32_slice()
                .ok_or_else(|| KoreError::UnsupportedDType(g.dtype()))?;
            let scaled: Vec<f32> = data.iter().map(|v| v * scale).collect();
            let shape = g.shape().dims().to_vec();
            *g = Tensor::from_f32(&scaled, &shape);
        }
    }

    Ok(total_norm)
}

/// Clip gradients element-wise to [-clip_value, clip_value].
///
/// # Arguments
/// * `grads` - Mutable slice of gradient tensors (must all be F32)
/// * `clip_value` - Maximum absolute value for each gradient element
///
/// # Errors
/// Returns `KoreError::UnsupportedDType` if any gradient tensor is not F32.
pub fn clip_grad_value_(grads: &mut [Tensor], clip_value: f32) -> kore_core::Result<()> {
    for g in grads.iter_mut() {
        let c = g.contiguous();
        let data = c.as_f32_slice()
            .ok_or_else(|| KoreError::UnsupportedDType(g.dtype()))?;
        let clipped: Vec<f32> = data.iter().map(|v| v.clamp(-clip_value, clip_value)).collect();
        let shape = g.shape().dims().to_vec();
        *g = Tensor::from_f32(&clipped, &shape);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_grad_norm_no_clip() {
        // Small gradients — should not be clipped
        let mut grads = vec![
            Tensor::from_f32(&[0.1, 0.2], &[2]),
        ];
        let norm = clip_grad_norm_(&mut grads, 10.0).unwrap();
        assert!(norm < 10.0);
        let data = grads[0].as_f32_slice().unwrap();
        assert!((data[0] - 0.1).abs() < 1e-6);
        assert!((data[1] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_clip_grad_norm_clips() {
        // Large gradients — should be scaled down
        let mut grads = vec![
            Tensor::from_f32(&[3.0, 4.0], &[2]), // norm = 5.0
        ];
        let norm = clip_grad_norm_(&mut grads, 1.0).unwrap();
        assert!((norm - 5.0).abs() < 1e-5);

        // After clipping, norm should be ~1.0
        let data = grads[0].as_f32_slice().unwrap();
        let new_norm = (data[0] * data[0] + data[1] * data[1]).sqrt();
        assert!((new_norm - 1.0).abs() < 1e-4, "new_norm={}", new_norm);
    }

    #[test]
    fn test_clip_grad_norm_multi_tensor() {
        let mut grads = vec![
            Tensor::from_f32(&[3.0], &[1]),
            Tensor::from_f32(&[4.0], &[1]),
        ];
        let norm = clip_grad_norm_(&mut grads, 2.5).unwrap();
        assert!((norm - 5.0).abs() < 1e-5);

        let d0 = grads[0].as_f32_slice().unwrap();
        let d1 = grads[1].as_f32_slice().unwrap();
        let new_norm = (d0[0] * d0[0] + d1[0] * d1[0]).sqrt();
        assert!((new_norm - 2.5).abs() < 1e-3, "new_norm={}", new_norm);
    }

    #[test]
    fn test_clip_grad_value() {
        let mut grads = vec![
            Tensor::from_f32(&[-5.0, 0.5, 3.0, -0.1], &[4]),
        ];
        clip_grad_value_(&mut grads, 1.0).unwrap();
        let data = grads[0].as_f32_slice().unwrap();
        assert_eq!(data, &[-1.0, 0.5, 1.0, -0.1]);
    }
}
