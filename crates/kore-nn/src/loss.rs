//! Loss functions for training: cross-entropy, MSE, L1, NLL.

use kore_core::{KoreError, Tensor, DType};

/// Cross-entropy loss: -sum(target * log_softmax(logits)) / batch_size.
///
/// `logits`: [batch, num_classes] — raw unnormalized scores
/// `targets`: [batch] — integer class indices (stored as f32)
///
/// Returns scalar loss tensor.
pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor, KoreError> {
    if logits.dtype() != DType::F32 || targets.dtype() != DType::F32 {
        return Err(KoreError::UnsupportedDType(logits.dtype()));
    }

    let logit_dims = logits.shape().dims();
    let target_dims = targets.shape().dims();

    if logit_dims.len() != 2 || target_dims.len() != 1 {
        return Err(KoreError::StorageError(
            format!("cross_entropy: expected logits [B, C] and targets [B], got {:?} and {:?}",
                logit_dims, target_dims)
        ));
    }

    let batch = logit_dims[0];
    let num_classes = logit_dims[1];

    if target_dims[0] != batch {
        return Err(KoreError::ShapeMismatch {
            expected: vec![batch],
            got: target_dims.to_vec(),
        });
    }

    let log_probs = logits.log_softmax(-1)?;
    let lp_data = log_probs.as_f32_slice().unwrap();
    let t_data = targets.contiguous();
    let t_slice = t_data.as_f32_slice().unwrap();

    let mut total_loss = 0.0f32;
    for b in 0..batch {
        let class_idx = t_slice[b] as usize;
        if class_idx >= num_classes {
            return Err(KoreError::StorageError(
                format!("cross_entropy: target {} out of range for {} classes", class_idx, num_classes)
            ));
        }
        total_loss -= lp_data[b * num_classes + class_idx];
    }

    Ok(Tensor::scalar(total_loss / batch as f32))
}

/// Cross-entropy with label smoothing.
///
/// `smoothing`: fraction of probability mass to redistribute uniformly.
/// When smoothing=0.0, equivalent to standard cross_entropy_loss.
pub fn cross_entropy_loss_smoothed(
    logits: &Tensor,
    targets: &Tensor,
    smoothing: f32,
) -> Result<Tensor, KoreError> {
    if smoothing <= 0.0 {
        return cross_entropy_loss(logits, targets);
    }

    let logit_dims = logits.shape().dims();
    let batch = logit_dims[0];
    let num_classes = logit_dims[1];

    let log_probs = logits.log_softmax(-1)?;
    let lp_data = log_probs.as_f32_slice().unwrap();
    let t_data = targets.contiguous();
    let t_slice = t_data.as_f32_slice().unwrap();

    let confidence = 1.0 - smoothing;
    let smooth_val = smoothing / num_classes as f32;

    let mut total_loss = 0.0f32;
    for b in 0..batch {
        let class_idx = t_slice[b] as usize;
        // NLL component: -confidence * log_prob[target]
        total_loss -= confidence * lp_data[b * num_classes + class_idx];
        // Smoothing component: -smooth_val * sum(log_probs)
        let row_sum: f32 = lp_data[b * num_classes..(b + 1) * num_classes].iter().sum();
        total_loss -= smooth_val * row_sum;
    }

    Ok(Tensor::scalar(total_loss / batch as f32))
}

/// Mean Squared Error loss: mean((pred - target)^2).
///
/// Both tensors must have the same shape.
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor, KoreError> {
    let diff = pred.sub(target)?;
    let sq = diff.mul(&diff)?;
    sq.mean()
}

/// L1 loss (Mean Absolute Error): mean(|pred - target|).
pub fn l1_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor, KoreError> {
    let diff = pred.sub(target)?;
    let abs_diff = diff.abs()?;
    abs_diff.mean()
}

/// Negative Log-Likelihood loss.
///
/// `log_probs`: [batch, num_classes] — log probabilities (output of log_softmax)
/// `targets`: [batch] — integer class indices (stored as f32)
pub fn nll_loss(log_probs: &Tensor, targets: &Tensor) -> Result<Tensor, KoreError> {
    let dims = log_probs.shape().dims();
    let batch = dims[0];
    let num_classes = dims[1];

    let lp_data = log_probs.contiguous();
    let lp_slice = lp_data.as_f32_slice().ok_or(KoreError::UnsupportedDType(log_probs.dtype()))?;
    let t_data = targets.contiguous();
    let t_slice = t_data.as_f32_slice().ok_or(KoreError::UnsupportedDType(targets.dtype()))?;

    let mut total = 0.0f32;
    for b in 0..batch {
        let idx = t_slice[b] as usize;
        if idx >= num_classes {
            return Err(KoreError::StorageError(
                format!("nll_loss: target {} out of range", idx)
            ));
        }
        total -= lp_slice[b * num_classes + idx];
    }

    Ok(Tensor::scalar(total / batch as f32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy_basic() {
        // Perfect prediction for class 1 should give low loss
        let logits = Tensor::from_f32(&[
            -10.0, 10.0, -10.0,  // batch 0: class 1 is dominant
            10.0, -10.0, -10.0,  // batch 1: class 0 is dominant
        ], &[2, 3]);
        let targets = Tensor::from_f32(&[1.0, 0.0], &[2]);
        let loss = cross_entropy_loss(&logits, &targets).unwrap();
        let val = loss.get_f32(0).unwrap();
        assert!(val < 0.01, "loss should be near 0 for correct predictions, got {}", val);
    }

    #[test]
    fn test_cross_entropy_wrong_prediction() {
        // Wrong predictions should give high loss
        let logits = Tensor::from_f32(&[
            10.0, -10.0, -10.0,  // predicts class 0
        ], &[1, 3]);
        let targets = Tensor::from_f32(&[2.0], &[1]); // actual class 2
        let loss = cross_entropy_loss(&logits, &targets).unwrap();
        let val = loss.get_f32(0).unwrap();
        assert!(val > 5.0, "loss should be high for wrong prediction, got {}", val);
    }

    #[test]
    fn test_cross_entropy_smoothed() {
        let logits = Tensor::from_f32(&[-10.0, 10.0, -10.0], &[1, 3]);
        let targets = Tensor::from_f32(&[1.0], &[1]);

        let loss_no_smooth = cross_entropy_loss(&logits, &targets).unwrap().get_f32(0).unwrap();
        let loss_smooth = cross_entropy_loss_smoothed(&logits, &targets, 0.1).unwrap().get_f32(0).unwrap();

        // Smoothed loss should be slightly higher than unsmoothed for correct predictions
        assert!(loss_smooth > loss_no_smooth);
    }

    #[test]
    fn test_mse_loss() {
        let pred = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        let target = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        let loss = mse_loss(&pred, &target).unwrap();
        assert!((loss.get_f32(0).unwrap()).abs() < 1e-6);
    }

    #[test]
    fn test_mse_loss_nonzero() {
        let pred = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        let target = Tensor::from_f32(&[2.0, 3.0, 4.0], &[3]);
        let loss = mse_loss(&pred, &target).unwrap();
        // MSE = mean((1)^2 + (1)^2 + (1)^2) = 1.0
        assert!((loss.get_f32(0).unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_l1_loss() {
        let pred = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        let target = Tensor::from_f32(&[2.0, 4.0, 0.0], &[3]);
        // L1 = mean(|1-2| + |2-4| + |3-0|) = mean(1 + 2 + 3) = 2.0
        let loss = l1_loss(&pred, &target).unwrap();
        assert!((loss.get_f32(0).unwrap() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_nll_loss() {
        let log_probs = Tensor::from_f32(&[
            -2.0, -0.1, -3.0,  // batch 0
            -0.5, -1.0, -2.0,  // batch 1
        ], &[2, 3]);
        let targets = Tensor::from_f32(&[1.0, 0.0], &[2]);
        let loss = nll_loss(&log_probs, &targets).unwrap();
        // loss = -(-0.1 + -0.5) / 2 = 0.3
        let val = loss.get_f32(0).unwrap();
        assert!((val - 0.3).abs() < 1e-5, "got {}", val);
    }
}
