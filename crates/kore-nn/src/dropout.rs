//! Dropout layer — randomly zeroes elements during training.

use kore_core::{KoreError, Tensor};
use crate::module::Module;

/// Dropout layer: randomly zeroes elements with probability `p` during training.
/// During evaluation, acts as identity.
pub struct Dropout {
    p: f32,
    training: bool,
    seed: u64,
}

impl Dropout {
    /// Create a new Dropout layer with the given drop probability.
    pub fn new(p: f32) -> Self {
        assert!((0.0..1.0).contains(&p), "Dropout probability must be in [0, 1)");
        Self {
            p,
            training: true,
            seed: 42,
        }
    }

    /// Set the random seed.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    /// Drop probability.
    pub fn p(&self) -> f32 {
        self.p
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> kore_core::Result<Tensor> {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        let data = input.contiguous();
        let slice = data.as_f32_slice().ok_or_else(|| {
            KoreError::UnsupportedDType(input.dtype())
        })?;
        let scale = 1.0 / (1.0 - self.p);

        // Use a local copy of seed for deterministic but varying masks
        let mut seed = self.seed;
        let mut next = || -> f32 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed as f32) / (u64::MAX as f32)
        };

        let result: Vec<f32> = slice
            .iter()
            .map(|&x| {
                if next() < self.p {
                    0.0
                } else {
                    x * scale
                }
            })
            .collect();

        Ok(Tensor::from_f32(&result, input.shape().dims()))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![] // No learnable parameters
    }

    fn named_parameters(&self) -> Vec<(&str, &Tensor)> {
        vec![]
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout_eval_identity() {
        let mut dropout = Dropout::new(0.5);
        dropout.train(false);

        let input = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let output = dropout.forward(&input).unwrap();
        assert_eq!(
            input.as_f32_slice().unwrap(),
            output.as_f32_slice().unwrap()
        );
    }

    #[test]
    fn test_dropout_zero_prob() {
        let dropout = Dropout::new(0.0);
        let input = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        let output = dropout.forward(&input).unwrap();
        assert_eq!(
            input.as_f32_slice().unwrap(),
            output.as_f32_slice().unwrap()
        );
    }

    #[test]
    fn test_dropout_training_has_zeros() {
        let dropout = Dropout::new(0.5);
        let input = Tensor::from_f32(&vec![1.0; 1000], &[1000]);
        let output = dropout.forward(&input).unwrap();
        let data = output.as_f32_slice().unwrap();

        let zero_count = data.iter().filter(|&&v| v == 0.0).count();
        // With p=0.5, roughly half should be zero (allow wide margin)
        assert!(zero_count > 200, "too few zeros: {}", zero_count);
        assert!(zero_count < 800, "too many zeros: {}", zero_count);
    }

    #[test]
    fn test_dropout_scaling() {
        let dropout = Dropout::new(0.5);
        let input = Tensor::from_f32(&vec![1.0; 1000], &[1000]);
        let output = dropout.forward(&input).unwrap();
        let data = output.as_f32_slice().unwrap();

        // Non-zero values should be scaled by 1/(1-p) = 2.0
        for &v in data {
            if v != 0.0 {
                assert!((v - 2.0).abs() < 1e-5, "expected 2.0, got {}", v);
            }
        }
    }

    #[test]
    fn test_dropout_no_parameters() {
        let dropout = Dropout::new(0.3);
        assert_eq!(dropout.parameters().len(), 0);
    }
}
