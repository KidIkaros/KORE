//! Trainer — high-level training loop for supervised learning.
//!
//! Provides a PyTorch/Keras-style `fit` / `evaluate` / `predict` API
//! that orchestrates forward pass, loss computation, backward pass,
//! and optimizer step.
//!
//! # Example (Rust)
//! ```ignore
//! use kore_nn::{Sequential, Linear, Trainer, TrainerConfig};
//! use kore_nn::loss::mse_loss;
//! use kore_optim::Adam;
//! use kore_data::{TensorDataset, DataLoader};
//! use kore_core::Tensor;
//!
//! let model = Sequential::new(vec![
//!     Box::new(Linear::new(16, 64, true)),
//!     Box::new(Linear::new(64, 1, true)),
//! ]);
//!
//! let x = Tensor::randn(&[100, 16]);
//! let y = Tensor::randn(&[100, 1]);
//! let ds = TensorDataset::new(&x, &y);
//! let train_loader = DataLoader::new(Box::new(ds), 32, true, true, Some(42));
//!
//! let optimizer = Adam::default_with_lr(0.001);
//! let config = TrainerConfig::default();
//! let mut trainer = Trainer::new(model, optimizer, mse_loss_wrapper, config);
//!
//! let history = trainer.fit(&train_loader, 10);
//! for (epoch, loss) in history.iter().enumerate() {
//!     println!("epoch {}: loss={:.4}", epoch, loss);
//! }
//! ```

use kore_core::Tensor;
use kore_data::DataLoader;
use kore_optim::Optimizer;

use crate::module::Module;

/// Loss function signature: `(prediction, target) -> scalar loss Tensor`.
pub type LossFn = fn(&Tensor, &Tensor) -> kore_core::Result<Tensor>;

/// Configuration for the Trainer.
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// Print loss every N epochs (0 = silent).
    pub log_every: usize,
    /// Gradient clipping max norm (0.0 = no clipping).
    pub grad_clip_norm: f32,
    /// Enable diagnostic messages (quantized-param warnings, etc.).
    /// Set to `false` for production or notebook use where stderr output
    /// is undesirable. Default: `true`.
    pub verbose: bool,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            log_every: 1,
            grad_clip_norm: 0.0,
            verbose: true,
        }
    }
}

/// Epoch metrics returned by a single epoch of training/evaluation.
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    /// Average loss over all batches.
    pub avg_loss: f32,
    /// Number of batches processed.
    pub num_batches: usize,
    /// Number of samples processed.
    pub num_samples: usize,
}

/// Training history: per-epoch metrics.
#[derive(Debug, Clone)]
pub struct TrainHistory {
    pub epochs: Vec<EpochMetrics>,
}

impl TrainHistory {
    fn new() -> Self {
        Self { epochs: Vec::new() }
    }

    /// Iterate over average loss per epoch.
    pub fn losses(&self) -> Vec<f32> {
        self.epochs.iter().map(|e| e.avg_loss).collect()
    }
}

/// High-level trainer that wraps model + optimizer + loss function.
pub struct Trainer {
    model: Box<dyn Module>,
    optimizer: Box<dyn Optimizer>,
    loss_fn: LossFn,
    config: TrainerConfig,
}

impl Trainer {
    /// Create a new Trainer.
    pub fn new(
        model: impl Module + 'static,
        optimizer: impl Optimizer + 'static,
        loss_fn: LossFn,
        config: TrainerConfig,
    ) -> Self {
        Self {
            model: Box::new(model),
            optimizer: Box::new(optimizer),
            loss_fn,
            config,
        }
    }

    /// Run the training loop for `epochs` epochs.
    ///
    /// Returns a `TrainHistory` with per-epoch metrics.
    pub fn fit(&mut self, train_loader: &DataLoader, epochs: usize) -> TrainHistory {
        self.model.train(true);

        // Warn about non-trainable quantized weights (BitLinear/QuatLinear)
        if self.config.verbose {
            let num_trainable: usize = self.model.parameters()
                .iter()
                .map(|p| p.shape().dims().iter().product::<usize>())
                .sum();
            let num_quantized = self.model.num_quantized_params();
            if num_quantized > 0 {
                eprintln!(
                    "warning: model contains {} quantized (frozen) weight parameters \
                     that cannot be updated via backpropagation. \
                     Only {} differentiable parameters will be trained. \
                     To train quantized layer weights, first train a full-precision \
                     model then quantize with from_linear().",
                    num_quantized, num_trainable,
                );
            }
        }

        let mut history = TrainHistory::new();

        for epoch in 0..epochs {
            let metrics = self.train_one_epoch(train_loader);

            if self.config.log_every > 0 && (epoch + 1) % self.config.log_every == 0 {
                eprintln!(
                    "epoch {}/{} — loss: {:.6} ({} batches, {} samples)",
                    epoch + 1,
                    epochs,
                    metrics.avg_loss,
                    metrics.num_batches,
                    metrics.num_samples,
                );
            }

            history.epochs.push(metrics);
        }

        history
    }

    /// Run a single training epoch over the given DataLoader.
    fn train_one_epoch(&mut self, loader: &DataLoader) -> EpochMetrics {
        let mut total_loss = 0.0f64;
        let mut num_batches = 0usize;
        let mut num_samples = 0usize;

        for batch in loader.iter() {
            // Forward pass
            let prediction = match self.model.forward(&batch.input) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("  forward error: {e}");
                    continue;
                }
            };

            // Compute loss
            let loss = match (self.loss_fn)(&prediction, &batch.target) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("  loss error: {e}");
                    continue;
                }
            };

            // Extract scalar loss value; skip batch on NaN to avoid poisoning
            let loss_val = loss.get_f32(0).unwrap_or(f32::NAN);
            if loss_val.is_nan() {
                eprintln!("  warning: NaN loss at batch {}, skipping", num_batches);
                continue;
            }
            total_loss += loss_val as f64;

            // Zero gradients before backward to prevent accumulation
            for p in self.model.parameters() {
                p.zero_grad();
            }

            // Backward pass
            if let Err(e) = loss.backward() {
                eprintln!("  backward error: {e}");
                continue;
            }

            // Collect gradients (immutable borrow, then released)
            let grads: Vec<Tensor> = self.model.parameters()
                .iter()
                .map(|p| {
                    p.grad().unwrap_or_else(|| {
                        Tensor::zeros(p.shape().dims(), p.dtype())
                    })
                })
                .collect();

            // Get mutable refs to model parameters for in-place update
            // (no clone needed — optimizer mutates directly, avoiding
            // copy-on-write overhead from Arc-based Tensor storage)
            let mut params_mut = self.model.parameters_mut();

            // Optional gradient clipping (in-place)
            let mut grads = grads;
            if self.config.grad_clip_norm > 0.0 {
                let _ = kore_optim::clip_grad_norm_(&mut grads, self.config.grad_clip_norm);
            }
            self.optimizer.step(&mut params_mut, &grads);

            num_batches += 1;
            num_samples += batch.size;
        }

        let avg_loss = if num_batches > 0 {
            (total_loss / num_batches as f64) as f32
        } else {
            0.0
        };

        EpochMetrics { avg_loss, num_batches, num_samples }
    }

    /// Evaluate the model on a dataset (no gradient computation).
    ///
    /// Returns metrics with average loss over all batches.
    pub fn evaluate(&mut self, eval_loader: &DataLoader) -> EpochMetrics {
        let was_training = self.model.is_training();
        self.model.train(false);
        let mut total_loss = 0.0f64;
        let mut num_batches = 0usize;
        let mut num_samples = 0usize;

        let _no_grad = kore_core::autograd::NoGradGuard::new();

        for batch in eval_loader.iter() {
            let prediction = match self.model.forward(&batch.input) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("  eval forward error: {e}");
                    continue;
                }
            };

            let loss = match (self.loss_fn)(&prediction, &batch.target) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("  eval loss error: {e}");
                    continue;
                }
            };

            let loss_val = loss.get_f32(0).unwrap_or(f32::NAN);
            if loss_val.is_nan() {
                eprintln!("  warning: NaN eval loss at batch {}, skipping", num_batches);
                continue;
            }
            total_loss += loss_val as f64;
            num_batches += 1;
            num_samples += batch.size;
        }

        self.model.train(was_training);

        let avg_loss = if num_batches > 0 {
            (total_loss / num_batches as f64) as f32
        } else {
            0.0
        };

        EpochMetrics { avg_loss, num_batches, num_samples }
    }

    /// Run inference on all batches and collect predictions.
    pub fn predict(&mut self, loader: &DataLoader) -> Vec<Tensor> {
        let was_training = self.model.is_training();
        self.model.train(false);
        let _no_grad = kore_core::autograd::NoGradGuard::new();

        let mut predictions = Vec::new();
        for batch in loader.iter() {
            match self.model.forward(&batch.input) {
                Ok(p) => predictions.push(p),
                Err(e) => eprintln!("  predict error: {e}"),
            }
        }

        self.model.train(was_training);
        predictions
    }

    /// Get a reference to the underlying model.
    pub fn model(&self) -> &dyn Module {
        self.model.as_ref()
    }

    /// Get the current learning rate from the optimizer.
    pub fn lr(&self) -> f32 {
        self.optimizer.lr()
    }

    /// Set the optimizer learning rate.
    pub fn set_lr(&mut self, lr: f32) {
        self.optimizer.set_lr(lr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Sequential, Linear};
    use kore_data::TensorDataset;

    fn mse_loss(pred: &Tensor, target: &Tensor) -> kore_core::Result<Tensor> {
        crate::loss::mse_loss(pred, target)
    }

    #[test]
    fn test_trainer_creation() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(4, 2, true)),
        ]);
        let optimizer = kore_optim::SGD::new(0.01, 0.0, 0.0);
        let config = TrainerConfig::default();
        let trainer = Trainer::new(model, optimizer, mse_loss, config);
        assert!(trainer.lr() > 0.0);
    }

    #[test]
    fn test_trainer_fit_smoke() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(4, 2, true)),
        ]);
        let optimizer = kore_optim::SGD::new(0.01, 0.0, 0.0);
        let config = TrainerConfig { log_every: 0, grad_clip_norm: 0.0, verbose: false };
        let mut trainer = Trainer::new(model, optimizer, mse_loss, config);

        let x = Tensor::ones(&[8, 4]);
        let y = Tensor::zeros(&[8, 2], kore_core::DType::F32);
        let ds = TensorDataset::new(&x, &y);
        let loader = DataLoader::new(Box::new(ds), 4, false, false, None);

        let history = trainer.fit(&loader, 3);
        assert_eq!(history.epochs.len(), 3);
        for m in &history.epochs {
            assert_eq!(m.num_batches, 2);
            assert_eq!(m.num_samples, 8);
        }
    }

    #[test]
    fn test_trainer_evaluate_smoke() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(4, 2, true)),
        ]);
        let optimizer = kore_optim::SGD::new(0.01, 0.0, 0.0);
        let config = TrainerConfig { log_every: 0, grad_clip_norm: 0.0, verbose: false };
        let mut trainer = Trainer::new(model, optimizer, mse_loss, config);

        let x = Tensor::ones(&[8, 4]);
        let y = Tensor::zeros(&[8, 2], kore_core::DType::F32);
        let ds = TensorDataset::new(&x, &y);
        let loader = DataLoader::new(Box::new(ds), 4, false, false, None);

        let metrics = trainer.evaluate(&loader);
        assert_eq!(metrics.num_batches, 2);
    }

    #[test]
    fn test_trainer_predict_smoke() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(4, 2, true)),
        ]);
        let optimizer = kore_optim::SGD::new(0.01, 0.0, 0.0);
        let config = TrainerConfig { log_every: 0, grad_clip_norm: 0.0, verbose: false };
        let mut trainer = Trainer::new(model, optimizer, mse_loss, config);

        let x = Tensor::ones(&[8, 4]);
        let y = Tensor::zeros(&[8, 2], kore_core::DType::F32);
        let ds = TensorDataset::new(&x, &y);
        let loader = DataLoader::new(Box::new(ds), 4, false, false, None);

        let preds = trainer.predict(&loader);
        assert_eq!(preds.len(), 2);
        for p in &preds {
            assert_eq!(p.shape().dims()[1], 2);
        }
    }

    #[test]
    fn test_trainer_set_lr() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(4, 2, true)),
        ]);
        let optimizer = kore_optim::Adam::default_with_lr(0.001);
        let config = TrainerConfig::default();
        let mut trainer = Trainer::new(model, optimizer, mse_loss, config);

        assert!((trainer.lr() - 0.001).abs() < 1e-6);
        trainer.set_lr(0.01);
        assert!((trainer.lr() - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_train_history_losses() {
        let mut h = TrainHistory::new();
        h.epochs.push(EpochMetrics { avg_loss: 1.0, num_batches: 2, num_samples: 8 });
        h.epochs.push(EpochMetrics { avg_loss: 0.5, num_batches: 2, num_samples: 8 });
        assert_eq!(h.losses(), vec![1.0, 0.5]);
    }
}
