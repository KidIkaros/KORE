//! # kore-optim
//!
//! Optimizers and learning rate schedulers for Kore.

pub mod sgd;
pub mod adam;
pub mod scheduler;
pub mod clip;

pub use sgd::SGD;
pub use adam::{Adam, ParamGroup};
pub use scheduler::{LrScheduler, CosineAnnealing, WarmupCosine, OneCycle, StepDecay};
pub use clip::{clip_grad_norm_, clip_grad_value_};

use kore_core::Tensor;

/// Trait for all optimizers.
///
/// Provides a unified interface for parameter updates, matching the
/// signature used by both `SGD` and `Adam`.
pub trait Optimizer: Send {
    /// Perform one optimization step given current parameters and their gradients.
    fn step(&mut self, params: &mut [&mut Tensor], grads: &[Tensor]);

    /// Get the current learning rate.
    fn lr(&self) -> f32;

    /// Set the learning rate.
    fn set_lr(&mut self, lr: f32);
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut [&mut Tensor], grads: &[Tensor]) {
        SGD::step(self, params, grads);
    }
    fn lr(&self) -> f32 { SGD::lr(self) }
    fn set_lr(&mut self, lr: f32) { self.set_lr(lr); }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut [&mut Tensor], grads: &[Tensor]) {
        Adam::step(self, params, grads);
    }
    fn lr(&self) -> f32 { Adam::lr(self) }
    fn set_lr(&mut self, lr: f32) { Adam::set_lr(self, lr); }
}

impl Optimizer for Box<dyn Optimizer> {
    fn step(&mut self, params: &mut [&mut Tensor], grads: &[Tensor]) {
        (**self).step(params, grads);
    }
    fn lr(&self) -> f32 { (**self).lr() }
    fn set_lr(&mut self, lr: f32) { (**self).set_lr(lr); }
}
