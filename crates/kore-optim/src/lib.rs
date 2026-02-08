//! # kore-optim
//!
//! Optimizers and learning rate schedulers for Kore.

pub mod sgd;
pub mod adam;

pub use sgd::SGD;
pub use adam::Adam;
