//! # kore-nn
//!
//! Neural network layers and modules for Kore.

pub mod module;
pub mod linear;
pub mod activations;

pub use module::Module;
pub use linear::Linear;
