//! # kore-nn
//!
//! Neural network layers and modules for Kore.

pub mod module;
pub mod linear;
pub mod bit_linear;
pub mod quat_linear;
pub mod lora;
pub mod activations;
pub mod layer_norm;
pub mod embedding;
pub mod dropout;
pub mod conv;
pub mod pool;
pub mod loss;
pub mod geometric;
pub mod squeezenet;
pub mod serialization;

pub use module::Module;
pub use linear::Linear;
pub use bit_linear::BitLinear;
pub use quat_linear::QuatLinear;
pub use lora::{LoraLinear, QLoraLinear};
pub use layer_norm::LayerNorm;
pub use embedding::Embedding;
pub use dropout::Dropout;
pub use conv::{Conv1d, Conv2d};
pub use pool::{MaxPool2d, AvgPool2d, AdaptiveAvgPool2d};
pub use squeezenet::{Fire, SqueezeNet};
pub use loss::{cross_entropy_loss, mse_loss, l1_loss, nll_loss};
pub use serialization::{save_state_dict, load_state_dict, save_module, load_module_state};
