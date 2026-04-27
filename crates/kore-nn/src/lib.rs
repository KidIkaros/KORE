//! # kore-nn
//!
//! Neural network layers and modules for Kore.

pub mod activations;
pub mod bit_linear;
pub mod conv;
pub mod dropout;
pub mod embedding;
pub mod geometric;
pub mod layer_norm;
pub mod linear;
pub mod lora;
pub mod loss;
pub mod module;
pub mod pool;
pub mod prelude;
pub mod quat_linear;
pub mod rms_norm;
pub mod sampler;
pub mod serialization;
pub mod squeezenet;

pub use bit_linear::BitLinear;
pub use conv::{Conv1d, Conv2d};
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use layer_norm::LayerNorm;
pub use linear::Linear;
pub use lora::{LoraLinear, QLoraLinear};
pub use loss::{cross_entropy_loss, l1_loss, mse_loss, nll_loss, FusedSoftmaxBackward};
pub use module::Module;
pub use pool::{AdaptiveAvgPool2d, AvgPool2d, MaxPool2d};
pub use quat_linear::QuatLinear;
pub use rms_norm::RMSNorm;
pub use serialization::{load_module_state, load_state_dict, save_module, save_state_dict};
pub use squeezenet::{Fire, SqueezeNet};
