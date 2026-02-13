//! Convenience re-exports for common kore-nn types.
//!
//! ```rust
//! use kore_nn::prelude::*;
//! ```

pub use crate::Module;
pub use crate::Linear;
pub use crate::BitLinear;
pub use crate::QuatLinear;
pub use crate::{LoraLinear, QLoraLinear};
pub use crate::LayerNorm;
pub use crate::RMSNorm;
pub use crate::Embedding;
pub use crate::Dropout;
pub use crate::{Conv1d, Conv2d};
pub use crate::{MaxPool2d, AvgPool2d, AdaptiveAvgPool2d};
pub use crate::Sequential;
pub use crate::ModuleList;
pub use crate::{Trainer, TrainerConfig};
