use std::collections::HashMap;

use kore_core::{Tensor, Result};

/// Base trait for all neural network modules.
///
/// Implement this trait to define custom layers and models.
/// Use `#[derive(Module)]` (future) for automatic parameter collection.
pub trait Module: Send + Sync {
    /// Forward pass.
    fn forward(&self, input: &Tensor) -> Result<Tensor>;

    /// Get all trainable parameters.
    fn parameters(&self) -> Vec<&Tensor>;

    /// Get mutable references to all trainable parameters.
    ///
    /// Returns parameters in the same order as `parameters()`. This enables
    /// in-place optimizer updates without cloning, avoiding copy-on-write
    /// overhead from the Arc-based Tensor storage.
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;

    /// Get named parameters (for state_dict).
    fn named_parameters(&self) -> Vec<(String, &Tensor)>;

    /// Write updated parameters back into the module.
    ///
    /// Consumes parameters from the front of the slice in the same order
    /// as `parameters()` returns them. Returns how many were consumed.
    fn set_parameters(&mut self, _params: &[Tensor]) -> usize { 0 }

    /// Deep-clone this module into a boxed trait object.
    ///
    /// Preserves all internal state including packed/quantized weights.
    fn clone_box(&self) -> Box<dyn Module>;

    /// Number of non-differentiable (quantized/packed) weight elements in this module.
    ///
    /// Quantized layers like `BitLinear` and `QuatLinear` store weights in
    /// packed integer formats that are not `Tensor` parameters and cannot be
    /// updated via backpropagation. This method returns the count of such
    /// elements so that callers (e.g. `Trainer`) can warn users about layers
    /// whose weights are frozen during training. Default is 0.
    fn num_quantized_params(&self) -> usize { 0 }

    /// Set training/eval mode.
    fn train(&mut self, _mode: bool) {}

    /// Whether the module is in training mode.
    fn is_training(&self) -> bool {
        true
    }

    /// Export state dictionary.
    fn state_dict(&self) -> HashMap<String, Tensor> {
        self.named_parameters()
            .into_iter()
            .map(|(name, t)| (name, t.clone()))
            .collect()
    }
}
