//! Autograd support for Mamba-3 scan operations.
//!
//! Provides `MambaScanBackward`, a custom `GradFn` that analytically computes
//! gradients through the Mamba-3 trapezoidal SSM recurrence.

mod scan_backward;
pub mod scan_forward_saved;

#[cfg(test)]
mod tests;

pub use scan_backward::MambaScanBackward;
pub use scan_forward_saved::{MambaScanSaved, mamba3_scan_with_grad};
#[cfg(feature = "cuda")]
pub use scan_forward_saved::GpuSavedContext;
