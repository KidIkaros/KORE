//! Error types for kore-vulkan.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, VulkanError>;

#[derive(Error, Debug)]
pub enum VulkanError {
    #[error("Vulkan initialization failed: {0}")]
    Initialization(String),

    #[error("No Vulkan-capable device found")]
    NoDevice,

    #[error("Operation not supported: {0}")]
    UnsupportedOperation(String),

    #[error("Tensor device mismatch: expected {expected}, got {actual}")]
    DeviceMismatch { expected: String, actual: String },

    #[error("Shape mismatch for {op}: {lhs:?} vs {rhs:?}")]
    ShapeMismatch {
        op: String,
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },

    #[error("Vulkan kernel error: {0}")]
    Kernel(String),

    #[error("Buffer transfer failed: {0}")]
    Transfer(String),

    #[error("Quantized operations not yet implemented for Vulkan")]
    QuantizedNotImplemented,

    #[error("Backend not initialized")]
    NotInitialized,
}

impl From<vulkan_kernels::VulkanKernelError> for VulkanError {
    fn from(e: vulkan_kernels::VulkanKernelError) -> Self {
        VulkanError::Kernel(e.to_string())
    }
}
