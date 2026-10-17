//! # kore-vulkan
//!
//! Vulkan compute backend for KORE ML framework.
//!
//! Bridges `kore-core` tensors with `vulkan-kernels` compute operations,
//! enabling cross-platform GPU acceleration without CUDA.
//!
//! ## Features
//! - Matrix operations (GEMM, batch matmul)
//! - Flash Attention with KV cache
//! - LayerNorm, RMSNorm, Softmax
//! - RoPE (Rotary Position Embeddings)
//! - Quantized operations (2-bit, 1.58-bit) - *experimental*
//! - Kernel fusion optimizer
//!
//! ## Example
//! ```rust,no_run
//! use kore_vulkan::VulkanBackend;
//! use kore_core::Tensor;
//!
//! // Initialize Vulkan backend
//! let backend = VulkanBackend::new().unwrap();
//!
//! // Create CPU tensors
//! let x = Tensor::randn(&[1024, 1024]);
//! let y = Tensor::randn(&[1024, 1024]);
//!
//! // Perform matmul via Vulkan
//! let z = backend.matmul(&x, &y).unwrap();
//! ```

pub mod backend;
pub mod buffer;
pub mod error;
pub mod ops;

pub use backend::{op_counts, register_vulkan_backend, reset_op_counts, VulkanBackend};
pub use buffer::{buffer_alloc_stats, KoreVulkanBuffer, ToVulkanBuffer};
pub use error::{Result, VulkanError};

#[cfg(test)]
mod tests {
    use super::VulkanBackend;

    #[test]
    fn test_backend_type_exists() {
        let _ = std::mem::size_of::<VulkanBackend>();
    }
}
