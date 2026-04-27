//! Tensor Storage ↔ Vulkan Buffer bridge.
//!
//! Maps KORE's `Storage` to vulkan-kernels `Buffer` for compute operations.

use std::sync::Arc;

use kore_core::{DType, Storage};
use vulkan_kernels::runtime::{Buffer as VulkanBuffer, BufferUsage};

use crate::error::{Result, VulkanError};

/// Wraps a Vulkan buffer with KORE-compatible interface.
pub struct KoreVulkanBuffer {
    buffer: Arc<dyn VulkanBuffer>,
    dtype: DType,
    shape: Vec<usize>,
}

impl KoreVulkanBuffer {
    /// Create a new KoreVulkanBuffer from components.
    pub fn new(buffer: Arc<dyn VulkanBuffer>, dtype: DType, shape: Vec<usize>) -> Self {
        Self {
            buffer,
            dtype,
            shape,
        }
    }

    /// Create from KORE storage (transfers data to Vulkan).
    pub fn from_storage(
        storage: &Storage,
        context: &vulkan_kernels::api::ComputeContext,
    ) -> Result<Self> {
        let dtype = storage.dtype();
        let numel = storage.numel();
        let nbytes = dtype.storage_bytes(numel);

        // Create Vulkan buffer
        let usage = BufferUsage::storage();
        let vulkan_buffer = context
            .create_buffer(nbytes as u64, usage)
            .map_err(|e| VulkanError::Transfer(e.to_string()))?;

        // Upload data if on CPU
        if storage.device().is_cpu() {
            let data = storage.as_bytes();
            vulkan_buffer
                .copy_from_host(data, 0)
                .map_err(|e| VulkanError::Transfer(e.to_string()))?;
        }

        // Note: Storage doesn't have a shape method - we need to track it separately
        // For now, use a default shape based on numel
        Ok(Self {
            buffer: vulkan_buffer,
            dtype,
            shape: vec![numel], // Simplified - caller should provide actual shape
        })
    }

    /// Download data back to CPU storage.
    pub fn to_storage(&self) -> Result<Storage> {
        let numel = self.shape.iter().product();
        let nbytes = self.dtype.storage_bytes(numel);
        let mut data = vec![0u8; nbytes];

        self.buffer
            .copy_to_host(&mut data, 0)
            .map_err(|e| VulkanError::Transfer(e.to_string()))?;

        Storage::from_bytes(self.dtype, numel, data)
            .map_err(|e| VulkanError::Transfer(e.to_string()))
    }

    /// Get raw Vulkan buffer for kernel dispatch.
    pub fn vulkan_buffer(&self) -> Arc<dyn VulkanBuffer> {
        self.buffer.clone()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

/// Trait for KORE types that can provide Vulkan buffers.
pub trait ToVulkanBuffer {
    fn to_vulkan(&self, context: &vulkan_kernels::api::ComputeContext) -> Result<KoreVulkanBuffer>;
}

impl ToVulkanBuffer for Storage {
    fn to_vulkan(&self, context: &vulkan_kernels::api::ComputeContext) -> Result<KoreVulkanBuffer> {
        KoreVulkanBuffer::from_storage(self, context)
    }
}
