//! Vulkan backend for KORE operations.

use std::sync::Arc;

use kore_core::{Tensor, DType, Device, Storage};
use vulkan_kernels::api::ComputeContext;

use crate::buffer::{KoreVulkanBuffer, ToVulkanBuffer};
use crate::error::{Result, VulkanError};

/// Helper to calculate total storage bytes
fn storage_bytes(dtype: DType, numel: usize) -> usize {
    dtype.storage_bytes(numel)
}

/// Vulkan compute backend for KORE.
///
/// Wraps a `vulkan_kernels::ComputeContext` and provides KORE-compatible
/// tensor operations.
pub struct VulkanBackend {
    context: Arc<ComputeContext>,
    device_index: usize,
}

impl VulkanBackend {
    /// Create new Vulkan backend with default device.
    pub fn new() -> Result<Self> {
        Self::with_device(0)
    }
    
    /// Create backend with specific device index.
    pub fn with_device(device_index: usize) -> Result<Self> {
        use vulkan_kernels::runtime::ash_runtime::AshRuntime;
        use vulkan_kernels::device::DeviceSelector;
        
        let selector = DeviceSelector::portable();
        let runtime = AshRuntime::new(selector, 256 * 1024 * 1024, None)
            .map_err(|e| VulkanError::Initialization(e.to_string()))?;
        
        let context = ComputeContext::new(Arc::new(runtime), true)
            .map_err(|e| VulkanError::Initialization(e.to_string()))?;
        
        Ok(Self {
            context,
            device_index,
        })
    }
    
    /// Get the device this backend manages.
    pub fn device(&self) -> Device {
        Device::Vulkan(self.device_index)
    }
    
    /// Matrix multiplication: C = A @ B
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.validate_device(a)?;
        self.validate_device(b)?;
        
        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();
        
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(VulkanError::UnsupportedOperation(
                "Only 2D matmul supported for now".into()
            ));
        }
        
        if a_shape[1] != b_shape[0] {
            return Err(VulkanError::ShapeMismatch {
                op: "matmul".into(),
                lhs: a_shape.to_vec(),
                rhs: b_shape.to_vec(),
            });
        }
        
        let m = a_shape[0];
        let n = b_shape[1];
        let _k = a_shape[1];
        
        // Upload tensors to Vulkan
        let _a_buf = a.storage_ref().to_vulkan(&self.context)?;
        let _b_buf = b.storage_ref().to_vulkan(&self.context)?;
        
        // Create output buffer
        let output_size = storage_bytes(a.dtype(), m * n);
        let c_vulkan_buf = self.context.create_buffer(output_size as u64, 
            vulkan_kernels::runtime::BufferUsage::storage())
            .map_err(|e| VulkanError::Kernel(e.to_string()))?;
        
        // Execute GEMM kernel with proper push constants
        // Push constants layout: M, N, K, lda, ldb, ldc, alpha, beta
        let push_constants: Vec<u8> = [
            (m as u32).to_ne_bytes(),
            (n as u32).to_ne_bytes(),
            (a_shape[1] as u32).to_ne_bytes(),
            (a_shape[1] as u32).to_ne_bytes(), // lda = K
            (b_shape[1] as u32).to_ne_bytes(), // ldb = N
            (n as u32).to_ne_bytes(),          // ldc = N
            (1.0f32).to_ne_bytes(),            // alpha = 1.0
            (0.0f32).to_ne_bytes(),            // beta = 0.0
        ]
        .concat();
        
        self.context.execute_kernel(
            "gemm",
            &[_a_buf.vulkan_buffer(), _b_buf.vulkan_buffer(), c_vulkan_buf.clone()],
            &push_constants,
            vulkan_kernels::runtime::WorkgroupSize::new_2d(16, 16),
            vulkan_kernels::runtime::DispatchSize::new_2d(
                ((m as u32 + 31) / 32).max(1),  // TILE_M = 32
                ((n as u32 + 31) / 32).max(1),  // TILE_N = 32
            ),
        ).map_err(|e| VulkanError::Kernel(e.to_string()))?;
        
        // Download result back to CPU storage
        let c_buf = KoreVulkanBuffer::new(c_vulkan_buf, a.dtype(), vec![m, n]);
        let c_storage = c_buf.to_storage()?;
        
        Ok(Tensor::from_storage(c_storage, &[m, n]))
    }
    
    /// Element-wise addition: C = A + B
    pub fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Note: "elementwise_add" is an alias for "elementwise" kernel
        // Operation selection requires specialization constants (not yet supported in execute_kernel API)
        self.elementwise_binary(a, b, "elementwise")
    }
    
    /// Element-wise multiplication: C = A * B
    pub fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Note: "elementwise_mul" is an alias for "elementwise" kernel
        self.elementwise_binary(a, b, "elementwise")
    }
    
    /// Softmax along last dimension
    pub fn softmax(&self, input: &Tensor) -> Result<Tensor> {
        self.validate_device(input)?;
        
        let shape = input.shape().dims().to_vec();
        let numel = input.numel();
        
        // Simplified implementation - actual Vulkan dispatch would go here
        let out_data = vec![0u8; storage_bytes(input.dtype(), numel)];
        let out_storage = Storage::from_bytes(input.dtype(), numel, out_data)
            .map_err(|e| VulkanError::Transfer(e.to_string()))?;
        
        Ok(Tensor::from_storage(out_storage, &shape))
    }
    
    /// RMS Normalization (placeholder)
    pub fn rms_norm(&self, input: &Tensor, _weight: &Tensor, _eps: f32) -> Result<Tensor> {
        self.validate_device(input)?;
        
        let shape = input.shape().dims().to_vec();
        let numel = input.numel();
        
        let out_data = vec![0u8; storage_bytes(input.dtype(), numel)];
        let out_storage = Storage::from_bytes(input.dtype(), numel, out_data)
            .map_err(|e| VulkanError::Transfer(e.to_string()))?;
        
        Ok(Tensor::from_storage(out_storage, &shape))
    }
    
    /// Flash Attention (placeholder)
    pub fn flash_attention(
        &self,
        q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _scale: f32,
    ) -> Result<Tensor> {
        self.validate_device(q)?;
        
        let seq_len = q.shape().dims()[0];
        let head_dim = q.shape().dims()[1];
        let numel = seq_len * head_dim;
        
        let out_data = vec![0u8; storage_bytes(q.dtype(), numel)];
        let out_storage = Storage::from_bytes(q.dtype(), numel, out_data)
            .map_err(|e| VulkanError::Transfer(e.to_string()))?;
        
        Ok(Tensor::from_storage(out_storage, &[seq_len, head_dim]))
    }
    
    // Internal helpers
    
    fn validate_device(&self, tensor: &Tensor) -> Result<()> {
        let expected = Device::Vulkan(self.device_index);
        let actual = tensor.device().clone();
        
        // Allow CPU tensors (will be uploaded)
        if actual.is_cpu() || actual == expected {
            Ok(())
        } else {
            Err(VulkanError::DeviceMismatch {
                expected: format!("{:?}", expected),
                actual: format!("{:?}", actual),
            })
        }
    }
    
    fn elementwise_binary(&self, a: &Tensor, b: &Tensor, _kernel: &str) -> Result<Tensor> {
        self.validate_device(a)?;
        self.validate_device(b)?;
        
        if a.shape().dims() != b.shape().dims() {
            return Err(VulkanError::ShapeMismatch {
                op: "elementwise".into(),
                lhs: a.shape().dims().to_vec(),
                rhs: b.shape().dims().to_vec(),
            });
        }
        
        let shape = a.shape().dims().to_vec();
        let numel = a.numel();
        
        // Upload input tensors
        let a_buf = a.storage_ref().to_vulkan(&self.context)?;
        let b_buf = b.storage_ref().to_vulkan(&self.context)?;
        
        // Create output buffer
        let output_size = storage_bytes(a.dtype(), numel);
        let c_vulkan_buf = self.context.create_buffer(output_size as u64,
            vulkan_kernels::runtime::BufferUsage::storage())
            .map_err(|e| VulkanError::Kernel(e.to_string()))?;
        
        // Execute elementwise kernel
        // Note: Full operation selection requires specialization constants support
        let push_constants: Vec<u8> = [
            (numel as u32).to_ne_bytes(),
            (1.0f32).to_ne_bytes(), // alpha
            (0.0f32).to_ne_bytes(), // beta
        ]
        .concat();
        
        self.context.execute_kernel(
            _kernel, // "elementwise" kernel - operation selected via specialization
            &[a_buf.vulkan_buffer(), b_buf.vulkan_buffer(), c_vulkan_buf.clone(), c_vulkan_buf.clone()],
            &push_constants,
            vulkan_kernels::runtime::WorkgroupSize::new_1d(256),
            vulkan_kernels::runtime::DispatchSize::new_1d(((numel as u32 + 255) / 256).max(1)),
        ).map_err(|e| VulkanError::Kernel(e.to_string()))?;
        
        // Download result
        let c_buf = KoreVulkanBuffer::new(c_vulkan_buf, a.dtype(), shape.clone());
        let out_storage = c_buf.to_storage()?;
        
        Ok(Tensor::from_storage(out_storage, &shape))
    }
}

impl Default for VulkanBackend {
    fn default() -> Self {
        Self::new().expect("Failed to create default Vulkan backend")
    }
}
