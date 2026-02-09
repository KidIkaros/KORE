//! GPU memory allocation and host↔device transfer utilities.
//!
//! Wraps `cudarc` allocation into Kore's storage model.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};

use super::context::{get_device, CudaError};

/// A GPU memory buffer holding raw bytes on a specific CUDA device.
///
/// This is the GPU counterpart of `Vec<u8>` in `StorageData::Cpu`.
/// Reference-counted via `Arc` so multiple tensors can share GPU memory.
#[derive(Debug, Clone)]
pub struct CudaBuffer {
    /// The raw device allocation.
    pub(crate) inner: Arc<CudaSlice<u8>>,
    /// Device index this buffer lives on.
    pub(crate) device_idx: usize,
    /// Number of bytes.
    pub(crate) nbytes: usize,
}

impl CudaBuffer {
    /// Allocate zeroed GPU memory.
    pub fn zeros(device_idx: usize, nbytes: usize) -> Result<Self, CudaError> {
        let dev = get_device(device_idx)?;
        let slice = dev
            .alloc_zeros::<u8>(nbytes)
            .map_err(|e| CudaError::MemoryError(format!("alloc_zeros({} bytes): {}", nbytes, e)))?;
        Ok(Self {
            inner: Arc::new(slice),
            device_idx,
            nbytes,
        })
    }

    /// Copy host bytes to a new GPU buffer (H2D).
    pub fn from_host(device_idx: usize, data: &[u8]) -> Result<Self, CudaError> {
        let dev = get_device(device_idx)?;
        let slice = dev
            .htod_copy(data.to_vec())
            .map_err(|e| CudaError::MemoryError(format!("htod_copy({} bytes): {}", data.len(), e)))?;
        Ok(Self {
            inner: Arc::new(slice),
            device_idx,
            nbytes: data.len(),
        })
    }

    /// Copy GPU buffer back to host (D2H).
    pub fn to_host(&self) -> Result<Vec<u8>, CudaError> {
        let dev = get_device(self.device_idx)?;
        let data = dev
            .dtoh_sync_copy(&*self.inner)
            .map_err(|e| CudaError::MemoryError(format!("dtoh_sync_copy: {}", e)))?;
        Ok(data)
    }

    /// Number of bytes in this buffer.
    pub fn len(&self) -> usize {
        self.nbytes
    }

    /// Whether this buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.nbytes == 0
    }

    /// Device index.
    pub fn device_idx(&self) -> usize {
        self.device_idx
    }

    /// Get a reference to the underlying CudaSlice for kernel launches.
    pub fn as_cuda_slice(&self) -> &CudaSlice<u8> {
        &self.inner
    }

    /// Whether this buffer is uniquely owned (no other Arc references).
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.inner) == 1
    }

    /// Get the CudaDevice this buffer is on.
    pub fn device(&self) -> Result<Arc<CudaDevice>, CudaError> {
        get_device(self.device_idx)
    }
}

/// Copy a GPU buffer to a different device (D2D via host staging).
pub fn copy_to_device(buf: &CudaBuffer, target_device: usize) -> Result<CudaBuffer, CudaError> {
    if buf.device_idx == target_device {
        return Ok(buf.clone());
    }
    // D2D via host staging (simple path; peer-to-peer can be added later)
    let host_data = buf.to_host()?;
    CudaBuffer::from_host(target_device, &host_data)
}
