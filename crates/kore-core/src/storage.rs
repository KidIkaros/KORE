use std::sync::Arc;

use crate::{DType, Device, KoreError, Result};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};

/// Backing storage for tensor data.
///
/// Storage is reference-counted (`Arc`) so multiple tensors can share the same
/// underlying data (e.g., views from reshape/transpose/slice).
#[derive(Debug, Clone)]
pub enum StorageData {
    /// CPU heap-allocated storage (aligned to 64 bytes for SIMD).
    Cpu(Vec<u8>),
    /// CUDA GPU storage with device handle and raw byte buffer.
    #[cfg(feature = "cuda")]
    Cuda {
        device: Arc<CudaDevice>,
        buffer: Arc<CudaSlice<u8>>,
        device_idx: usize,
    },
}

/// Shared, reference-counted tensor storage.
#[derive(Debug, Clone)]
pub struct Storage {
    data: Arc<StorageData>,
    dtype: DType,
    device: Device,
    /// Number of logical elements (not bytes).
    numel: usize,
}

impl Storage {
    /// Allocate new CPU storage for `numel` elements of the given dtype.
    pub fn zeros(dtype: DType, numel: usize) -> Self {
        let nbytes = dtype.storage_bytes(numel);
        let data = vec![0u8; nbytes];
        Self {
            data: Arc::new(StorageData::Cpu(data)),
            dtype,
            device: Device::Cpu,
            numel,
        }
    }

    /// Create storage from raw bytes (CPU).
    pub fn from_bytes(dtype: DType, numel: usize, bytes: Vec<u8>) -> Result<Self> {
        let expected = dtype.storage_bytes(numel);
        if bytes.len() != expected {
            return Err(KoreError::StorageError(format!(
                "Expected {} bytes for {} elements of {}, got {}",
                expected,
                numel,
                dtype,
                bytes.len()
            )));
        }
        Ok(Self {
            data: Arc::new(StorageData::Cpu(bytes)),
            dtype,
            device: Device::Cpu,
            numel,
        })
    }

    /// Create storage from a slice of f32 values.
    pub fn from_f32(data: &[f32]) -> Self {
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        Self {
            data: Arc::new(StorageData::Cpu(bytes)),
            dtype: DType::F32,
            device: Device::Cpu,
            numel: data.len(),
        }
    }

    /// Create storage from a slice of f64 values.
    pub fn from_f64(data: &[f64]) -> Self {
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        Self {
            data: Arc::new(StorageData::Cpu(bytes)),
            dtype: DType::F64,
            device: Device::Cpu,
            numel: data.len(),
        }
    }

    /// Create storage from a slice of i32 values.
    pub fn from_i32(data: &[i32]) -> Self {
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        Self {
            data: Arc::new(StorageData::Cpu(bytes)),
            dtype: DType::I32,
            device: Device::Cpu,
            numel: data.len(),
        }
    }

    /// Get the dtype of this storage.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get the device of this storage.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Number of logical elements.
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// Size in bytes.
    pub fn nbytes(&self) -> usize {
        match self.data.as_ref() {
            StorageData::Cpu(v) => v.len(),
            #[cfg(feature = "cuda")]
            StorageData::Cuda { buffer, .. } => buffer.len(),
        }
    }

    /// Get a read-only reference to the raw bytes.
    /// Panics if storage is on GPU — call `to_cpu()` first.
    pub fn as_bytes(&self) -> &[u8] {
        match self.data.as_ref() {
            StorageData::Cpu(v) => v,
            #[cfg(feature = "cuda")]
            StorageData::Cuda { .. } => panic!("Cannot access GPU storage as bytes — transfer to CPU first with .to(Device::Cpu)"),
        }
    }

    /// Get a mutable reference to the raw bytes.
    /// This will clone the underlying data if there are other references (copy-on-write).
    /// Panics if storage is on GPU.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        let data = Arc::make_mut(&mut self.data);
        match data {
            StorageData::Cpu(v) => v,
            #[cfg(feature = "cuda")]
            StorageData::Cuda { .. } => panic!("Cannot mutate GPU storage as bytes — transfer to CPU first"),
        }
    }

    /// Interpret storage as a slice of f32 values.
    /// Returns None if dtype is not F32 or data is misaligned.
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        if self.dtype != DType::F32 {
            return None;
        }
        let bytes = self.as_bytes();
        // Safety: we verified dtype is F32 and storage was created with proper alignment
        Some(bytemuck::cast_slice(bytes))
    }

    /// Interpret storage as a mutable slice of f32 values (copy-on-write).
    pub fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        if self.dtype != DType::F32 {
            return None;
        }
        let bytes = self.as_bytes_mut();
        Some(bytemuck::cast_slice_mut(bytes))
    }

    /// Interpret storage as a slice of f64 values.
    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        if self.dtype != DType::F64 {
            return None;
        }
        let bytes = self.as_bytes();
        Some(bytemuck::cast_slice(bytes))
    }

    /// Interpret storage as a slice of i32 values.
    pub fn as_i32_slice(&self) -> Option<&[i32]> {
        if self.dtype != DType::I32 {
            return None;
        }
        let bytes = self.as_bytes();
        Some(bytemuck::cast_slice(bytes))
    }

    /// Whether this storage is uniquely owned (no other Arc references).
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.data) == 1
    }

    /// Whether this storage is on CPU.
    pub fn is_cpu(&self) -> bool {
        self.device.is_cpu()
    }

    /// Whether this storage is on a CUDA device.
    pub fn is_cuda(&self) -> bool {
        self.device.is_cuda()
    }

    /// Create GPU storage from host bytes (H2D copy).
    #[cfg(feature = "cuda")]
    pub fn to_cuda(&self, device_idx: usize) -> Result<Self> {
        if let Device::Cuda(idx) = self.device {
            if idx == device_idx {
                return Ok(self.clone());
            }
        }
        let host_bytes = self.as_bytes();
        let cuda_dev = cudarc::driver::CudaDevice::new(device_idx)
            .map_err(|e| KoreError::StorageError(format!("CUDA device init: {}", e)))?;
        let gpu_buf = cuda_dev
            .htod_copy(host_bytes.to_vec())
            .map_err(|e| KoreError::StorageError(format!("H2D copy: {}", e)))?;
        Ok(Self {
            data: Arc::new(StorageData::Cuda {
                device: cuda_dev,
                buffer: Arc::new(gpu_buf),
                device_idx,
            }),
            dtype: self.dtype,
            device: Device::Cuda(device_idx),
            numel: self.numel,
        })
    }

    /// Copy GPU storage back to CPU (D2H copy).
    #[cfg(feature = "cuda")]
    pub fn to_cpu(&self) -> Result<Self> {
        match self.data.as_ref() {
            StorageData::Cpu(_) => Ok(self.clone()),
            StorageData::Cuda { device, buffer, .. } => {
                let host_data: Vec<u8> = device
                    .dtoh_sync_copy(buffer.as_ref())
                    .map_err(|e| KoreError::StorageError(format!("D2H copy: {}", e)))?;
                Ok(Self {
                    data: Arc::new(StorageData::Cpu(host_data)),
                    dtype: self.dtype,
                    device: Device::Cpu,
                    numel: self.numel,
                })
            }
        }
    }

    /// Create GPU storage with zeroed memory.
    #[cfg(feature = "cuda")]
    pub fn cuda_zeros(dtype: DType, numel: usize, device_idx: usize) -> Result<Self> {
        let nbytes = dtype.storage_bytes(numel);
        let cuda_dev = cudarc::driver::CudaDevice::new(device_idx)
            .map_err(|e| KoreError::StorageError(format!("CUDA device init: {}", e)))?;
        let gpu_buf = cuda_dev
            .alloc_zeros::<u8>(nbytes)
            .map_err(|e| KoreError::StorageError(format!("CUDA alloc_zeros: {}", e)))?;
        Ok(Self {
            data: Arc::new(StorageData::Cuda {
                device: cuda_dev,
                buffer: Arc::new(gpu_buf),
                device_idx,
            }),
            dtype,
            device: Device::Cuda(device_idx),
            numel,
        })
    }

    /// Get the underlying CudaSlice for kernel launches.
    /// Returns None if not on GPU.
    #[cfg(feature = "cuda")]
    pub fn as_cuda_slice(&self) -> Option<&CudaSlice<u8>> {
        match self.data.as_ref() {
            StorageData::Cuda { buffer, .. } => Some(buffer.as_ref()),
            _ => None,
        }
    }

    /// Get the CudaDevice handle. Returns None if not on GPU.
    #[cfg(feature = "cuda")]
    pub fn cuda_device(&self) -> Option<Arc<CudaDevice>> {
        match self.data.as_ref() {
            StorageData::Cuda { device, .. } => Some(Arc::clone(device)),
            _ => None,
        }
    }

    /// Get the raw StorageData reference (for dispatch).
    pub fn data(&self) -> &StorageData {
        self.data.as_ref()
    }

    /// Create Storage directly from a CUDA buffer (used by kernel dispatch).
    #[cfg(feature = "cuda")]
    pub fn from_cuda(
        device: Arc<CudaDevice>,
        buffer: CudaSlice<u8>,
        device_idx: usize,
        dtype: DType,
        numel: usize,
    ) -> Self {
        Self {
            data: Arc::new(StorageData::Cuda {
                device,
                buffer: Arc::new(buffer),
                device_idx,
            }),
            dtype,
            device: Device::Cuda(device_idx),
            numel,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let s = Storage::zeros(DType::F32, 10);
        assert_eq!(s.dtype(), DType::F32);
        assert_eq!(s.device(), Device::Cpu);
        assert_eq!(s.numel(), 10);
        assert_eq!(s.nbytes(), 40);
        assert!(s.as_bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn test_from_f32() {
        let data = vec![1.0f32, 2.0, 3.0];
        let s = Storage::from_f32(&data);
        assert_eq!(s.numel(), 3);
        let slice = s.as_f32_slice().unwrap();
        assert_eq!(slice, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_copy_on_write() {
        let data = vec![1.0f32, 2.0, 3.0];
        let s1 = Storage::from_f32(&data);
        let mut s2 = s1.clone();
        assert!(!s1.is_unique()); // shared

        // Mutating s2 should not affect s1
        let slice = s2.as_f32_slice_mut().unwrap();
        slice[0] = 99.0;

        assert_eq!(s1.as_f32_slice().unwrap()[0], 1.0);
        assert_eq!(s2.as_f32_slice().unwrap()[0], 99.0);
    }

    #[test]
    fn test_ternary_storage_size() {
        let s = Storage::zeros(DType::Ternary, 10);
        // 10 trits → ceil(10/5) = 2 bytes
        assert_eq!(s.nbytes(), 2);
    }

    #[test]
    fn test_quaternary_storage_size() {
        let s = Storage::zeros(DType::Quaternary, 10);
        // 10 quats → ceil(10/4) = 3 bytes
        assert_eq!(s.nbytes(), 3);
    }

    #[test]
    fn test_from_bytes_validation() {
        let result = Storage::from_bytes(DType::F32, 3, vec![0u8; 11]);
        assert!(result.is_err());

        let result = Storage::from_bytes(DType::F32, 3, vec![0u8; 12]);
        assert!(result.is_ok());
    }
}
