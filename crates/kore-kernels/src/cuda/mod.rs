//! CUDA GPU backend for Kore.
//!
//! Provides:
//! - Device context management (lazy singleton per GPU)
//! - GPU memory allocation and host↔device transfers
//! - Kernel launcher with PTX caching
//! - Optimized CUDA kernels (GEMM, element-wise, reductions, etc.)

#![allow(
    clippy::manual_div_ceil,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

pub mod context;
pub mod launch;
pub mod memory;
pub mod ops;

use kore_core::device::{Device, DeviceBackend};
use kore_core::{KoreError, Result, Storage};
use std::sync::Arc;

/// CUDA implementation of `kore_core::device::DeviceBackend`.
#[derive(Debug, Default)]
pub struct CudaBackend;

impl DeviceBackend for CudaBackend {
    fn device_name(&self) -> &'static str {
        "cuda"
    }

    fn supports(&self, device: Device) -> bool {
        matches!(device, Device::Cuda(_))
    }

    fn copy_to_device(&self, storage: &Storage, target: Device) -> Result<Storage> {
        let dev_idx = target.cuda_index().unwrap_or(0);
        let buf = memory::CudaBuffer::from_host(dev_idx, storage.as_bytes())
            .map_err(|e| KoreError::StorageError(e.to_string()))?;
        Ok(Storage::from_device(
            storage.dtype(),
            storage.numel(),
            Arc::new(buf),
        ))
    }

    fn copy_to_cpu(&self, storage: &Storage) -> Result<Storage> {
        let dev_storage = storage
            .device_storage()
            .ok_or_else(|| KoreError::StorageError("Storage is not a device storage".into()))?;
        let cuda_buf = dev_storage
            .as_any()
            .downcast_ref::<memory::CudaBuffer>()
            .ok_or_else(|| KoreError::StorageError("DeviceStorage is not a CudaBuffer".into()))?;
        let host_bytes = cuda_buf
            .to_host()
            .map_err(|e| KoreError::StorageError(e.to_string()))?;
        Storage::from_bytes(storage.dtype(), storage.numel(), host_bytes)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.matmul(a, b)
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.add(a, b)
    }

    fn sub(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.sub(a, b)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.mul(a, b)
    }

    fn div(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.div(a, b)
    }

    fn add_scalar(&self, a: &Tensor, scalar: f32) -> Result<Tensor> {
        self.add_scalar(a, scalar)
    }

    fn mul_scalar(&self, a: &Tensor, scalar: f32) -> Result<Tensor> {
        self.mul_scalar(a, scalar)
    }

    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        self.rms_norm(input, weight, eps)
    }

    fn softmax(&self, input: &Tensor) -> Result<Tensor> {
        self.softmax(input)
    }
}

/// Register the CUDA backend with `kore_core`'s global backend registry.
pub fn register_cuda_backend() {
    kore_core::device::register_backend(Arc::new(CudaBackend));
}

use kore_core::{DType, Tensor};

impl CudaBackend {
    pub fn new() -> Self {
        register_cuda_backend();
        Self
    }

    fn get_cuda_buffer<'a>(&self, t: &'a Tensor) -> Result<(&'a memory::CudaBuffer, usize)> {
        let dev_idx = match t.device() {
            Device::Cuda(idx) => idx,
            d => {
                return Err(KoreError::DeviceMismatch {
                    expected: Device::Cuda(0),
                    got: d,
                })
            }
        };
        let dev_storage = t
            .storage_ref()
            .device_storage()
            .ok_or_else(|| KoreError::StorageError("Tensor does not have device storage".into()))?;
        let cuda_buf = dev_storage
            .as_any()
            .downcast_ref::<memory::CudaBuffer>()
            .ok_or_else(|| KoreError::StorageError("DeviceStorage is not a CudaBuffer".into()))?;
        Ok((cuda_buf, dev_idx))
    }

    fn make_tensor(
        &self,
        slice: cudarc::driver::CudaSlice<u8>,
        shape: &[usize],
        dev_idx: usize,
    ) -> Tensor {
        let numel: usize = shape.iter().product();
        let buf = memory::CudaBuffer {
            inner: Arc::new(slice),
            device_idx: dev_idx,
            nbytes: numel * 4,
        };
        let storage = Storage::from_device(DType::F32, numel, Arc::new(buf));
        Tensor::from_storage(storage, shape)
    }

    pub fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let out_shape =
            a.shape()
                .broadcast_with(b.shape())
                .ok_or_else(|| KoreError::ShapeMismatch {
                    expected: a.shape().dims().to_vec(),
                    got: b.shape().dims().to_vec(),
                })?;
        let (a_buf, dev_idx) = self.get_cuda_buffer(a)?;
        let (b_buf, _) = self.get_cuda_buffer(b)?;
        let dev =
            context::get_device(dev_idx).map_err(|e| KoreError::StorageError(e.to_string()))?;
        let out = ops::cuda_add_broadcast_f32(
            &dev,
            dev_idx,
            a_buf.as_cuda_slice(),
            b_buf.as_cuda_slice(),
            out_shape.numel(),
            a.numel(),
            b.numel(),
        )
        .map_err(|e| KoreError::StorageError(e.to_string()))?;
        Ok(self.make_tensor(out, out_shape.dims(), dev_idx))
    }

    pub fn sub(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let out_shape =
            a.shape()
                .broadcast_with(b.shape())
                .ok_or_else(|| KoreError::ShapeMismatch {
                    expected: a.shape().dims().to_vec(),
                    got: b.shape().dims().to_vec(),
                })?;
        let (a_buf, dev_idx) = self.get_cuda_buffer(a)?;
        let (b_buf, _) = self.get_cuda_buffer(b)?;
        let dev =
            context::get_device(dev_idx).map_err(|e| KoreError::StorageError(e.to_string()))?;
        let out = ops::cuda_sub_broadcast_f32(
            &dev,
            dev_idx,
            a_buf.as_cuda_slice(),
            b_buf.as_cuda_slice(),
            out_shape.numel(),
            a.numel(),
            b.numel(),
        )
        .map_err(|e| KoreError::StorageError(e.to_string()))?;
        Ok(self.make_tensor(out, out_shape.dims(), dev_idx))
    }

    pub fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let out_shape =
            a.shape()
                .broadcast_with(b.shape())
                .ok_or_else(|| KoreError::ShapeMismatch {
                    expected: a.shape().dims().to_vec(),
                    got: b.shape().dims().to_vec(),
                })?;
        let (a_buf, dev_idx) = self.get_cuda_buffer(a)?;
        let (b_buf, _) = self.get_cuda_buffer(b)?;
        let dev =
            context::get_device(dev_idx).map_err(|e| KoreError::StorageError(e.to_string()))?;
        let out = ops::cuda_mul_broadcast_f32(
            &dev,
            dev_idx,
            a_buf.as_cuda_slice(),
            b_buf.as_cuda_slice(),
            out_shape.numel(),
            a.numel(),
            b.numel(),
        )
        .map_err(|e| KoreError::StorageError(e.to_string()))?;
        Ok(self.make_tensor(out, out_shape.dims(), dev_idx))
    }

    pub fn div(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let out_shape =
            a.shape()
                .broadcast_with(b.shape())
                .ok_or_else(|| KoreError::ShapeMismatch {
                    expected: a.shape().dims().to_vec(),
                    got: b.shape().dims().to_vec(),
                })?;
        let (a_buf, dev_idx) = self.get_cuda_buffer(a)?;
        let (b_buf, _) = self.get_cuda_buffer(b)?;
        let dev =
            context::get_device(dev_idx).map_err(|e| KoreError::StorageError(e.to_string()))?;
        let out = ops::cuda_div_broadcast_f32(
            &dev,
            dev_idx,
            a_buf.as_cuda_slice(),
            b_buf.as_cuda_slice(),
            out_shape.numel(),
            a.numel(),
            b.numel(),
        )
        .map_err(|e| KoreError::StorageError(e.to_string()))?;
        Ok(self.make_tensor(out, out_shape.dims(), dev_idx))
    }

    pub fn neg(&self, a: &Tensor) -> Result<Tensor> {
        let (a_buf, dev_idx) = self.get_cuda_buffer(a)?;
        let dev =
            context::get_device(dev_idx).map_err(|e| KoreError::StorageError(e.to_string()))?;
        let out = ops::cuda_neg_f32(&dev, dev_idx, a_buf.as_cuda_slice(), a.numel())
            .map_err(|e| KoreError::StorageError(e.to_string()))?;
        Ok(self.make_tensor(out, a.shape().dims(), dev_idx))
    }

    pub fn abs(&self, a: &Tensor) -> Result<Tensor> {
        let (a_buf, dev_idx) = self.get_cuda_buffer(a)?;
        let dev =
            context::get_device(dev_idx).map_err(|e| KoreError::StorageError(e.to_string()))?;
        let out = ops::cuda_abs_f32(&dev, dev_idx, a_buf.as_cuda_slice(), a.numel())
            .map_err(|e| KoreError::StorageError(e.to_string()))?;
        Ok(self.make_tensor(out, a.shape().dims(), dev_idx))
    }

    pub fn sqrt(&self, a: &Tensor) -> Result<Tensor> {
        let (a_buf, dev_idx) = self.get_cuda_buffer(a)?;
        let dev =
            context::get_device(dev_idx).map_err(|e| KoreError::StorageError(e.to_string()))?;
        let out = ops::cuda_sqrt_f32(&dev, dev_idx, a_buf.as_cuda_slice(), a.numel())
            .map_err(|e| KoreError::StorageError(e.to_string()))?;
        Ok(self.make_tensor(out, a.shape().dims(), dev_idx))
    }

    pub fn exp(&self, a: &Tensor) -> Result<Tensor> {
        let (a_buf, dev_idx) = self.get_cuda_buffer(a)?;
        let dev =
            context::get_device(dev_idx).map_err(|e| KoreError::StorageError(e.to_string()))?;
        let out = ops::cuda_exp_f32(&dev, dev_idx, a_buf.as_cuda_slice(), a.numel())
            .map_err(|e| KoreError::StorageError(e.to_string()))?;
        Ok(self.make_tensor(out, a.shape().dims(), dev_idx))
    }

    pub fn log(&self, a: &Tensor) -> Result<Tensor> {
        let (a_buf, dev_idx) = self.get_cuda_buffer(a)?;
        let dev =
            context::get_device(dev_idx).map_err(|e| KoreError::StorageError(e.to_string()))?;
        let out = ops::cuda_log_f32(&dev, dev_idx, a_buf.as_cuda_slice(), a.numel())
            .map_err(|e| KoreError::StorageError(e.to_string()))?;
        Ok(self.make_tensor(out, a.shape().dims(), dev_idx))
    }

    pub fn add_scalar(&self, a: &Tensor, scalar: f32) -> Result<Tensor> {
        let (a_buf, dev_idx) = self.get_cuda_buffer(a)?;
        let dev =
            context::get_device(dev_idx).map_err(|e| KoreError::StorageError(e.to_string()))?;
        let out = ops::cuda_add_scalar_f32(&dev, dev_idx, a_buf.as_cuda_slice(), scalar, a.numel())
            .map_err(|e| KoreError::StorageError(e.to_string()))?;
        Ok(self.make_tensor(out, a.shape().dims(), dev_idx))
    }

    pub fn mul_scalar(&self, a: &Tensor, scalar: f32) -> Result<Tensor> {
        let (a_buf, dev_idx) = self.get_cuda_buffer(a)?;
        let dev =
            context::get_device(dev_idx).map_err(|e| KoreError::StorageError(e.to_string()))?;
        let out = ops::cuda_mul_scalar_f32(&dev, dev_idx, a_buf.as_cuda_slice(), scalar, a.numel())
            .map_err(|e| KoreError::StorageError(e.to_string()))?;
        Ok(self.make_tensor(out, a.shape().dims(), dev_idx))
    }

    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let (a_buf, dev_idx) = self.get_cuda_buffer(a)?;
        let (b_buf, _) = self.get_cuda_buffer(b)?;
        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();
        let m = a_dims[0];
        let k = a_dims[1];
        let n = b_dims[1];
        let dev =
            context::get_device(dev_idx).map_err(|e| KoreError::StorageError(e.to_string()))?;
        let out = ops::cuda_matmul_f32(
            &dev,
            dev_idx,
            a_buf.as_cuda_slice(),
            b_buf.as_cuda_slice(),
            m,
            n,
            k,
        )
        .map_err(|e| KoreError::StorageError(e.to_string()))?;
        Ok(self.make_tensor(out, &[m, n], dev_idx))
    }

    pub fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let (in_buf, dev_idx) = self.get_cuda_buffer(input)?;
        let (w_buf, _) = self.get_cuda_buffer(weight)?;
        let in_shape = input.shape().dims();
        let cols = in_shape[in_shape.len() - 1];
        let rows = input.numel() / cols;
        let dev =
            context::get_device(dev_idx).map_err(|e| KoreError::StorageError(e.to_string()))?;
        let out = ops::cuda_rms_norm_f32(
            &dev,
            dev_idx,
            in_buf.as_cuda_slice(),
            w_buf.as_cuda_slice(),
            rows,
            cols,
            eps,
        )
        .map_err(|e| KoreError::StorageError(e.to_string()))?;
        Ok(self.make_tensor(out, in_shape, dev_idx))
    }

    pub fn softmax(&self, input: &Tensor) -> Result<Tensor> {
        let (in_buf, dev_idx) = self.get_cuda_buffer(input)?;
        let in_shape = input.shape().dims();
        let cols = in_shape[in_shape.len() - 1];
        let rows = input.numel() / cols;
        let dev =
            context::get_device(dev_idx).map_err(|e| KoreError::StorageError(e.to_string()))?;
        let out = ops::cuda_softmax_f32(&dev, dev_idx, in_buf.as_cuda_slice(), rows, cols)
            .map_err(|e| KoreError::StorageError(e.to_string()))?;
        Ok(self.make_tensor(out, in_shape, dev_idx))
    }
}
