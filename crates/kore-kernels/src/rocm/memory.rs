//! HIP GPU memory allocation and host↔device transfer utilities.
//!
//! Mirrors `cuda/memory.rs` using runtime-loaded HIP function pointers.

use std::ffi::c_void;

use super::context::{init_device, set_device, device_synchronize, RocmError};
use super::ffi::{self, check_hip, HipDeviceptr, HIP_MEMCPY_HOST_TO_DEVICE, HIP_MEMCPY_DEVICE_TO_HOST};

/// A GPU memory buffer holding raw bytes on a specific HIP device.
///
/// Automatically freed via `hipFree` on drop.
pub struct HipBuffer {
    ptr: HipDeviceptr,
    device_idx: usize,
    nbytes: usize,
}

// HIP device pointers are safe to send across threads.
unsafe impl Send for HipBuffer {}
unsafe impl Sync for HipBuffer {}

impl HipBuffer {
    /// Allocate zeroed GPU memory.
    pub fn zeros(device_idx: usize, nbytes: usize) -> Result<Self, RocmError> {
        if nbytes == 0 {
            return Ok(Self { ptr: std::ptr::null_mut(), device_idx, nbytes: 0 });
        }
        let api = ffi::hip_api().ok_or(RocmError::NotAvailable)?;
        init_device(device_idx)?;
        set_device(device_idx)?;

        let mut ptr: HipDeviceptr = std::ptr::null_mut();
        check_hip(
            unsafe { (api.hip_malloc)(&mut ptr, nbytes) },
            &format!("hipMalloc({} bytes)", nbytes),
        )?;
        check_hip(
            unsafe { (api.hip_memset)(ptr, 0, nbytes) },
            &format!("hipMemset({} bytes)", nbytes),
        )?;
        Ok(Self { ptr, device_idx, nbytes })
    }

    /// Copy host bytes to a new GPU buffer (H2D).
    pub fn from_host(device_idx: usize, data: &[u8]) -> Result<Self, RocmError> {
        if data.is_empty() {
            return Ok(Self { ptr: std::ptr::null_mut(), device_idx, nbytes: 0 });
        }
        let api = ffi::hip_api().ok_or(RocmError::NotAvailable)?;
        init_device(device_idx)?;
        set_device(device_idx)?;

        let nbytes = data.len();
        let mut ptr: HipDeviceptr = std::ptr::null_mut();
        check_hip(
            unsafe { (api.hip_malloc)(&mut ptr, nbytes) },
            &format!("hipMalloc({} bytes)", nbytes),
        )?;
        check_hip(
            unsafe {
                (api.hip_memcpy)(
                    ptr,
                    data.as_ptr() as *const c_void,
                    nbytes,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                )
            },
            "hipMemcpy H2D",
        )?;
        Ok(Self { ptr, device_idx, nbytes })
    }

    /// Copy GPU buffer back to host (D2H).
    pub fn to_host(&self) -> Result<Vec<u8>, RocmError> {
        if self.nbytes == 0 {
            return Ok(Vec::new());
        }
        let api = ffi::hip_api().ok_or(RocmError::NotAvailable)?;
        set_device(self.device_idx)?;
        device_synchronize()?;

        let mut host = vec![0u8; self.nbytes];
        check_hip(
            unsafe {
                (api.hip_memcpy)(
                    host.as_mut_ptr() as *mut c_void,
                    self.ptr as *const c_void,
                    self.nbytes,
                    HIP_MEMCPY_DEVICE_TO_HOST,
                )
            },
            "hipMemcpy D2H",
        )?;
        Ok(host)
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

    /// Raw device pointer for kernel launches.
    pub fn as_device_ptr(&self) -> HipDeviceptr {
        self.ptr
    }
}

impl Drop for HipBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            if let Some(api) = ffi::hip_api() {
                let _ = set_device(self.device_idx);
                unsafe { (api.hip_free)(self.ptr) };
            }
        }
    }
}
