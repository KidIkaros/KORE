//! HIP device context management.
//!
//! Provides lazy-initialized device state per GPU index, mirroring `cuda/context.rs`.
//! All HIP calls go through runtime-loaded function pointers from `ffi.rs`.

use std::sync::OnceLock;

use parking_lot::Mutex;

use super::ffi::{self, check_hip, HIP_SUCCESS};

/// Tracks which HIP device indices have been initialized.
static INIT_DEVICES: OnceLock<Mutex<Vec<bool>>> = OnceLock::new();

fn init_devices() -> &'static Mutex<Vec<bool>> {
    INIT_DEVICES.get_or_init(|| Mutex::new(Vec::new()))
}

/// Initialize the HIP runtime and select the given device.
/// No-op if already initialized for this device index.
pub fn init_device(device_idx: usize) -> Result<(), RocmError> {
    let api = ffi::hip_api().ok_or(RocmError::NotAvailable)?;

    let mut devs = init_devices().lock();

    // Ensure vec is big enough
    if devs.len() <= device_idx {
        devs.resize(device_idx + 1, false);
    }

    if !devs[device_idx] {
        check_hip(unsafe { (api.hip_init)(0) }, "hipInit")?;
        check_hip(
            unsafe { (api.hip_set_device)(device_idx as i32) },
            "hipSetDevice",
        )?;
        devs[device_idx] = true;
    }

    Ok(())
}

/// Set the current HIP device (must be already initialized).
pub fn set_device(device_idx: usize) -> Result<(), RocmError> {
    let api = ffi::hip_api().ok_or(RocmError::NotAvailable)?;
    check_hip(
        unsafe { (api.hip_set_device)(device_idx as i32) },
        "hipSetDevice",
    )
}

/// Check if ROCm/HIP is available (libamdhip64.so loadable + at least 1 GPU).
pub fn is_rocm_available() -> bool {
    let api = match ffi::hip_api() {
        Some(a) => a,
        None => return false,
    };
    // Try to init + count devices
    if unsafe { (api.hip_init)(0) } != HIP_SUCCESS {
        return false;
    }
    let mut count: i32 = 0;
    if unsafe { (api.hip_get_device_count)(&mut count) } != HIP_SUCCESS {
        return false;
    }
    count > 0
}

/// Number of available HIP devices.
pub fn device_count() -> usize {
    let api = match ffi::hip_api() {
        Some(a) => a,
        None => return 0,
    };
    if unsafe { (api.hip_init)(0) } != HIP_SUCCESS {
        return 0;
    }
    let mut count: i32 = 0;
    if unsafe { (api.hip_get_device_count)(&mut count) } != HIP_SUCCESS {
        return 0;
    }
    count.max(0) as usize
}

/// Synchronize the current HIP device (wait for all pending operations).
pub fn device_synchronize() -> Result<(), RocmError> {
    let api = ffi::hip_api().ok_or(RocmError::NotAvailable)?;
    check_hip(unsafe { (api.hip_device_synchronize)() }, "hipDeviceSynchronize")
}

/// ROCm-specific errors.
#[derive(Debug, thiserror::Error)]
pub enum RocmError {
    #[error("ROCm/HIP not available (libamdhip64.so not found)")]
    NotAvailable,

    #[error("HIP error {code} in {context}")]
    HipError { code: i32, context: String },

    #[error("hiprtc error {code} in {context}")]
    HiprtcError { code: i32, context: String },

    #[error("hiprtc compilation failed for '{module}': {log}")]
    CompileError { module: String, log: String },

    #[error("Function '{func}' not found in module '{module}'")]
    FuncNotFound { module: String, func: String },

    #[error("ROCm kernel launch failed: {0}")]
    LaunchError(String),

    #[error("ROCm memory error: {0}")]
    MemoryError(String),
}
