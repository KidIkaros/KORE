//! Runtime-loaded HIP and hiprtc function pointers via dlopen.
//!
//! This avoids pinning to a specific ROCm version — works with any ROCm install
//! that provides `libamdhip64.so` and `libhiprtc.so`.

use std::ffi::{c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

use libloading::Library;

use super::context::RocmError;

// ---------------------------------------------------------------------------
// HIP status codes
// ---------------------------------------------------------------------------

pub type HipErrorT = c_int;
pub const HIP_SUCCESS: HipErrorT = 0;

// hipMemcpyKind enum values
pub const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const HIP_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;

// hiprtc result codes
pub type HiprtcResult = c_int;
pub const HIPRTC_SUCCESS: HiprtcResult = 0;

// Opaque handles
pub type HipModule = *mut c_void;
pub type HipFunction = *mut c_void;
pub type HipDeviceptr = *mut c_void;
pub type HiprtcProgram = *mut c_void;
pub type HipStream = *mut c_void;

// ---------------------------------------------------------------------------
// HIP runtime API function signatures
// ---------------------------------------------------------------------------

type FnHipInit = unsafe extern "C" fn(c_uint) -> HipErrorT;
type FnHipGetDeviceCount = unsafe extern "C" fn(*mut c_int) -> HipErrorT;
type FnHipSetDevice = unsafe extern "C" fn(c_int) -> HipErrorT;
type FnHipMalloc = unsafe extern "C" fn(*mut HipDeviceptr, usize) -> HipErrorT;
type FnHipFree = unsafe extern "C" fn(HipDeviceptr) -> HipErrorT;
type FnHipMemcpy = unsafe extern "C" fn(
    *mut c_void, *const c_void, usize, c_int,
) -> HipErrorT;
type FnHipMemset = unsafe extern "C" fn(
    HipDeviceptr, c_int, usize,
) -> HipErrorT;
type FnHipDeviceSynchronize = unsafe extern "C" fn() -> HipErrorT;
type FnHipModuleLoadData = unsafe extern "C" fn(
    *mut HipModule, *const c_void,
) -> HipErrorT;
type FnHipModuleGetFunction = unsafe extern "C" fn(
    *mut HipFunction, HipModule, *const c_char,
) -> HipErrorT;
type FnHipModuleLaunchKernel = unsafe extern "C" fn(
    HipFunction,
    c_uint, c_uint, c_uint,   // grid x, y, z
    c_uint, c_uint, c_uint,   // block x, y, z
    c_uint,                    // shared mem bytes
    HipStream,                 // stream (null = default)
    *mut *mut c_void,          // kernel params
    *mut *mut c_void,          // extra (null)
) -> HipErrorT;
type FnHipModuleUnload = unsafe extern "C" fn(HipModule) -> HipErrorT;

// ---------------------------------------------------------------------------
// hiprtc API function signatures
// ---------------------------------------------------------------------------

type FnHiprtcCreateProgram = unsafe extern "C" fn(
    *mut HiprtcProgram,
    *const c_char,     // source
    *const c_char,     // name
    c_int,             // numHeaders
    *const *const c_char,  // headers
    *const *const c_char,  // includeNames
) -> HiprtcResult;
type FnHiprtcCompileProgram = unsafe extern "C" fn(
    HiprtcProgram, c_int, *const *const c_char,
) -> HiprtcResult;
type FnHiprtcGetCodeSize = unsafe extern "C" fn(
    HiprtcProgram, *mut usize,
) -> HiprtcResult;
type FnHiprtcGetCode = unsafe extern "C" fn(
    HiprtcProgram, *mut c_char,
) -> HiprtcResult;
type FnHiprtcGetProgramLogSize = unsafe extern "C" fn(
    HiprtcProgram, *mut usize,
) -> HiprtcResult;
type FnHiprtcGetProgramLog = unsafe extern "C" fn(
    HiprtcProgram, *mut c_char,
) -> HiprtcResult;
type FnHiprtcDestroyProgram = unsafe extern "C" fn(
    *mut HiprtcProgram,
) -> HiprtcResult;

// ---------------------------------------------------------------------------
// Loaded API structs
// ---------------------------------------------------------------------------

pub struct HipApi {
    _lib: Library,
    pub hip_init: FnHipInit,
    pub hip_get_device_count: FnHipGetDeviceCount,
    pub hip_set_device: FnHipSetDevice,
    pub hip_malloc: FnHipMalloc,
    pub hip_free: FnHipFree,
    pub hip_memcpy: FnHipMemcpy,
    pub hip_memset: FnHipMemset,
    pub hip_device_synchronize: FnHipDeviceSynchronize,
    pub hip_module_load_data: FnHipModuleLoadData,
    pub hip_module_get_function: FnHipModuleGetFunction,
    pub hip_module_launch_kernel: FnHipModuleLaunchKernel,
    pub hip_module_unload: FnHipModuleUnload,
}

pub struct HiprtcApi {
    _lib: Library,
    pub hiprtc_create_program: FnHiprtcCreateProgram,
    pub hiprtc_compile_program: FnHiprtcCompileProgram,
    pub hiprtc_get_code_size: FnHiprtcGetCodeSize,
    pub hiprtc_get_code: FnHiprtcGetCode,
    pub hiprtc_get_program_log_size: FnHiprtcGetProgramLogSize,
    pub hiprtc_get_program_log: FnHiprtcGetProgramLog,
    pub hiprtc_destroy_program: FnHiprtcDestroyProgram,
}

// Safety: The loaded function pointers are process-global and thread-safe
// (HIP runtime is internally synchronized).
unsafe impl Send for HipApi {}
unsafe impl Sync for HipApi {}
unsafe impl Send for HiprtcApi {}
unsafe impl Sync for HiprtcApi {}

// ---------------------------------------------------------------------------
// Library loading
// ---------------------------------------------------------------------------

static HIP_API: OnceLock<Option<HipApi>> = OnceLock::new();
static HIPRTC_API: OnceLock<Option<HiprtcApi>> = OnceLock::new();

impl HipApi {
    fn try_load() -> Option<Self> {
        let lib = unsafe { Library::new("libamdhip64.so") }.ok()?;
        unsafe {
            let api = HipApi {
                hip_init: *lib.get::<FnHipInit>(b"hipInit\0").ok()?,
                hip_get_device_count: *lib.get::<FnHipGetDeviceCount>(b"hipGetDeviceCount\0").ok()?,
                hip_set_device: *lib.get::<FnHipSetDevice>(b"hipSetDevice\0").ok()?,
                hip_malloc: *lib.get::<FnHipMalloc>(b"hipMalloc\0").ok()?,
                hip_free: *lib.get::<FnHipFree>(b"hipFree\0").ok()?,
                hip_memcpy: *lib.get::<FnHipMemcpy>(b"hipMemcpy\0").ok()?,
                hip_memset: *lib.get::<FnHipMemset>(b"hipMemset\0").ok()?,
                hip_device_synchronize: *lib.get::<FnHipDeviceSynchronize>(b"hipDeviceSynchronize\0").ok()?,
                hip_module_load_data: *lib.get::<FnHipModuleLoadData>(b"hipModuleLoadData\0").ok()?,
                hip_module_get_function: *lib.get::<FnHipModuleGetFunction>(b"hipModuleGetFunction\0").ok()?,
                hip_module_launch_kernel: *lib.get::<FnHipModuleLaunchKernel>(b"hipModuleLaunchKernel\0").ok()?,
                hip_module_unload: *lib.get::<FnHipModuleUnload>(b"hipModuleUnload\0").ok()?,
                _lib: lib,
            };
            Some(api)
        }
    }
}

impl HiprtcApi {
    fn try_load() -> Option<Self> {
        let lib = unsafe { Library::new("libhiprtc.so") }.ok()?;
        unsafe {
            let api = HiprtcApi {
                hiprtc_create_program: *lib.get::<FnHiprtcCreateProgram>(b"hiprtcCreateProgram\0").ok()?,
                hiprtc_compile_program: *lib.get::<FnHiprtcCompileProgram>(b"hiprtcCompileProgram\0").ok()?,
                hiprtc_get_code_size: *lib.get::<FnHiprtcGetCodeSize>(b"hiprtcGetCodeSize\0").ok()?,
                hiprtc_get_code: *lib.get::<FnHiprtcGetCode>(b"hiprtcGetCode\0").ok()?,
                hiprtc_get_program_log_size: *lib.get::<FnHiprtcGetProgramLogSize>(b"hiprtcGetProgramLogSize\0").ok()?,
                hiprtc_get_program_log: *lib.get::<FnHiprtcGetProgramLog>(b"hiprtcGetProgramLog\0").ok()?,
                hiprtc_destroy_program: *lib.get::<FnHiprtcDestroyProgram>(b"hiprtcDestroyProgram\0").ok()?,
                _lib: lib,
            };
            Some(api)
        }
    }
}

/// Get the runtime-loaded HIP API. Returns None if libamdhip64.so not found.
pub fn hip_api() -> Option<&'static HipApi> {
    HIP_API.get_or_init(|| HipApi::try_load()).as_ref()
}

/// Get the runtime-loaded hiprtc API. Returns None if libhiprtc.so not found.
pub fn hiprtc_api() -> Option<&'static HiprtcApi> {
    HIPRTC_API.get_or_init(|| HiprtcApi::try_load()).as_ref()
}

// ---------------------------------------------------------------------------
// Error checking helpers
// ---------------------------------------------------------------------------

/// Check a HIP API return code.
pub fn check_hip(code: HipErrorT, context: &str) -> Result<(), RocmError> {
    if code == HIP_SUCCESS {
        Ok(())
    } else {
        Err(RocmError::HipError { code, context: context.to_string() })
    }
}

/// Check a hiprtc return code.
pub fn check_hiprtc(code: HiprtcResult, context: &str) -> Result<(), RocmError> {
    if code == HIPRTC_SUCCESS {
        Ok(())
    } else {
        Err(RocmError::HiprtcError { code, context: context.to_string() })
    }
}
