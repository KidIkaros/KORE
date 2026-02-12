//! hiprtc kernel compilation and HIP kernel launch utilities.
//!
//! Mirrors `cuda/launch.rs` — compiles kernel source via hiprtc at runtime,
//! caches compiled modules, and launches kernels via hipModuleLaunchKernel.

use std::collections::HashMap;
use std::ffi::{c_char, c_uint, c_void, CString};
use std::sync::OnceLock;

use parking_lot::Mutex;

use super::context::{set_device, device_synchronize, RocmError};
use super::ffi::{self, check_hip, check_hiprtc, HipFunction, HipModule};

/// Wrapper around `HipModule` (`*mut c_void`) to make it Send+Sync.
/// HIP modules are process-global and thread-safe once loaded.
#[derive(Clone, Copy)]
struct SendModule(HipModule);
unsafe impl Send for SendModule {}
unsafe impl Sync for SendModule {}

/// Cache of compiled HIP modules. Key: (device_idx, module_name).
static MODULE_CACHE: OnceLock<Mutex<HashMap<(usize, String), SendModule>>> = OnceLock::new();

fn module_cache() -> &'static Mutex<HashMap<(usize, String), SendModule>> {
    MODULE_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// HIP preprocessor shim prepended to CUDA kernel source before hiprtc compilation.
/// hiprtc already understands __global__, __device__, etc., but we define
/// the explicit attribute forms as a safety net for older ROCm versions.
const HIP_SHIM: &str = r#"
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <hip/hip_runtime.h>
"#;

/// Compile kernel source via hiprtc and load the module on the given device.
/// Returns the compiled module handle. Caches by (device_idx, module_name).
pub fn ensure_module(
    device_idx: usize,
    module_name: &str,
    kernel_source: &str,
) -> Result<HipModule, RocmError> {
    let key = (device_idx, module_name.to_string());
    {
        let cache = module_cache().lock();
        if let Some(&SendModule(module)) = cache.get(&key) {
            return Ok(module);
        }
    }

    let hiprtc = ffi::hiprtc_api().ok_or(RocmError::NotAvailable)?;
    let hip = ffi::hip_api().ok_or(RocmError::NotAvailable)?;

    set_device(device_idx)?;

    // Prepend HIP shim to source
    let full_source = format!("{}\n{}", HIP_SHIM, kernel_source);
    let c_source = CString::new(full_source)
        .map_err(|_| RocmError::CompileError {
            module: module_name.to_string(),
            log: "source contains null byte".to_string(),
        })?;
    let c_name = CString::new(module_name)
        .map_err(|_| RocmError::CompileError {
            module: module_name.to_string(),
            log: "module name contains null byte".to_string(),
        })?;

    // Create hiprtc program
    let mut prog: ffi::HiprtcProgram = std::ptr::null_mut();
    check_hiprtc(
        unsafe {
            (hiprtc.hiprtc_create_program)(
                &mut prog,
                c_source.as_ptr(),
                c_name.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        },
        "hiprtcCreateProgram",
    )?;

    // Compile with default options
    let compile_result = unsafe {
        (hiprtc.hiprtc_compile_program)(prog, 0, std::ptr::null())
    };

    if compile_result != ffi::HIPRTC_SUCCESS {
        // Get compile log for error reporting
        let mut log_size: usize = 0;
        let _ = unsafe { (hiprtc.hiprtc_get_program_log_size)(prog, &mut log_size) };
        let mut log_buf = vec![0u8; log_size];
        let _ = unsafe {
            (hiprtc.hiprtc_get_program_log)(prog, log_buf.as_mut_ptr() as *mut c_char)
        };
        let log = String::from_utf8_lossy(&log_buf).to_string();
        unsafe { (hiprtc.hiprtc_destroy_program)(&mut prog) };
        return Err(RocmError::CompileError {
            module: module_name.to_string(),
            log,
        });
    }

    // Get compiled code
    let mut code_size: usize = 0;
    check_hiprtc(
        unsafe { (hiprtc.hiprtc_get_code_size)(prog, &mut code_size) },
        "hiprtcGetCodeSize",
    )?;
    let mut code = vec![0u8; code_size];
    check_hiprtc(
        unsafe { (hiprtc.hiprtc_get_code)(prog, code.as_mut_ptr() as *mut c_char) },
        "hiprtcGetCode",
    )?;
    unsafe { (hiprtc.hiprtc_destroy_program)(&mut prog) };

    // Load compiled code as a HIP module
    let mut module: HipModule = std::ptr::null_mut();
    check_hip(
        unsafe { (hip.hip_module_load_data)(&mut module, code.as_ptr() as *const c_void) },
        "hipModuleLoadData",
    )?;

    module_cache().lock().insert(key, SendModule(module));
    Ok(module)
}

/// Get a kernel function handle from a compiled module.
pub fn get_function(
    module: HipModule,
    func_name: &str,
    module_name: &str,
) -> Result<HipFunction, RocmError> {
    let hip = ffi::hip_api().ok_or(RocmError::NotAvailable)?;
    let c_func = CString::new(func_name).map_err(|_| RocmError::FuncNotFound {
        module: module_name.to_string(),
        func: func_name.to_string(),
    })?;

    let mut func: HipFunction = std::ptr::null_mut();
    check_hip(
        unsafe { (hip.hip_module_get_function)(&mut func, module, c_func.as_ptr()) },
        &format!("hipModuleGetFunction({})", func_name),
    )?;
    Ok(func)
}

/// Ensure module is loaded, then get the named function.
pub fn get_or_load_func(
    device_idx: usize,
    module_name: &str,
    func_name: &str,
    kernel_source: &str,
) -> Result<HipFunction, RocmError> {
    let module = ensure_module(device_idx, module_name, kernel_source)?;
    get_function(module, func_name, module_name)
}

/// Launch configuration for HIP kernels.
pub struct HipLaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

/// Launch a HIP kernel with the given parameters.
///
/// # Safety
/// The caller must ensure `params` contains valid pointers matching the kernel signature,
/// and that all GPU buffers are valid on the current device.
pub unsafe fn launch_kernel(
    device_idx: usize,
    func: HipFunction,
    config: &HipLaunchConfig,
    params: &mut [*mut c_void],
) -> Result<(), RocmError> {
    let hip = ffi::hip_api().ok_or(RocmError::NotAvailable)?;
    set_device(device_idx)?;

    check_hip(
        (hip.hip_module_launch_kernel)(
            func,
            config.grid_dim.0 as c_uint,
            config.grid_dim.1 as c_uint,
            config.grid_dim.2 as c_uint,
            config.block_dim.0 as c_uint,
            config.block_dim.1 as c_uint,
            config.block_dim.2 as c_uint,
            config.shared_mem_bytes as c_uint,
            std::ptr::null_mut(), // default stream
            params.as_mut_ptr(),
            std::ptr::null_mut(), // extra
        ),
        "hipModuleLaunchKernel",
    )?;

    // Synchronize to catch errors eagerly (can be relaxed later for perf)
    device_synchronize()?;
    Ok(())
}
