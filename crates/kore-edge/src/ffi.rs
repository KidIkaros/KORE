//! C FFI for native mobile integration (Android/iOS).
//!
//! Provides a C-compatible API for Swift, Kotlin/JNI, C#, and Dart.
//! Enable with the `ffi` feature flag.
//!
//! ```c
//! #include "kore_edge.h"
//! KoreSession* session = kore_edge_load("model.koref");
//! KoreOutput* out = kore_edge_generate(session, ids, len, 32);
//! kore_edge_free_output(out);
//! kore_edge_free(session);
//! ```

use crate::format::KorefModel;
use crate::runtime::Session;
use std::ffi::CStr;
use std::os::raw::c_char;

/// Opaque session handle for C API.
pub struct KoreSession {
    inner: Session,
}

/// Output from inference — token IDs or logits.
#[repr(C)]
pub struct KoreOutput {
    pub tokens: *mut u32,
    pub token_count: usize,
    pub logits: *mut f32,
    pub logit_count: usize,
}

/// Load a .koref model from a file path. Returns null on failure.
#[no_mangle]
pub unsafe extern "C" fn kore_edge_load(path: *const c_char) -> *mut KoreSession {
    if path.is_null() {
        return std::ptr::null_mut();
    }

    let c_str = match CStr::from_ptr(path).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    let data = match std::fs::read(c_str) {
        Ok(d) => d,
        Err(_) => return std::ptr::null_mut(),
    };

    let model = match KorefModel::from_bytes(&data) {
        Ok(m) => m,
        Err(_) => return std::ptr::null_mut(),
    };

    let session = Session::new(model);
    Box::into_raw(Box::new(KoreSession { inner: session }))
}

/// Load a .koref model from raw bytes. Returns null on failure.
#[no_mangle]
pub unsafe extern "C" fn kore_edge_load_bytes(data: *const u8, len: usize) -> *mut KoreSession {
    if data.is_null() || len == 0 {
        return std::ptr::null_mut();
    }

    let bytes = std::slice::from_raw_parts(data, len);
    let model = match KorefModel::from_bytes(bytes) {
        Ok(m) => m,
        Err(_) => return std::ptr::null_mut(),
    };

    let session = Session::new(model);
    Box::into_raw(Box::new(KoreSession { inner: session }))
}

/// Run a forward pass, returning logits for the last token.
#[no_mangle]
pub unsafe extern "C" fn kore_edge_run(
    session: *mut KoreSession,
    input_ids: *const u32,
    input_len: usize,
) -> *mut KoreOutput {
    if session.is_null() || input_ids.is_null() || input_len == 0 {
        return std::ptr::null_mut();
    }

    let session = &mut (*session).inner;
    let ids = std::slice::from_raw_parts(input_ids, input_len);
    let logits = session.forward(ids);

    let logit_count = logits.len();
    let logits_ptr = Box::into_raw(logits.into_boxed_slice()) as *mut f32;

    Box::into_raw(Box::new(KoreOutput {
        tokens: std::ptr::null_mut(),
        token_count: 0,
        logits: logits_ptr,
        logit_count,
    }))
}

/// Generate tokens autoregressively.
#[no_mangle]
pub unsafe extern "C" fn kore_edge_generate(
    session: *mut KoreSession,
    input_ids: *const u32,
    input_len: usize,
    max_new_tokens: usize,
) -> *mut KoreOutput {
    if session.is_null() || input_ids.is_null() || input_len == 0 {
        return std::ptr::null_mut();
    }

    let session = &mut (*session).inner;
    let ids = std::slice::from_raw_parts(input_ids, input_len);
    let output = session.generate(ids, max_new_tokens);

    let token_count = output.len();
    let tokens_ptr = Box::into_raw(output.into_boxed_slice()) as *mut u32;

    Box::into_raw(Box::new(KoreOutput {
        tokens: tokens_ptr,
        token_count,
        logits: std::ptr::null_mut(),
        logit_count: 0,
    }))
}

/// Reset the session (clear KV cache).
#[no_mangle]
pub unsafe extern "C" fn kore_edge_reset(session: *mut KoreSession) {
    if !session.is_null() {
        (*session).inner.reset();
    }
}

/// Free a KoreOutput.
#[no_mangle]
pub unsafe extern "C" fn kore_edge_free_output(output: *mut KoreOutput) {
    if !output.is_null() {
        let out = Box::from_raw(output);
        if !out.tokens.is_null() {
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(out.tokens, out.token_count) as *mut [u32]);
        }
        if !out.logits.is_null() {
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(out.logits, out.logit_count) as *mut [f32]);
        }
    }
}

/// Free a KoreSession.
#[no_mangle]
pub unsafe extern "C" fn kore_edge_free(session: *mut KoreSession) {
    if !session.is_null() {
        let _ = Box::from_raw(session);
    }
}
