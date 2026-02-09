//! WASM-specific JS/TS bindings via wasm-bindgen.
//!
//! This module is only compiled when targeting `wasm32`.
//! Build with: `wasm-pack build --target web crates/kore-edge`
//!
//! ## JavaScript API
//!
//! ```js
//! import init, { KoreSession } from '@kore/edge';
//!
//! await init();
//! const response = await fetch('model.koref');
//! const bytes = new Uint8Array(await response.arrayBuffer());
//! const session = KoreSession.fromBytes(bytes);
//!
//! const logits = session.forward(new Uint32Array([1, 2, 3]));
//! const tokens = session.generate(new Uint32Array([1, 2, 3]), 32);
//! session.free();
//! ```

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// WASM session wrapper exposed to JavaScript.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct KoreSession {
    inner: crate::runtime::Session,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl KoreSession {
    /// Load a .koref model from raw bytes (Uint8Array).
    #[wasm_bindgen(js_name = "fromBytes")]
    pub fn from_bytes(data: &[u8]) -> Result<KoreSession, JsValue> {
        let model = crate::format::KorefModel::from_bytes(data)
            .map_err(|e| JsValue::from_str(&format!("{}", e)))?;
        Ok(KoreSession {
            inner: crate::runtime::Session::new(model),
        })
    }

    /// Run a forward pass, returning logits for the last token.
    pub fn forward(&mut self, input_ids: &[u32]) -> Vec<f32> {
        self.inner.forward(input_ids)
    }

    /// Generate tokens autoregressively.
    pub fn generate(&mut self, input_ids: &[u32], max_new_tokens: usize) -> Vec<u32> {
        self.inner.generate(input_ids, max_new_tokens)
    }

    /// Get model info string.
    pub fn info(&self) -> String {
        self.inner.info()
    }

    /// Reset the session (clear KV cache).
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

// Stub for non-wasm targets so the module can be referenced in lib.rs
#[cfg(not(target_arch = "wasm32"))]
pub struct KoreSession;
