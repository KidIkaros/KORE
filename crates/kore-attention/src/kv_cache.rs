//! KV-cache for autoregressive generation.
//!
//! Stores key and value tensors from previous tokens to avoid
//! recomputation during sequential generation. Supports full-precision (`F32`),
//! symmetric 8-bit (`Q8_0`), and asymmetric rotated `K8V4` (`INT8` Keys + Hadamard-rotated
//! `INT4` Values) storage modes, plus $O(1)$ speculative rollback via `truncate`.

use kore_core::{KoreError, Tensor};

/// Compression mode for the KV-cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvCacheQuantMode {
    /// Uncompressed 32-bit floating point (`f32`).
    #[default]
    F32,
    /// Symmetric per-token 8-bit integer (`INT8`) for both Keys and Values (~4x compression).
    Q8_0,
    /// Asymmetric 8-bit Keys (`INT8`) + Hadamard-rotated 4-bit Values (`INT4`, ~5.3x compression).
    K8V4,
}

/// Key-Value cache for a single attention layer.
///
/// Stores accumulated K and V tensors, growing as new tokens are generated.
pub struct KvCache {
    /// Cached keys on CPU (`F32` mode): `[current_seq_len, d_k]`
    keys: Option<Vec<f32>>,
    /// Cached values on CPU (`F32` mode): `[current_seq_len, d_v]`
    values: Option<Vec<f32>>,
    /// Quantized `INT8` keys (`Q8_0` and `K8V4` modes): `[current_seq_len, d_k]`
    q_keys: Vec<i8>,
    /// Per-token key scales (`Q8_0` and `K8V4` modes): `[current_seq_len]`
    k_scales: Vec<f32>,
    /// Quantized `INT8` values (`Q8_0` mode): `[current_seq_len, d_v]`
    q8_values: Vec<i8>,
    /// Packed `INT4` Hadamard-rotated values (`K8V4` mode): `[current_seq_len, ceil(d_v / 2)]`
    q4_values: Vec<u8>,
    /// Per-token value scales (`Q8_0` and `K8V4` modes): `[current_seq_len]`
    v_scales: Vec<f32>,
    /// Pre-allocated device key capacity buffer: `[max_seq_len, d_k]`
    gpu_keys: Option<Tensor>,
    /// Pre-allocated device value capacity buffer: `[max_seq_len, d_v]`
    gpu_values: Option<Tensor>,
    /// Key dimension
    d_k: usize,
    /// Value dimension
    d_v: usize,
    /// Current sequence length (number of cached tokens)
    seq_len: usize,
    /// Maximum sequence length (pre-allocated capacity)
    max_seq_len: usize,
    /// Active quantization mode
    quant_mode: KvCacheQuantMode,
}

impl KvCache {
    /// Create a new empty KV cache in `F32` mode.
    pub fn new(d_k: usize, d_v: usize, max_seq_len: usize) -> Self {
        Self::with_quant_mode(d_k, d_v, max_seq_len, KvCacheQuantMode::F32)
    }

    /// Create a new empty KV cache with the specified quantization mode (`F32`, `Q8_0`, or `K8V4`).
    pub fn with_quant_mode(
        d_k: usize,
        d_v: usize,
        max_seq_len: usize,
        quant_mode: KvCacheQuantMode,
    ) -> Self {
        let (keys, values) = if quant_mode == KvCacheQuantMode::F32 {
            (
                Some(Vec::with_capacity(max_seq_len * d_k)),
                Some(Vec::with_capacity(max_seq_len * d_v)),
            )
        } else {
            (None, None)
        };
        Self {
            keys,
            values,
            q_keys: Vec::new(),
            k_scales: Vec::new(),
            q8_values: Vec::new(),
            q4_values: Vec::new(),
            v_scales: Vec::new(),
            gpu_keys: None,
            gpu_values: None,
            d_k,
            d_v,
            seq_len: 0,
            max_seq_len,
            quant_mode,
        }
    }

    /// Active KV-cache quantization mode.
    pub fn quant_mode(&self) -> KvCacheQuantMode {
        self.quant_mode
    }

    /// Append new key and value vectors to the cache.
    ///
    /// `new_keys`: `[num_new_tokens, d_k]`
    /// `new_values`: `[num_new_tokens, d_v]`
    ///
    /// Returns the full cached `(keys, values)` as tensors on the same device as `new_keys`.
    pub fn update(
        &mut self,
        new_keys: &Tensor,
        new_values: &Tensor,
    ) -> Result<(Tensor, Tensor), KoreError> {
        let new_tokens = new_keys.shape().dims()[0];

        if self.seq_len + new_tokens > self.max_seq_len {
            return Err(KoreError::StorageError(format!(
                "KV cache overflow: {} + {} > {}",
                self.seq_len, new_tokens, self.max_seq_len
            )));
        }

        let dev = new_keys.device();
        if !dev.is_cpu() && self.quant_mode == KvCacheQuantMode::F32 {
            if let Some(backend) = kore_core::device::get_backend(dev) {
                if self.gpu_keys.is_none() {
                    self.gpu_keys = Some(
                        Tensor::zeros(&[self.max_seq_len, self.d_k], kore_core::DType::F32)
                            .to(dev)?,
                    );
                }
                if self.gpu_values.is_none() {
                    self.gpu_values = Some(
                        Tensor::zeros(&[self.max_seq_len, self.d_v], kore_core::DType::F32)
                            .to(dev)?,
                    );
                }
                let gk = self.gpu_keys.as_ref().unwrap();
                let gv = self.gpu_values.as_ref().unwrap();
                if let (Ok(k_view), Ok(v_view)) = (
                    backend.kv_cache_append(gk, new_keys, self.seq_len),
                    backend.kv_cache_append(gv, new_values, self.seq_len),
                ) {
                    self.seq_len += new_tokens;
                    return Ok((k_view, v_view));
                }
            }

            // Fallback for GPU backends that do not implement kv_cache_append natively
            let nk_cpu = new_keys.cpu()?;
            let nv_cpu = new_values.cpu()?;
            let (k_cpu, v_cpu) = self.update(&nk_cpu, &nv_cpu)?;
            return Ok((k_cpu.to(dev)?, v_cpu.to(dev)?));
        }

        if !dev.is_cpu() {
            let nk_cpu = new_keys.cpu()?;
            let nv_cpu = new_values.cpu()?;
            let (k_cpu, v_cpu) = self.update(&nk_cpu, &nv_cpu)?;
            return Ok((k_cpu.to(dev)?, v_cpu.to(dev)?));
        }

        let nk = new_keys.contiguous();
        let nv = new_values.contiguous();

        let nk_data = nk
            .as_f32_slice()
            .ok_or_else(|| KoreError::UnsupportedDType(new_keys.dtype()))?;
        let nv_data = nv
            .as_f32_slice()
            .ok_or_else(|| KoreError::UnsupportedDType(new_values.dtype()))?;

        match self.quant_mode {
            KvCacheQuantMode::F32 => {
                let keys = self.keys.as_mut().unwrap();
                let values = self.values.as_mut().unwrap();
                keys.extend_from_slice(nk_data);
                values.extend_from_slice(nv_data);
                self.seq_len += new_tokens;

                let k_tensor = Tensor::from_f32(keys, &[self.seq_len, self.d_k]);
                let v_tensor = Tensor::from_f32(values, &[self.seq_len, self.d_v]);
                Ok((k_tensor, v_tensor))
            }
            KvCacheQuantMode::Q8_0 => {
                for row in nk_data.chunks_exact(self.d_k) {
                    let absmax = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    let scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
                    let inv = 1.0 / scale;
                    self.k_scales.push(scale);
                    for &v in row {
                        self.q_keys
                            .push((v * inv).round().clamp(-127.0, 127.0) as i8);
                    }
                }
                for row in nv_data.chunks_exact(self.d_v) {
                    let absmax = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    let scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
                    let inv = 1.0 / scale;
                    self.v_scales.push(scale);
                    for &v in row {
                        self.q8_values
                            .push((v * inv).round().clamp(-127.0, 127.0) as i8);
                    }
                }
                self.seq_len += new_tokens;
                self.dequantize_kv()
            }
            KvCacheQuantMode::K8V4 => {
                for row in nk_data.chunks_exact(self.d_k) {
                    let absmax = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    let scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
                    let inv = 1.0 / scale;
                    self.k_scales.push(scale);
                    for &v in row {
                        self.q_keys
                            .push((v * inv).round().clamp(-127.0, 127.0) as i8);
                    }
                }
                // Rotate V with normalized Fast Walsh-Hadamard Transform before 4-bit quantization
                let nv_rot = nv.hadamard()?;
                let nv_rot_slice = nv_rot.as_f32_slice().unwrap();
                let row_bytes = self.d_v.div_ceil(2);
                for row in nv_rot_slice.chunks_exact(self.d_v) {
                    let absmax = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    let scale = if absmax > 0.0 { absmax / 7.0 } else { 1.0 };
                    let inv = 1.0 / scale;
                    self.v_scales.push(scale);
                    for b_idx in 0..row_bytes {
                        let i0 = b_idx * 2;
                        let q0 = (row[i0] * inv).round().clamp(-8.0, 7.0) as i8;
                        let q1 = if i0 + 1 < self.d_v {
                            (row[i0 + 1] * inv).round().clamp(-8.0, 7.0) as i8
                        } else {
                            0
                        };
                        let packed = ((q0 + 8) as u8 & 0x0F) | (((q1 + 8) as u8 & 0x0F) << 4);
                        self.q4_values.push(packed);
                    }
                }
                self.seq_len += new_tokens;
                self.dequantize_kv()
            }
        }
    }

    fn dequantize_kv(&self) -> Result<(Tensor, Tensor), KoreError> {
        let mut k_out = Vec::with_capacity(self.seq_len * self.d_k);
        for (t, row) in self.q_keys.chunks_exact(self.d_k).enumerate() {
            let scale = self.k_scales[t];
            for &q in row {
                k_out.push(q as f32 * scale);
            }
        }
        let k_tensor = Tensor::from_f32(&k_out, &[self.seq_len, self.d_k]);

        let v_tensor = match self.quant_mode {
            KvCacheQuantMode::Q8_0 => {
                let mut v_out = Vec::with_capacity(self.seq_len * self.d_v);
                for (t, row) in self.q8_values.chunks_exact(self.d_v).enumerate() {
                    let scale = self.v_scales[t];
                    for &q in row {
                        v_out.push(q as f32 * scale);
                    }
                }
                Tensor::from_f32(&v_out, &[self.seq_len, self.d_v])
            }
            KvCacheQuantMode::K8V4 => {
                let row_bytes = self.d_v.div_ceil(2);
                let mut v_rot = Vec::with_capacity(self.seq_len * self.d_v);
                for (t, packed_row) in self.q4_values.chunks_exact(row_bytes).enumerate() {
                    let scale = self.v_scales[t];
                    for (b_idx, &byte) in packed_row.iter().enumerate() {
                        let q0 = (byte & 0x0F) as i8 - 8;
                        v_rot.push(q0 as f32 * scale);
                        if b_idx * 2 + 1 < self.d_v {
                            let q1 = ((byte >> 4) & 0x0F) as i8 - 8;
                            v_rot.push(q1 as f32 * scale);
                        }
                    }
                }
                // Self-inverse FWHT restores original basis
                Tensor::from_f32(&v_rot, &[self.seq_len, self.d_v]).hadamard()?
            }
            KvCacheQuantMode::F32 => unreachable!(),
        };

        Ok((k_tensor, v_tensor))
    }

    /// Roll back the KV-cache to `new_seq_len` in $O(1)$ time (used by speculative decoding when a draft token is rejected).
    pub fn truncate(&mut self, new_seq_len: usize) {
        if new_seq_len >= self.seq_len {
            return;
        }
        self.seq_len = new_seq_len;
        if let Some(keys) = &mut self.keys {
            keys.truncate(new_seq_len * self.d_k);
        }
        if let Some(values) = &mut self.values {
            values.truncate(new_seq_len * self.d_v);
        }
        if self.quant_mode != KvCacheQuantMode::F32 {
            self.q_keys.truncate(new_seq_len * self.d_k);
            self.k_scales.truncate(new_seq_len);
            self.v_scales.truncate(new_seq_len);
            if self.quant_mode == KvCacheQuantMode::Q8_0 {
                self.q8_values.truncate(new_seq_len * self.d_v);
            } else {
                let row_bytes = self.d_v.div_ceil(2);
                self.q4_values.truncate(new_seq_len * row_bytes);
            }
        }
    }

    /// Memory usage in bytes of the currently cached tokens.
    pub fn memory_bytes(&self) -> usize {
        match self.quant_mode {
            KvCacheQuantMode::F32 => self.seq_len * (self.d_k + self.d_v) * 4,
            KvCacheQuantMode::Q8_0 => {
                self.q_keys.len()
                    + self.q8_values.len()
                    + (self.k_scales.len() + self.v_scales.len()) * 4
            }
            KvCacheQuantMode::K8V4 => {
                self.q_keys.len()
                    + self.q4_values.len()
                    + (self.k_scales.len() + self.v_scales.len()) * 4
            }
        }
    }

    /// Current number of cached tokens.
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Remaining capacity.
    pub fn remaining(&self) -> usize {
        self.max_seq_len - self.seq_len
    }

    /// Clear the cache for a new sequence.
    pub fn clear(&mut self) {
        self.truncate(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_basic() {
        let mut cache = KvCache::new(4, 4, 128);
        assert!(cache.is_empty());
        assert_eq!(cache.remaining(), 128);

        // Add 3 tokens
        let k = Tensor::from_f32(&[1.0; 12], &[3, 4]);
        let v = Tensor::from_f32(&[2.0; 12], &[3, 4]);
        let (ck, cv) = cache.update(&k, &v).unwrap();

        assert_eq!(cache.len(), 3);
        assert_eq!(ck.shape().dims(), &[3, 4]);
        assert_eq!(cv.shape().dims(), &[3, 4]);

        // Add 2 more tokens
        let k2 = Tensor::from_f32(&[3.0; 8], &[2, 4]);
        let v2 = Tensor::from_f32(&[4.0; 8], &[2, 4]);
        let (ck2, cv2) = cache.update(&k2, &v2).unwrap();

        assert_eq!(cache.len(), 5);
        assert_eq!(ck2.shape().dims(), &[5, 4]);
        assert_eq!(cv2.shape().dims(), &[5, 4]);

        // Verify data integrity
        let k_data = ck2.as_f32_slice().unwrap();
        assert_eq!(k_data[0], 1.0); // first token
        assert_eq!(k_data[12], 3.0); // fourth token
    }

    #[test]
    fn test_kv_cache_clear_and_truncate() {
        let mut cache = KvCache::new(4, 4, 128);
        let k = Tensor::from_f32(&[1.0; 16], &[4, 4]);
        let v = Tensor::from_f32(&[1.0; 16], &[4, 4]);
        cache.update(&k, &v).unwrap();

        assert_eq!(cache.len(), 4);
        cache.truncate(2);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.remaining(), 126);

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.remaining(), 128);
    }

    #[test]
    fn test_kv_cache_q8_and_k8v4_compression_and_accuracy() {
        let d = 64;
        let k_data: Vec<f32> = (0..(4 * d)).map(|i| (i as f32 * 0.13).sin()).collect();
        // Realistic activation distribution: pseudo-random channel values in [-0.5, 0.5] + an 8.0 outlier channel
        let v_data: Vec<f32> = (0..(4 * d))
            .map(|i| {
                ((i as f32 * 1.73 + (i * i) as f32 * 0.37).sin() * 0.5)
                    + if i == 5 { 8.0 } else { 0.0 }
            })
            .collect();
        let k = Tensor::from_f32(&k_data, &[4, d]);
        let v = Tensor::from_f32(&v_data, &[4, d]);

        let mut f32_cache = KvCache::new(d, d, 32);
        f32_cache.update(&k, &v).unwrap();

        let mut q8_cache = KvCache::with_quant_mode(d, d, 32, KvCacheQuantMode::Q8_0);
        let (ck8, cv8) = q8_cache.update(&k, &v).unwrap();
        assert!(q8_cache.memory_bytes() < f32_cache.memory_bytes() / 3);
        for (a, b) in k_data.iter().zip(ck8.as_f32_slice().unwrap().iter()) {
            assert!((a - b).abs() < 0.02, "Q8 K error too large: {a} vs {b}");
        }
        for (a, b) in v_data.iter().zip(cv8.as_f32_slice().unwrap().iter()) {
            assert!((a - b).abs() < 0.1, "Q8 V error too large: {a} vs {b}");
        }

        let mut k8v4_cache = KvCache::with_quant_mode(d, d, 32, KvCacheQuantMode::K8V4);
        let (_ck4, cv4) = k8v4_cache.update(&k, &v).unwrap();
        assert!(k8v4_cache.memory_bytes() < q8_cache.memory_bytes());
        // With an 8.0 outlier in V, Hadamard rotation spreads it across the 64-dim block (1.0 per bin),
        // cutting 4-bit quantization MSE dramatically vs unrotated 4-bit quantization.
        let cv4_slice = cv4.as_f32_slice().unwrap();
        let mut mse_rotated = 0.0f32;
        let mut mse_unrotated = 0.0f32;
        let row0_absmax = v_data[..d].iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let unrotated_scale = row0_absmax / 7.0f32;
        for (i, (&a, &b)) in v_data[..d].iter().zip(cv4_slice[..d].iter()).enumerate() {
            assert!(
                (a - b).abs() < 0.35,
                "K8V4 V error too large at {i}: {a} vs {b}"
            );
            mse_rotated += (a - b) * (a - b);
            let q_unrot = (a / unrotated_scale).round().clamp(-8.0, 7.0) * unrotated_scale;
            mse_unrotated += (a - q_unrot) * (a - q_unrot);
        }
        assert!(
            mse_rotated < mse_unrotated * 0.25,
            "Hadamard rotation should cut 4-bit outlier MSE by >4x: rotated={mse_rotated} vs unrotated={mse_unrotated}"
        );
    }

    #[test]
    fn test_kv_cache_overflow() {
        let mut cache = KvCache::new(2, 2, 4);
        let k = Tensor::from_f32(&[1.0; 6], &[3, 2]);
        let v = Tensor::from_f32(&[1.0; 6], &[3, 2]);
        cache.update(&k, &v).unwrap();

        // This should overflow (3 + 2 > 4)
        let k2 = Tensor::from_f32(&[1.0; 4], &[2, 2]);
        let v2 = Tensor::from_f32(&[1.0; 4], &[2, 2]);
        assert!(cache.update(&k2, &v2).is_err());
    }
}
