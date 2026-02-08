//! KV-cache for autoregressive generation.
//!
//! Stores key and value tensors from previous tokens to avoid
//! recomputation during sequential generation.

use kore_core::{KoreError, Tensor};

/// Key-Value cache for a single attention layer.
///
/// Stores accumulated K and V tensors, growing as new tokens are generated.
pub struct KvCache {
    /// Cached keys: [current_seq_len, d_k]
    keys: Option<Vec<f32>>,
    /// Cached values: [current_seq_len, d_v]
    values: Option<Vec<f32>>,
    /// Key dimension
    d_k: usize,
    /// Value dimension
    d_v: usize,
    /// Current sequence length (number of cached tokens)
    seq_len: usize,
    /// Maximum sequence length (pre-allocated capacity)
    max_seq_len: usize,
}

impl KvCache {
    /// Create a new empty KV cache.
    pub fn new(d_k: usize, d_v: usize, max_seq_len: usize) -> Self {
        Self {
            keys: Some(Vec::with_capacity(max_seq_len * d_k)),
            values: Some(Vec::with_capacity(max_seq_len * d_v)),
            d_k,
            d_v,
            seq_len: 0,
            max_seq_len,
        }
    }

    /// Append new key and value vectors to the cache.
    ///
    /// `new_keys`: [num_new_tokens, d_k]
    /// `new_values`: [num_new_tokens, d_v]
    ///
    /// Returns the full cached (keys, values) as tensors.
    pub fn update(
        &mut self,
        new_keys: &Tensor,
        new_values: &Tensor,
    ) -> Result<(Tensor, Tensor), KoreError> {
        let nk = new_keys.contiguous();
        let nv = new_values.contiguous();

        let nk_data = nk.as_f32_slice().ok_or_else(|| {
            KoreError::UnsupportedDType(new_keys.dtype())
        })?;
        let nv_data = nv.as_f32_slice().ok_or_else(|| {
            KoreError::UnsupportedDType(new_values.dtype())
        })?;

        let new_tokens = nk.shape().dims()[0];

        if self.seq_len + new_tokens > self.max_seq_len {
            return Err(KoreError::StorageError(format!(
                "KV cache overflow: {} + {} > {}",
                self.seq_len, new_tokens, self.max_seq_len
            )));
        }

        // Append to cache
        let keys = self.keys.as_mut().unwrap();
        let values = self.values.as_mut().unwrap();
        keys.extend_from_slice(nk_data);
        values.extend_from_slice(nv_data);
        self.seq_len += new_tokens;

        // Return full cached tensors
        let k_tensor = Tensor::from_f32(keys, &[self.seq_len, self.d_k]);
        let v_tensor = Tensor::from_f32(values, &[self.seq_len, self.d_v]);

        Ok((k_tensor, v_tensor))
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
        if let Some(keys) = &mut self.keys {
            keys.clear();
        }
        if let Some(values) = &mut self.values {
            values.clear();
        }
        self.seq_len = 0;
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
        assert_eq!(k_data[0], 1.0);  // first token
        assert_eq!(k_data[12], 3.0); // fourth token
    }

    #[test]
    fn test_kv_cache_clear() {
        let mut cache = KvCache::new(4, 4, 128);
        let k = Tensor::from_f32(&[1.0; 16], &[4, 4]);
        let v = Tensor::from_f32(&[1.0; 16], &[4, 4]);
        cache.update(&k, &v).unwrap();

        assert_eq!(cache.len(), 4);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.remaining(), 128);
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
