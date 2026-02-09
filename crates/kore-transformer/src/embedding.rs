//! Token and positional embedding lookup.

use kore_core::{KoreError, Tensor};

/// Token embedding table + optional positional embedding.
pub struct Embedding {
    /// Weight matrix: [vocab_size, d_model]
    pub weight: Tensor,
    /// Optional positional embedding: [max_seq_len, d_model]
    pub pos_weight: Option<Tensor>,
    pub vocab_size: usize,
    pub d_model: usize,
}

impl Embedding {
    /// Create a new embedding layer with random initialization.
    pub fn new(vocab_size: usize, d_model: usize, max_seq_len: Option<usize>) -> Self {
        // Xavier-style init: scale = sqrt(1/d_model)
        let scale = (1.0 / d_model as f64).sqrt() as f32;
        let weight_data: Vec<f32> = (0..vocab_size * d_model)
            .map(|i| {
                let x = ((i * 2654435761 + 1013904223) & 0xFFFFFF) as f32 / 0xFFFFFF as f32;
                (x * 2.0 - 1.0) * scale
            })
            .collect();
        let weight = Tensor::from_f32(&weight_data, &[vocab_size, d_model]);

        let pos_weight = max_seq_len.map(|max_len| {
            let pos_data: Vec<f32> = (0..max_len * d_model)
                .map(|i| {
                    let pos = i / d_model;
                    let dim = i % d_model;
                    if dim % 2 == 0 {
                        let freq = 1.0 / (10000.0_f32).powf(dim as f32 / d_model as f32);
                        (pos as f32 * freq).sin()
                    } else {
                        let freq = 1.0 / (10000.0_f32).powf((dim - 1) as f32 / d_model as f32);
                        (pos as f32 * freq).cos()
                    }
                })
                .collect();
            Tensor::from_f32(&pos_data, &[max_len, d_model])
        });

        Self { weight, pos_weight, vocab_size, d_model }
    }

    /// Look up token embeddings for a sequence of token IDs.
    ///
    /// `token_ids`: &[usize] of length seq_len
    /// Returns: [seq_len, d_model]
    pub fn forward(&self, token_ids: &[usize]) -> Result<Tensor, KoreError> {
        let seq_len = token_ids.len();
        let d = self.d_model;
        let w = self.weight.as_f32_slice()
            .ok_or(KoreError::StorageError("expected f32 embedding".into()))?;

        let mut out = vec![0.0f32; seq_len * d];
        for (i, &tid) in token_ids.iter().enumerate() {
            if tid >= self.vocab_size {
                return Err(KoreError::StorageError(
                    format!("token_id {} >= vocab_size {}", tid, self.vocab_size),
                ));
            }
            out[i * d..(i + 1) * d].copy_from_slice(&w[tid * d..(tid + 1) * d]);
        }

        // Add positional embedding if present
        if let Some(ref pos) = self.pos_weight {
            let p = pos.as_f32_slice().ok_or(KoreError::StorageError("expected f32 pos embedding".into()))?;
            for i in 0..seq_len {
                for j in 0..d {
                    out[i * d + j] += p[i * d + j];
                }
            }
        }

        Ok(Tensor::from_f32(&out, &[seq_len, d]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_lookup() {
        let emb = Embedding::new(100, 32, Some(64));
        let ids = vec![0, 5, 10, 99];
        let out = emb.forward(&ids).unwrap();
        assert_eq!(out.shape().dims(), &[4, 32]);
    }

    #[test]
    fn test_embedding_out_of_range() {
        let emb = Embedding::new(100, 32, None);
        let ids = vec![100]; // out of range
        assert!(emb.forward(&ids).is_err());
    }

    #[test]
    fn test_embedding_no_positional() {
        let emb = Embedding::new(50, 16, None);
        let ids = vec![0, 1, 2];
        let out = emb.forward(&ids).unwrap();
        assert_eq!(out.shape().dims(), &[3, 16]);
    }
}
