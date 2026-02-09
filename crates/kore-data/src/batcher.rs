//! TokenBatcher — efficient token-level batching for LLM training.
//!
//! Converts raw token sequences into padded Tensor batches ready for
//! model consumption. Supports both fixed-length and variable-length
//! (with padding) batching strategies.

use kore_core::{DType, Tensor};

/// Padding strategy for variable-length sequences.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PaddingStrategy {
    /// Pad all sequences to the longest in the batch.
    LongestInBatch,
    /// Pad all sequences to a fixed maximum length.
    Fixed(usize),
    /// No padding — all sequences must be the same length (panics otherwise).
    NoPadding,
}

/// Batches token sequences into Tensors for model training/inference.
pub struct TokenBatcher {
    /// Padding strategy.
    pub padding: PaddingStrategy,
    /// Token ID used for padding (typically 0 or a special <pad> token).
    pub pad_token_id: usize,
}

impl TokenBatcher {
    /// Create a new TokenBatcher.
    pub fn new(padding: PaddingStrategy, pad_token_id: usize) -> Self {
        Self { padding, pad_token_id }
    }

    /// Batch a set of token sequences into a padded Tensor.
    ///
    /// Returns `(input_ids, attention_mask)` where:
    /// - `input_ids`: `[batch_size, seq_len]` tensor of token IDs (as f32)
    /// - `attention_mask`: `[batch_size, seq_len]` tensor (1.0 for real tokens, 0.0 for padding)
    pub fn batch(&self, sequences: &[Vec<usize>]) -> (Tensor, Tensor) {
        if sequences.is_empty() {
            return (
                Tensor::zeros(&[0, 0], DType::F32),
                Tensor::zeros(&[0, 0], DType::F32),
            );
        }

        let batch_size = sequences.len();
        let seq_len = match self.padding {
            PaddingStrategy::LongestInBatch => sequences.iter().map(|s| s.len()).max().unwrap_or(0),
            PaddingStrategy::Fixed(len) => len,
            PaddingStrategy::NoPadding => {
                let first_len = sequences[0].len();
                assert!(
                    sequences.iter().all(|s| s.len() == first_len),
                    "NoPadding requires all sequences to have the same length"
                );
                first_len
            }
        };

        let mut input_data = vec![self.pad_token_id as f32; batch_size * seq_len];
        let mut mask_data = vec![0.0f32; batch_size * seq_len];

        for (i, seq) in sequences.iter().enumerate() {
            let copy_len = seq.len().min(seq_len);
            for j in 0..copy_len {
                input_data[i * seq_len + j] = seq[j] as f32;
                mask_data[i * seq_len + j] = 1.0;
            }
        }

        let input_ids = Tensor::from_f32(&input_data, &[batch_size, seq_len]);
        let attention_mask = Tensor::from_f32(&mask_data, &[batch_size, seq_len]);

        (input_ids, attention_mask)
    }

    /// Batch with labels for next-token prediction.
    ///
    /// Returns `(input_ids, attention_mask, labels)` where labels are
    /// shifted by 1 position. Label positions corresponding to padding
    /// are set to -100 (ignore index for cross-entropy).
    pub fn batch_with_labels(&self, sequences: &[Vec<usize>]) -> (Tensor, Tensor, Tensor) {
        let (input_ids, attention_mask) = self.batch(sequences);
        let dims = input_ids.shape().dims();
        if dims[0] == 0 || dims[1] == 0 {
            return (input_ids, attention_mask, Tensor::zeros(&[0, 0], DType::F32));
        }

        let batch_size = dims[0];
        let seq_len = dims[1];
        let input_data = input_ids.as_f32_slice().unwrap();
        let mask_data = attention_mask.as_f32_slice().unwrap();

        let mut label_data = vec![-100.0f32; batch_size * seq_len];

        for i in 0..batch_size {
            for j in 0..seq_len - 1 {
                // Label at position j is the token at position j+1
                if mask_data[i * seq_len + j + 1] > 0.5 {
                    label_data[i * seq_len + j] = input_data[i * seq_len + j + 1];
                }
            }
        }

        let labels = Tensor::from_f32(&label_data, &[batch_size, seq_len]);
        (input_ids, attention_mask, labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_longest() {
        let batcher = TokenBatcher::new(PaddingStrategy::LongestInBatch, 0);
        let sequences = vec![
            vec![1, 2, 3],
            vec![4, 5],
        ];

        let (ids, mask) = batcher.batch(&sequences);
        assert_eq!(ids.shape().dims(), &[2, 3]);
        assert_eq!(mask.shape().dims(), &[2, 3]);

        let id_data = ids.as_f32_slice().unwrap();
        assert_eq!(id_data, &[1.0, 2.0, 3.0, 4.0, 5.0, 0.0]);

        let mask_data = mask.as_f32_slice().unwrap();
        assert_eq!(mask_data, &[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_batch_fixed() {
        let batcher = TokenBatcher::new(PaddingStrategy::Fixed(5), 0);
        let sequences = vec![vec![1, 2, 3]];

        let (ids, mask) = batcher.batch(&sequences);
        assert_eq!(ids.shape().dims(), &[1, 5]);

        let id_data = ids.as_f32_slice().unwrap();
        assert_eq!(id_data, &[1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn test_batch_no_padding() {
        let batcher = TokenBatcher::new(PaddingStrategy::NoPadding, 0);
        let sequences = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
        ];

        let (ids, _mask) = batcher.batch(&sequences);
        assert_eq!(ids.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_batch_with_labels() {
        let batcher = TokenBatcher::new(PaddingStrategy::LongestInBatch, 0);
        let sequences = vec![
            vec![10, 20, 30, 40],
        ];

        let (ids, _mask, labels) = batcher.batch_with_labels(&sequences);
        assert_eq!(ids.shape().dims(), &[1, 4]);
        assert_eq!(labels.shape().dims(), &[1, 4]);

        let label_data = labels.as_f32_slice().unwrap();
        // Labels: [20, 30, 40, -100]
        assert!((label_data[0] - 20.0).abs() < 1e-6);
        assert!((label_data[1] - 30.0).abs() < 1e-6);
        assert!((label_data[2] - 40.0).abs() < 1e-6);
        assert!((label_data[3] - (-100.0)).abs() < 1e-6);
    }

    #[test]
    fn test_batch_with_labels_padding() {
        let batcher = TokenBatcher::new(PaddingStrategy::LongestInBatch, 0);
        let sequences = vec![
            vec![10, 20, 30],
            vec![40, 50],
        ];

        let (_ids, _mask, labels) = batcher.batch_with_labels(&sequences);
        let label_data = labels.as_f32_slice().unwrap();

        // Row 0: labels = [20, 30, -100]
        assert!((label_data[0] - 20.0).abs() < 1e-6);
        assert!((label_data[1] - 30.0).abs() < 1e-6);
        assert!((label_data[2] - (-100.0)).abs() < 1e-6);

        // Row 1: labels = [50, -100, -100] (second token is padding)
        assert!((label_data[3] - 50.0).abs() < 1e-6);
        assert!((label_data[4] - (-100.0)).abs() < 1e-6);
        assert!((label_data[5] - (-100.0)).abs() < 1e-6);
    }

    #[test]
    fn test_empty_batch() {
        let batcher = TokenBatcher::new(PaddingStrategy::LongestInBatch, 0);
        let (ids, mask) = batcher.batch(&[]);
        assert_eq!(ids.shape().dims(), &[0, 0]);
        assert_eq!(mask.shape().dims(), &[0, 0]);
    }

    #[test]
    fn test_custom_pad_token() {
        let batcher = TokenBatcher::new(PaddingStrategy::Fixed(4), 999);
        let sequences = vec![vec![1, 2]];

        let (ids, _mask) = batcher.batch(&sequences);
        let id_data = ids.as_f32_slice().unwrap();
        assert_eq!(id_data, &[1.0, 2.0, 999.0, 999.0]);
    }
}
