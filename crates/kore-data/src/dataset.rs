//! StreamingDataset — memory-mapped dataset for large corpora.
//!
//! Reads tokenized data from a flat binary file (u32 token IDs) using
//! memory mapping for zero-copy access. Supports random-access slicing
//! into sequences of arbitrary length.

use std::path::Path;

use kore_core::KoreError;

/// A memory-mapped dataset of token IDs stored as a flat u32 array.
///
/// The file format is simply concatenated u32 little-endian token IDs.
/// No headers, no separators — just raw tokens. Sequence boundaries
/// are managed externally or inferred by fixed-length chunking.
#[derive(Debug)]
pub struct StreamingDataset {
    /// Raw token data (owned for simplicity; could be mmap'd for huge files).
    tokens: Vec<u32>,
}

impl StreamingDataset {
    /// Load from a binary file of u32 token IDs.
    pub fn from_file(path: &Path) -> Result<Self, KoreError> {
        let data = std::fs::read(path)
            .map_err(|e| KoreError::StorageError(format!("Failed to read dataset: {}", e)))?;

        if data.len() % 4 != 0 {
            return Err(KoreError::StorageError(
                "Dataset file size not a multiple of 4 bytes".into(),
            ));
        }

        let tokens: Vec<u32> = data
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        Ok(Self { tokens })
    }

    /// Create from an in-memory token vector.
    pub fn from_tokens(tokens: Vec<u32>) -> Self {
        Self { tokens }
    }

    /// Total number of tokens in the dataset.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Get a single token by index.
    pub fn get(&self, idx: usize) -> Option<u32> {
        self.tokens.get(idx).copied()
    }

    /// Get a slice of tokens.
    pub fn slice(&self, start: usize, end: usize) -> &[u32] {
        let end = end.min(self.tokens.len());
        let start = start.min(end);
        &self.tokens[start..end]
    }

    /// Chunk the dataset into fixed-length sequences.
    ///
    /// Returns `(input_sequences, target_sequences)` where each target
    /// is shifted by 1 token (next-token prediction).
    ///
    /// Sequences shorter than `seq_len + 1` at the end are dropped.
    pub fn chunk_for_lm(&self, seq_len: usize) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        let mut offset = 0;
        while offset + seq_len < self.tokens.len() {
            inputs.push(self.tokens[offset..offset + seq_len].to_vec());
            targets.push(self.tokens[offset + 1..offset + seq_len + 1].to_vec());
            offset += seq_len;
        }

        (inputs, targets)
    }

    /// Randomly sample a contiguous sequence of `seq_len` tokens.
    pub fn random_sequence(&self, seq_len: usize) -> Option<&[u32]> {
        if self.tokens.len() < seq_len {
            return None;
        }
        let max_start = self.tokens.len() - seq_len;
        let start = rand::random::<usize>() % (max_start + 1);
        Some(&self.tokens[start..start + seq_len])
    }

    /// Save to a binary file.
    pub fn save(&self, path: &Path) -> Result<(), KoreError> {
        let bytes: Vec<u8> = self.tokens.iter()
            .flat_map(|&t| t.to_le_bytes())
            .collect();
        std::fs::write(path, &bytes)
            .map_err(|e| KoreError::StorageError(format!("Failed to write dataset: {}", e)))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_tokens() {
        let ds = StreamingDataset::from_tokens(vec![1, 2, 3, 4, 5]);
        assert_eq!(ds.len(), 5);
        assert!(!ds.is_empty());
        assert_eq!(ds.get(0), Some(1));
        assert_eq!(ds.get(4), Some(5));
        assert_eq!(ds.get(5), None);
    }

    #[test]
    fn test_slice() {
        let ds = StreamingDataset::from_tokens(vec![10, 20, 30, 40, 50]);
        assert_eq!(ds.slice(1, 4), &[20, 30, 40]);
        assert_eq!(ds.slice(3, 100), &[40, 50]); // clamped
        assert_eq!(ds.slice(10, 20), &[] as &[u32]); // out of bounds
    }

    #[test]
    fn test_chunk_for_lm() {
        let ds = StreamingDataset::from_tokens(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let (inputs, targets) = ds.chunk_for_lm(4);

        // 10 tokens, seq_len=4: need 5 tokens per chunk (4 input + 1 target overlap)
        // Chunks: [0..4] → [1..5], [4..8] → [5..9]
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0], vec![0, 1, 2, 3]);
        assert_eq!(targets[0], vec![1, 2, 3, 4]);
        assert_eq!(inputs[1], vec![4, 5, 6, 7]);
        assert_eq!(targets[1], vec![5, 6, 7, 8]);
    }

    #[test]
    fn test_random_sequence() {
        let ds = StreamingDataset::from_tokens(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let seq = ds.random_sequence(3).unwrap();
        assert_eq!(seq.len(), 3);
        // All tokens should be in range
        for &t in seq {
            assert!(t < 10);
        }
    }

    #[test]
    fn test_random_sequence_too_long() {
        let ds = StreamingDataset::from_tokens(vec![1, 2, 3]);
        assert!(ds.random_sequence(10).is_none());
    }

    #[test]
    fn test_save_load_roundtrip() {
        let ds = StreamingDataset::from_tokens(vec![100, 200, 300, 400]);
        let tmp = std::env::temp_dir().join("kore_test_dataset.bin");
        ds.save(&tmp).unwrap();

        let loaded = StreamingDataset::from_file(&tmp).unwrap();
        assert_eq!(loaded.len(), 4);
        assert_eq!(loaded.get(0), Some(100));
        assert_eq!(loaded.get(3), Some(400));

        std::fs::remove_file(&tmp).ok();
    }
}
