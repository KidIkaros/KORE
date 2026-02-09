//! MultipackSampler — pack multiple sequences into one batch to eliminate padding waste.
//!
//! Inspired by Axolotl's multipack approach: instead of padding all sequences
//! to the longest in the batch, pack shorter sequences together into a single
//! "super-sequence" up to `max_seq_len`. This dramatically reduces wasted compute
//! on padding tokens.
//!
//! # Algorithm
//! 1. Sort sequences by length (longest first)
//! 2. Greedily bin-pack into bins of capacity `max_seq_len`
//! 3. Each bin becomes one training example (with attention masking to prevent
//!    cross-sequence attention)

use rand::seq::SliceRandom;
use rand::thread_rng;

/// A packed batch: multiple sequences concatenated with their boundaries.
#[derive(Debug, Clone)]
pub struct PackedBatch {
    /// Concatenated token IDs for this batch element.
    pub tokens: Vec<usize>,
    /// Sequence boundaries: `boundaries[i]..boundaries[i+1]` is sequence i.
    pub boundaries: Vec<usize>,
    /// Number of sequences packed into this batch element.
    pub num_sequences: usize,
}

impl PackedBatch {
    /// Total number of tokens (including any padding at the end).
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether this batch is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Get the i-th sequence's tokens.
    pub fn sequence(&self, i: usize) -> &[usize] {
        let start = self.boundaries[i];
        let end = self.boundaries[i + 1];
        &self.tokens[start..end]
    }

    /// Generate an attention mask that prevents cross-sequence attention.
    ///
    /// Returns a `[len, len]` mask where `mask[i][j] = 1.0` if token i can
    /// attend to token j, and `0.0` otherwise.
    pub fn attention_mask(&self) -> Vec<f32> {
        let n = self.tokens.len();
        let mut mask = vec![0.0f32; n * n];

        for seq_idx in 0..self.num_sequences {
            let start = self.boundaries[seq_idx];
            let end = self.boundaries[seq_idx + 1];

            // Tokens within the same sequence can attend to each other (causal)
            for i in start..end {
                for j in start..=i {
                    mask[i * n + j] = 1.0;
                }
            }
        }

        mask
    }
}

/// Packs variable-length sequences into fixed-size bins to minimize padding.
pub struct MultipackSampler {
    /// Maximum sequence length per bin.
    pub max_seq_len: usize,
    /// Whether to shuffle sequences before packing.
    pub shuffle: bool,
}

impl MultipackSampler {
    /// Create a new MultipackSampler.
    pub fn new(max_seq_len: usize, shuffle: bool) -> Self {
        Self { max_seq_len, shuffle }
    }

    /// Pack a set of sequences into batches.
    ///
    /// Each input sequence is a `Vec<usize>` of token IDs.
    /// Returns a vector of `PackedBatch`es, each containing multiple sequences
    /// packed up to `max_seq_len` tokens.
    ///
    /// Sequences longer than `max_seq_len` are truncated.
    pub fn pack(&self, sequences: &[Vec<usize>]) -> Vec<PackedBatch> {
        // Create indices sorted by length (longest first for better packing)
        let mut indices: Vec<usize> = (0..sequences.len()).collect();

        if self.shuffle {
            indices.shuffle(&mut thread_rng());
        }

        // Sort by length descending (stable sort preserves shuffle order for equal lengths)
        indices.sort_by(|&a, &b| sequences[b].len().cmp(&sequences[a].len()));

        // Greedy first-fit decreasing bin packing
        let mut bins: Vec<PackedBatch> = Vec::new();
        let mut bin_remaining: Vec<usize> = Vec::new();

        for &idx in &indices {
            let seq = &sequences[idx];
            let seq_len = seq.len().min(self.max_seq_len);

            if seq_len == 0 {
                continue;
            }

            // Find first bin with enough space
            let mut placed = false;
            for (bin_idx, remaining) in bin_remaining.iter_mut().enumerate() {
                if *remaining >= seq_len {
                    // Add to this bin
                    let batch = &mut bins[bin_idx];
                    let start = batch.tokens.len();
                    batch.tokens.extend_from_slice(&seq[..seq_len]);
                    batch.boundaries.push(batch.tokens.len());
                    batch.num_sequences += 1;
                    *remaining -= seq_len;
                    placed = true;
                    break;
                }
            }

            if !placed {
                // Create new bin
                let mut batch = PackedBatch {
                    tokens: Vec::with_capacity(self.max_seq_len),
                    boundaries: vec![0],
                    num_sequences: 0,
                };
                batch.tokens.extend_from_slice(&seq[..seq_len]);
                batch.boundaries.push(batch.tokens.len());
                batch.num_sequences = 1;
                bin_remaining.push(self.max_seq_len - seq_len);
                bins.push(batch);
            }
        }

        bins
    }

    /// Compute packing efficiency: total tokens / total capacity.
    pub fn efficiency(batches: &[PackedBatch], max_seq_len: usize) -> f32 {
        let total_tokens: usize = batches.iter().map(|b| b.tokens.len()).sum();
        let total_capacity = batches.len() * max_seq_len;
        if total_capacity == 0 { return 0.0; }
        total_tokens as f32 / total_capacity as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_packing() {
        let sampler = MultipackSampler::new(10, false);
        let sequences = vec![
            vec![1, 2, 3],       // len 3
            vec![4, 5, 6, 7],    // len 4
            vec![8, 9],          // len 2
        ];

        let batches = sampler.pack(&sequences);
        // Total tokens = 3 + 4 + 2 = 9, fits in one bin of size 10
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_sequences, 3);
        assert_eq!(batches[0].tokens.len(), 9);
    }

    #[test]
    fn test_multiple_bins() {
        let sampler = MultipackSampler::new(5, false);
        let sequences = vec![
            vec![1, 2, 3],    // len 3
            vec![4, 5, 6],    // len 3
            vec![7, 8, 9],    // len 3
        ];

        let batches = sampler.pack(&sequences);
        // Each bin holds max 5 tokens. 3+3=6 > 5, so can't fit two len-3 seqs.
        // Actually 3+3=6 > 5, so we need at least 2 bins: [3,_] and [3,_] and [3,_]
        // Wait: first bin gets seq of len 3, remaining=2. Next seq len 3 doesn't fit.
        // New bin. Third seq len 3 doesn't fit in either (remaining 2 each). New bin.
        assert_eq!(batches.len(), 3);
    }

    #[test]
    fn test_packing_efficiency() {
        let sampler = MultipackSampler::new(10, false);
        let sequences = vec![
            vec![1; 5],
            vec![2; 5],
            vec![3; 3],
            vec![4; 2],
        ];

        let batches = sampler.pack(&sequences);
        let eff = MultipackSampler::efficiency(&batches, 10);
        // Total tokens = 5+5+3+2 = 15
        assert!(eff > 0.5, "efficiency={}", eff);
    }

    #[test]
    fn test_attention_mask() {
        let batch = PackedBatch {
            tokens: vec![1, 2, 3, 4, 5],
            boundaries: vec![0, 3, 5],
            num_sequences: 2,
        };

        let mask = batch.attention_mask();
        // Seq 0: tokens 0,1,2 — causal within
        assert_eq!(mask[0 * 5 + 0], 1.0); // token 0 attends to 0
        assert_eq!(mask[1 * 5 + 0], 1.0); // token 1 attends to 0
        assert_eq!(mask[1 * 5 + 1], 1.0); // token 1 attends to 1
        assert_eq!(mask[2 * 5 + 0], 1.0); // token 2 attends to 0

        // Seq 1: tokens 3,4 — causal within
        assert_eq!(mask[3 * 5 + 3], 1.0); // token 3 attends to 3
        assert_eq!(mask[4 * 5 + 3], 1.0); // token 4 attends to 3

        // Cross-sequence: blocked
        assert_eq!(mask[3 * 5 + 0], 0.0); // token 3 cannot attend to token 0
        assert_eq!(mask[0 * 5 + 3], 0.0); // token 0 cannot attend to token 3
    }

    #[test]
    fn test_sequence_extraction() {
        let batch = PackedBatch {
            tokens: vec![10, 20, 30, 40, 50],
            boundaries: vec![0, 2, 5],
            num_sequences: 2,
        };

        assert_eq!(batch.sequence(0), &[10, 20]);
        assert_eq!(batch.sequence(1), &[30, 40, 50]);
    }

    #[test]
    fn test_empty_sequences_skipped() {
        let sampler = MultipackSampler::new(10, false);
        let sequences = vec![
            vec![],
            vec![1, 2],
            vec![],
            vec![3],
        ];

        let batches = sampler.pack(&sequences);
        let total_tokens: usize = batches.iter().map(|b| b.tokens.len()).sum();
        assert_eq!(total_tokens, 3); // only non-empty sequences
    }

    #[test]
    fn test_truncation() {
        let sampler = MultipackSampler::new(3, false);
        let sequences = vec![
            vec![1, 2, 3, 4, 5], // len 5, truncated to 3
        ];

        let batches = sampler.pack(&sequences);
        assert_eq!(batches[0].tokens.len(), 3);
        assert_eq!(batches[0].tokens, vec![1, 2, 3]);
    }
}
