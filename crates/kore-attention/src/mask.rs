//! Attention masks: causal, sliding window, custom.

use kore_core::Tensor;

/// Create a causal (lower-triangular) mask for autoregressive attention.
///
/// Returns a [seq_len, seq_len] tensor where:
/// - 0.0 = attend (not masked)
/// - -inf = don't attend (masked)
pub fn causal_mask(seq_len: usize) -> Tensor {
    let mut data = vec![f32::NEG_INFINITY; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..=i {
            data[i * seq_len + j] = 0.0;
        }
    }
    Tensor::from_f32(&data, &[seq_len, seq_len])
}

/// Create a sliding window causal mask.
///
/// Each position can attend to at most `window_size` previous positions.
pub fn sliding_window_mask(seq_len: usize, window_size: usize) -> Tensor {
    let mut data = vec![f32::NEG_INFINITY; seq_len * seq_len];
    for i in 0..seq_len {
        let start = if i >= window_size { i - window_size + 1 } else { 0 };
        for j in start..=i {
            data[i * seq_len + j] = 0.0;
        }
    }
    Tensor::from_f32(&data, &[seq_len, seq_len])
}

/// Create a padding mask from sequence lengths.
///
/// `lengths`: [batch_size] tensor of actual sequence lengths.
/// Returns [batch_size, max_len] mask where padded positions are -inf.
pub fn padding_mask(lengths: &[usize], max_len: usize) -> Tensor {
    let batch = lengths.len();
    let mut data = vec![f32::NEG_INFINITY; batch * max_len];
    for (b, &len) in lengths.iter().enumerate() {
        for j in 0..len.min(max_len) {
            data[b * max_len + j] = 0.0;
        }
    }
    Tensor::from_f32(&data, &[batch, max_len])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask() {
        let mask = causal_mask(4);
        let data = mask.as_f32_slice().unwrap();

        // Row 0: [0, -inf, -inf, -inf]
        assert_eq!(data[0], 0.0);
        assert!(data[1].is_infinite());

        // Row 3: [0, 0, 0, 0]
        assert_eq!(data[12], 0.0);
        assert_eq!(data[13], 0.0);
        assert_eq!(data[14], 0.0);
        assert_eq!(data[15], 0.0);
    }

    #[test]
    fn test_sliding_window_mask() {
        let mask = sliding_window_mask(6, 3);
        let data = mask.as_f32_slice().unwrap();

        // Row 5: can attend to positions 3,4,5 only
        assert!(data[5 * 6 + 2].is_infinite()); // pos 2: masked
        assert_eq!(data[5 * 6 + 3], 0.0);       // pos 3: visible
        assert_eq!(data[5 * 6 + 5], 0.0);       // pos 5: visible
    }

    #[test]
    fn test_padding_mask() {
        let mask = padding_mask(&[3, 5], 5);
        let data = mask.as_f32_slice().unwrap();

        // Batch 0: length 3 → [0, 0, 0, -inf, -inf]
        assert_eq!(data[0], 0.0);
        assert_eq!(data[2], 0.0);
        assert!(data[3].is_infinite());

        // Batch 1: length 5 → [0, 0, 0, 0, 0]
        assert_eq!(data[5], 0.0);
        assert_eq!(data[9], 0.0);
    }
}
