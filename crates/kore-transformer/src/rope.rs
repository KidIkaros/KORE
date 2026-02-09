//! Rotary Position Embeddings (RoPE).
//!
//! Applies rotation to Q and K vectors based on position, enabling
//! relative position encoding without explicit positional embeddings.
//! Used by LLaMA, Mistral, and most modern LLMs.

/// Precomputed RoPE frequency table.
pub struct RopeTable {
    /// cos values: [max_seq_len, d_head/2]
    cos: Vec<f32>,
    /// sin values: [max_seq_len, d_head/2]
    sin: Vec<f32>,
    pub d_head: usize,
    pub max_seq_len: usize,
    /// Base frequency (default 10000.0, some models use 1000000.0)
    pub base: f32,
}

impl RopeTable {
    /// Precompute the RoPE frequency table.
    ///
    /// `d_head`: dimension per attention head
    /// `max_seq_len`: maximum sequence length to precompute
    /// `base`: base frequency (typically 10000.0)
    pub fn new(d_head: usize, max_seq_len: usize, base: f32) -> Self {
        let half_d = d_head / 2;
        let mut cos = vec![0.0f32; max_seq_len * half_d];
        let mut sin = vec![0.0f32; max_seq_len * half_d];

        for pos in 0..max_seq_len {
            for i in 0..half_d {
                let freq = 1.0 / base.powf(2.0 * i as f32 / d_head as f32);
                let angle = pos as f32 * freq;
                cos[pos * half_d + i] = angle.cos();
                sin[pos * half_d + i] = angle.sin();
            }
        }

        Self { cos, sin, d_head, max_seq_len, base }
    }

    /// Apply RoPE to Q and K vectors in-place.
    ///
    /// `q`: mutable slice of shape [seq_len * d_head] (flattened [seq_len, d_head])
    /// `k`: mutable slice of shape [seq_len * d_head]
    /// `seq_offset`: position offset (for KV-cache continuation)
    /// `seq_len`: number of tokens in this batch
    pub fn apply(&self, q: &mut [f32], k: &mut [f32], seq_offset: usize, seq_len: usize) {
        let d = self.d_head;
        let half_d = d / 2;

        for t in 0..seq_len {
            let pos = seq_offset + t;
            if pos >= self.max_seq_len {
                break;
            }

            let cos_row = &self.cos[pos * half_d..(pos + 1) * half_d];
            let sin_row = &self.sin[pos * half_d..(pos + 1) * half_d];

            // Apply to Q
            rotate_half(&mut q[t * d..(t + 1) * d], cos_row, sin_row, half_d);
            // Apply to K
            rotate_half(&mut k[t * d..(t + 1) * d], cos_row, sin_row, half_d);
        }
    }

    /// Apply RoPE to a single vector (Q or K) in-place.
    pub fn apply_single(&self, vec: &mut [f32], seq_offset: usize, seq_len: usize) {
        let d = self.d_head;
        let half_d = d / 2;

        for t in 0..seq_len {
            let pos = seq_offset + t;
            if pos >= self.max_seq_len {
                break;
            }

            let cos_row = &self.cos[pos * half_d..(pos + 1) * half_d];
            let sin_row = &self.sin[pos * half_d..(pos + 1) * half_d];

            rotate_half(&mut vec[t * d..(t + 1) * d], cos_row, sin_row, half_d);
        }
    }
}

/// Apply the rotation: for each pair (x_i, x_{i+half_d}):
///   x_i'       = x_i * cos - x_{i+half_d} * sin
///   x_{i+half_d}' = x_i * sin + x_{i+half_d} * cos
#[inline]
fn rotate_half(x: &mut [f32], cos: &[f32], sin: &[f32], half_d: usize) {
    for i in 0..half_d {
        let x0 = x[i];
        let x1 = x[i + half_d];
        x[i] = x0 * cos[i] - x1 * sin[i];
        x[i + half_d] = x0 * sin[i] + x1 * cos[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_table_creation() {
        let table = RopeTable::new(64, 128, 10000.0);
        assert_eq!(table.d_head, 64);
        assert_eq!(table.max_seq_len, 128);
        // cos/sin should have max_seq_len * (d_head/2) entries
        assert_eq!(table.cos.len(), 128 * 32);
        assert_eq!(table.sin.len(), 128 * 32);
    }

    #[test]
    fn test_rope_position_zero_is_identity() {
        let table = RopeTable::new(4, 16, 10000.0);
        // At position 0, angle = 0, so cos=1, sin=0 → identity
        let mut q = vec![1.0, 2.0, 3.0, 4.0];
        let mut k = vec![5.0, 6.0, 7.0, 8.0];
        let q_orig = q.clone();
        let k_orig = k.clone();

        table.apply(&mut q, &mut k, 0, 1);

        for i in 0..4 {
            assert!(
                (q[i] - q_orig[i]).abs() < 1e-5,
                "q[{}]: expected {}, got {}", i, q_orig[i], q[i]
            );
            assert!(
                (k[i] - k_orig[i]).abs() < 1e-5,
                "k[{}]: expected {}, got {}", i, k_orig[i], k[i]
            );
        }
    }

    #[test]
    fn test_rope_changes_values_at_nonzero_position() {
        let table = RopeTable::new(4, 16, 10000.0);
        let mut q = vec![1.0, 2.0, 3.0, 4.0];
        let q_orig = q.clone();

        table.apply_single(&mut q, 5, 1);

        // At position 5, values should be rotated (different from original)
        let changed = q.iter().zip(q_orig.iter()).any(|(a, b)| (a - b).abs() > 1e-5);
        assert!(changed, "RoPE should modify values at non-zero position");
    }

    #[test]
    fn test_rope_preserves_norm() {
        let table = RopeTable::new(8, 32, 10000.0);
        let mut q = vec![1.0, 0.5, -0.3, 0.8, 0.2, -0.7, 0.4, 0.6];
        let norm_before: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();

        table.apply_single(&mut q, 10, 1);

        let norm_after: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm_before - norm_after).abs() < 1e-5,
            "RoPE should preserve vector norm: {} vs {}", norm_before, norm_after
        );
    }

    #[test]
    fn test_rope_multi_token() {
        let table = RopeTable::new(4, 16, 10000.0);
        // 3 tokens, d_head=4
        let mut q = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut k = q.clone();

        table.apply(&mut q, &mut k, 0, 3);

        // First token (pos=0) should be unchanged
        assert!((q[0] - 1.0).abs() < 1e-5);
        assert!((q[1] - 2.0).abs() < 1e-5);
        // Second token (pos=1) should be rotated
        let changed = (q[4] - 5.0).abs() > 1e-5 || (q[5] - 6.0).abs() > 1e-5;
        assert!(changed, "Token at pos=1 should be rotated");
    }

    #[test]
    fn test_rope_with_offset() {
        let table = RopeTable::new(4, 16, 10000.0);
        let mut q1 = vec![1.0, 2.0, 3.0, 4.0];
        let mut q2 = vec![1.0, 2.0, 3.0, 4.0];

        // Apply at offset 0 vs offset 5 — should give different results
        table.apply_single(&mut q1, 0, 1);
        table.apply_single(&mut q2, 5, 1);

        let different = q1.iter().zip(q2.iter()).any(|(a, b)| (a - b).abs() > 1e-5);
        assert!(different, "Different offsets should produce different rotations");
    }
}
