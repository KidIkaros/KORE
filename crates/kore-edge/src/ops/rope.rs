//! Rotary Position Embeddings (RoPE) on raw slices.

/// Apply RoPE to Q or K tensor in-place.
///
/// `data`: [seq_len, n_heads, head_dim] flattened row-major
/// `seq_len`, `n_heads`, `head_dim`: dimensions
/// `start_pos`: position offset (for KV-cache continuation)
/// `base`: RoPE base frequency (typically 10000.0)
pub fn apply_rope(
    data: &mut [f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    start_pos: usize,
    base: f32,
) {
    let half_dim = head_dim / 2;

    for pos in 0..seq_len {
        let abs_pos = (start_pos + pos) as f32;
        for h in 0..n_heads {
            let offset = pos * n_heads * head_dim + h * head_dim;
            for i in 0..half_dim {
                let freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
                let angle = abs_pos * freq;
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                let x0 = data[offset + i];
                let x1 = data[offset + half_dim + i];
                data[offset + i] = x0 * cos_val - x1 * sin_val;
                data[offset + half_dim + i] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_position_zero() {
        // At position 0, cos=1, sin=0 → identity
        let mut data = vec![1.0, 2.0, 3.0, 4.0]; // [1, 1, 4]
        let original = data.clone();
        apply_rope(&mut data, 1, 1, 4, 0, 10000.0);
        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "{} != {}", a, b);
        }
    }

    #[test]
    fn test_rope_changes_values() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0]; // [1, 1, 4]
        let original = data.clone();
        apply_rope(&mut data, 1, 1, 4, 5, 10000.0);
        // At position 5, values should change
        let changed = data.iter().zip(original.iter()).any(|(a, b)| (a - b).abs() > 1e-5);
        assert!(changed, "RoPE should modify values at non-zero position");
    }

    #[test]
    fn test_rope_preserves_norm() {
        // RoPE is a rotation, so it preserves L2 norm of each (x0, x1) pair
        let mut data: Vec<f32> = vec![3.0, 4.0, 1.0, 2.0]; // [1, 1, 4]
        let norm_before: f32 = (data[0] * data[0] + data[2] * data[2]).sqrt();
        apply_rope(&mut data, 1, 1, 4, 7, 10000.0);
        let norm_after: f32 = (data[0] * data[0] + data[2] * data[2]).sqrt();
        assert!((norm_before - norm_after).abs() < 1e-4);
    }
}
