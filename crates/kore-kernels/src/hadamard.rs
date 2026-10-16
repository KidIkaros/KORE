//! Fast Walsh-Hadamard Transform (`fwht`) for incoherence processing.
//!
//! Applies a normalized, self-inverse block Fast Walsh-Hadamard Transform along rows
//! to spread activation and weight outliers before extreme low-bit quantization
//! (QuIP#, SpinQuant, and rotated `BitLinear` / `QuatLinear` / `K8V4` KV-cache).

/// Compute the block size for the normalized Fast Walsh-Hadamard Transform given a row width.
///
/// Picks the largest power-of-2 divisor of `cols` up to 256, ensuring both power-of-2
/// dimensions (`64`, `128`, `256`, `512`, `1024`, ...) and composite hidden dimensions
/// (`384 = 3 * 128`, `1536 = 6 * 256`) are orthogonal and self-inverse while fitting in
/// a single 256-thread GPU shared-memory butterfly.
#[inline]
pub fn fwht_block_size(cols: usize) -> usize {
    if cols == 0 {
        0
    } else {
        (1usize << cols.trailing_zeros()).min(256)
    }
}

/// Apply the normalized in-place Fast Walsh-Hadamard Transform (`fwht`) to each row of `data`.
///
/// Since the normalized Hadamard matrix $H = H^T$ satisfies $H^2 = I$, calling
/// `fwht_inplace(data, cols)` twice restores the original vector within floating-point precision.
pub fn fwht_inplace(data: &mut [f32], cols: usize) {
    let block_size = fwht_block_size(cols);
    if block_size <= 1 {
        return;
    }
    let scale = 1.0f32 / (block_size as f32).sqrt();
    for block in data.chunks_exact_mut(block_size) {
        let mut h = 1usize;
        while h < block_size {
            for i in (0..block_size).step_by(h * 2) {
                for j in i..(i + h) {
                    let x = block[j];
                    let y = block[j + h];
                    block[j] = x + y;
                    block[j + h] = x - y;
                }
            }
            h *= 2;
        }
        for v in block.iter_mut() {
            *v *= scale;
        }
    }
}

/// Out-of-place normalized Fast Walsh-Hadamard Transform.
pub fn fwht(data: &[f32], cols: usize) -> Vec<f32> {
    let mut out = data.to_vec();
    fwht_inplace(&mut out, cols);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fwht_self_inverse_and_norm_preservation() {
        for &cols in &[8usize, 64, 128, 256, 384, 512] {
            let orig: Vec<f32> = (0..(cols * 2))
                .map(|i| ((i as f32) * 0.37).sin() + if i == 3 { 50.0 } else { 0.0 })
                .collect();
            let orig_norm_sq: f32 = orig.iter().map(|x| x * x).sum();

            let rotated = fwht(&orig, cols);
            let rot_norm_sq: f32 = rotated.iter().map(|x| x * x).sum();
            assert!(
                (orig_norm_sq - rot_norm_sq).abs() / orig_norm_sq < 1e-5,
                "FWHT must preserve L2 norm (cols={}): {} vs {}",
                cols,
                orig_norm_sq,
                rot_norm_sq
            );

            // Outlier at index 3 (value ~50.0) should be spread across the block
            let orig_max = orig.iter().copied().fold(0.0f32, f32::max);
            let rot_max = rotated.iter().copied().fold(0.0f32, f32::max);
            assert!(
                rot_max < orig_max * 0.5,
                "FWHT should reduce peak outlier magnitude (cols={}): {} -> {}",
                cols,
                orig_max,
                rot_max
            );

            // Self-inverse check: H(H(x)) == x
            let recovered = fwht(&rotated, cols);
            for (a, b) in orig.iter().zip(recovered.iter()) {
                assert!(
                    (a - b).abs() < 1e-4,
                    "FWHT must be self-inverse (cols={}): {} vs {}",
                    cols,
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn test_fwht_dot_product_invariance() {
        let cols = 128;
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.13).cos()).collect();
        let w: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.29).sin()).collect();

        let dot_orig: f32 = x.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        let x_rot = fwht(&x, cols);
        let w_rot = fwht(&w, cols);
        let dot_rot: f32 = x_rot.iter().zip(w_rot.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (dot_orig - dot_rot).abs() < 1e-4,
            "Orthogonal FWHT must preserve inner product: {} vs {}",
            dot_orig,
            dot_rot
        );
    }
}
