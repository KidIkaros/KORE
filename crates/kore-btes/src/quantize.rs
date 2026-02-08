//! Float → ternary/quaternary quantization with per-row scaling.

use crate::encoder::{Trit, Quat};

/// Quantize f32 weights to ternary {-1, 0, +1} with a threshold.
///
/// Returns (trits, scale) where scale reconstructs: weight ≈ trit * scale
pub fn quantize_ternary(weights: &[f32], threshold: f32) -> (Vec<Trit>, f32) {
    let abs_max = weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
    let scale = if abs_max < 1e-8 { 1.0 } else { abs_max };

    let trits: Vec<Trit> = weights
        .iter()
        .map(|&w| {
            let normalized = w / scale;
            if normalized > threshold {
                Trit::Pos
            } else if normalized < -threshold {
                Trit::Neg
            } else {
                Trit::Zero
            }
        })
        .collect();

    (trits, scale)
}

/// Quantize f32 weights to quaternary {-3, -1, +1, +3} with per-row scaling.
///
/// Returns (quats, scale) where scale reconstructs: weight ≈ quat_value * scale
pub fn quantize_quaternary(weights: &[f32]) -> (Vec<Quat>, f32) {
    let abs_max = weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
    let scale = if abs_max < 1e-8 { 1.0 } else { abs_max / 3.0 };

    let quats: Vec<Quat> = weights
        .iter()
        .map(|&w| {
            let normalized = w / scale;
            if normalized < -2.0 {
                Quat::Neg3
            } else if normalized < 0.0 {
                Quat::Neg1
            } else if normalized < 2.0 {
                Quat::Pos1
            } else {
                Quat::Pos3
            }
        })
        .collect();

    (quats, scale)
}

/// Dequantize ternary values back to f32.
pub fn dequantize_ternary(trits: &[Trit], scale: f32) -> Vec<f32> {
    trits.iter().map(|&t| (t as i8 as f32) * scale).collect()
}

/// Dequantize quaternary values back to f32.
pub fn dequantize_quaternary(quats: &[Quat], scale: f32) -> Vec<f32> {
    quats.iter().map(|q| q.to_f32() * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_quantize() {
        let weights = vec![0.9, -0.8, 0.01, 0.5, -0.3];
        let (trits, scale) = quantize_ternary(&weights, 0.3);

        assert_eq!(trits[0], Trit::Pos);  // 0.9/0.9 = 1.0 > 0.3
        assert_eq!(trits[1], Trit::Neg);  // -0.8/0.9 ≈ -0.89 < -0.3
        assert_eq!(trits[2], Trit::Zero); // 0.01/0.9 ≈ 0.01, within threshold
        assert!(scale > 0.0);
    }

    #[test]
    fn test_quaternary_quantize() {
        let weights = vec![3.0, 1.0, -1.0, -3.0, 0.5];
        let (quats, scale) = quantize_quaternary(&weights);

        assert_eq!(quats[0], Quat::Pos3);
        assert_eq!(quats[3], Quat::Neg3);
        assert!(scale > 0.0);
    }

    #[test]
    fn test_ternary_roundtrip() {
        let weights = vec![1.0, -1.0, 0.0, 0.5, -0.5];
        let (trits, scale) = quantize_ternary(&weights, 0.3);
        let reconstructed = dequantize_ternary(&trits, scale);

        // Ternary is lossy, but signs should match for large values
        assert!(reconstructed[0] > 0.0);
        assert!(reconstructed[1] < 0.0);
        assert_eq!(reconstructed[2], 0.0);
    }

    #[test]
    fn test_quaternary_roundtrip() {
        let weights = vec![3.0, 1.0, -1.0, -3.0];
        let (quats, scale) = quantize_quaternary(&weights);
        let reconstructed = dequantize_quaternary(&quats, scale);

        for (orig, recon) in weights.iter().zip(reconstructed.iter()) {
            assert!((orig - recon).abs() < 0.1, "orig={}, recon={}", orig, recon);
        }
    }
}
