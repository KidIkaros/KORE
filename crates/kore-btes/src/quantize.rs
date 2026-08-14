//! Float → ternary/quaternary quantization with per-row scaling.

use crate::encoder::{Quat, Trit};

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

/// Quantize f32 weights to binary {-1, +1} using sign-only quantization.
///
/// This matches PrismML's Bonsai Q1_0_g128 format:
/// - Each weight is ±1 (sign only, no zero values)
/// - A single float scale factor per block of 128 weights
/// - 128 weights packed as 16 bytes (1 bit per weight)
///
/// Returns (binary_bits, scale) where: bit=0 → -1, bit=1 → +1
pub fn quantize_binary(weights: &[f32]) -> (Vec<u8>, f32) {
    let abs_max = weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
    let scale = if abs_max < 1e-8 { 1.0 } else { abs_max };

    let n_bytes = weights.len().div_ceil(8);
    let mut packed = vec![0u8; n_bytes];
    for (i, &w) in weights.iter().enumerate() {
        if w >= 0.0 {
            packed[i / 8] |= 1 << (i % 8);
        }
    }

    (packed, scale)
}

/// Dequantize binary weights back to f32.
/// bit=0 → -1.0, bit=1 → +1.0, multiplied by scale.
pub fn dequantize_binary(packed: &[u8], scale: f32) -> Vec<f32> {
    let n_weights = packed.len() * 8;
    let mut result = Vec::with_capacity(n_weights);
    for &byte in packed {
        for bit in 0..8 {
            let val = if (byte >> bit) & 1 == 1 { 1.0 } else { -1.0 };
            result.push(val * scale);
        }
    }
    result
}

/// Pack binary values into Q1_0_g128 blocks, matching PrismML's format exactly.
///
/// Format per block (18 bytes):
///   - bytes 0-1: FP16 scale factor (little-endian)
///   - bytes 2-17: 128 packed bits (16 bytes, LSB first)
///
/// weights.len() must be a multiple of 128.
pub fn pack_binary_block(weights: &[bool]) -> Vec<u8> {
    assert!(
        weights.len().is_multiple_of(128),
        "Binary block must be multiple of 128 weights"
    );

    let n_blocks = weights.len() / 128;
    let mut result = Vec::with_capacity(n_blocks * 18);

    for block_idx in 0..n_blocks {
        let start = block_idx * 128;
        let block = &weights[start..start + 128];

        // FP16 scale = 1.0 → 0x3C00
        result.extend_from_slice(&0x3C00u16.to_le_bytes());

        let mut qs = [0u8; 16];
        for (i, &val) in block.iter().enumerate() {
            if val {
                qs[i / 8] |= 1 << (i % 8);
            }
        }
        result.extend_from_slice(&qs);
    }

    result
}

/// Unpack Q1_0_g128 blocks back to boolean values.
///
/// Input data length must be multiple of 18 bytes.
/// Output length = (data.len() / 18) * 128.
pub fn unpack_binary_block(data: &[u8]) -> Vec<bool> {
    assert!(
        data.len().is_multiple_of(18),
        "Q1_0_g128 block data must be multiple of 18 bytes"
    );

    let n_blocks = data.len() / 18;
    let mut result = Vec::with_capacity(n_blocks * 128);

    for block_idx in 0..n_blocks {
        let offset = block_idx * 18 + 2; // skip 2-byte FP16 scale
        let qs = &data[offset..offset + 16];

        for &byte in qs {
            for bit in 0..8 {
                result.push((byte >> bit) & 1 == 1);
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_quantize() {
        let weights = vec![0.9, -0.8, 0.01, 0.5, -0.3];
        let (trits, scale) = quantize_ternary(&weights, 0.3);

        assert_eq!(trits[0], Trit::Pos); // 0.9/0.9 = 1.0 > 0.3
        assert_eq!(trits[1], Trit::Neg); // -0.8/0.9 ≈ -0.89 < -0.3
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

    #[test]
    fn test_binary_quantize() {
        let weights = vec![0.9, -0.8, 0.01, 0.5, -0.3, -1.2, 0.7, -0.4];
        let (packed, scale) = quantize_binary(&weights);

        assert_eq!(packed.len(), 1); // 8 weights = 1 byte
        assert_eq!(scale, 1.2); // abs_max = 1.2

        // Check individual bits
        let byte = packed[0];
        assert_eq!(byte & 0x01, 1); // +0.9 → 1
        assert_eq!((byte >> 1) & 1, 0); // -0.8 → 0
        assert_eq!((byte >> 2) & 1, 1); // +0.01 → 1
        assert_eq!((byte >> 3) & 1, 1); // +0.5 → 1
        assert_eq!((byte >> 4) & 1, 0); // -0.3 → 0
        assert_eq!((byte >> 5) & 1, 0); // -1.2 → 0
        assert_eq!((byte >> 6) & 1, 1); // +0.7 → 1
        assert_eq!((byte >> 7) & 1, 0); // -0.4 → 0
    }

    #[test]
    fn test_binary_dequantize() {
        let packed: Vec<u8> = vec![0b10110001]; // 8 values
        let scale = 2.0;
        let result = dequantize_binary(&packed, scale);

        assert_eq!(result.len(), 8);
        assert_eq!(result[0], 2.0); // bit 1 → +1 * 2.0
        assert_eq!(result[1], -2.0); // bit 0 → -1 * 2.0
        assert_eq!(result[2], -2.0); // bit 0 → -1 * 2.0
        assert_eq!(result[3], -2.0); // bit 0 → -1 * 2.0
        assert_eq!(result[4], 2.0); // bit 1 → +1 * 2.0
        assert_eq!(result[5], 2.0); // bit 1 → +1 * 2.0
        assert_eq!(result[6], -2.0); // bit 0 → -1 * 2.0
        assert_eq!(result[7], 2.0); // bit 1 → +1 * 2.0
    }

    #[test]
    fn test_binary_roundtrip() {
        let weights = vec![1.0, -1.0, 0.0, 0.5, -0.5, 2.0, -3.0, 0.1];
        let (packed, scale) = quantize_binary(&weights);
        let reconstructed = dequantize_binary(&packed, scale);

        // Check signs match (magnitude is all scale)
        for (orig, recon) in weights.iter().zip(reconstructed.iter()) {
            if orig.abs() > 1e-8 {
                assert_eq!(recon.signum(), orig.signum());
            }
        }
    }

    #[test]
    fn test_binary_block_pack_unpack() {
        let weights: Vec<bool> = (0..128).map(|i| (i % 3 == 0)).collect();
        let packed = pack_binary_block(&weights);
        assert_eq!(packed.len(), 18); // 2 FP16 + 16 bits

        let unpacked = unpack_binary_block(&packed);
        assert_eq!(weights, unpacked);
    }

    #[test]
    fn test_binary_block_multiple_blocks() {
        let weights: Vec<bool> = (0..256).map(|i| (i % 2 == 0)).collect();
        let packed = pack_binary_block(&weights);
        assert_eq!(packed.len(), 36); // 2 blocks * 18 bytes

        let unpacked = unpack_binary_block(&packed);
        assert_eq!(weights.len(), unpacked.len());
        assert_eq!(weights, unpacked);
    }

    #[test]
    fn test_binary_block_fp16_scale() {
        let weights: Vec<bool> = vec![true; 128];
        let packed = pack_binary_block(&weights);

        // Check scale is stored as FP16 1.0 (0x3C00 little-endian)
        assert_eq!(packed[0], 0x00);
        assert_eq!(packed[1], 0x3C);
    }
}
