//! Q2_0 quantization (PrismML Bonsai format, GGML type id 42).
//!
//! Block: 64 weights, 18 bytes = fp16 scale + 16 bytes of packed 2-bit codes.
//! Each byte holds 4 codes, little-endian bit order: code i = (byte >> (2*i)) & 3.
//! Code -> value: {0,1,2,3} -> {-1,0,1,2}. Value * scale gives the weight.

use kore_core::{KoreError, Tensor};

pub const Q2_0_BLOCK: usize = 64;
pub const Q2_0_BLOCK_BYTES: usize = 18;
pub const Q2_0_TYPE_ID: u32 = 42;

/// fp16 -> f32 without the `half` crate.
fn fp16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let f = if exp == 0 {
        (mant as f32) * (2f32.powi(-24))
    } else if exp == 31 {
        if mant == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1024.0 + mant as f32) * (2f32.powi(exp as i32 - 25))
    };
    if sign == 1 { -f } else { f }
}

/// Dequantize `n` elements of Q2_0 data into f32.
pub fn dequantize_q2_0(data: &[u8], n: usize) -> Result<Vec<f32>, KoreError> {
    let nb = n.div_ceil(Q2_0_BLOCK);
    if data.len() < nb * Q2_0_BLOCK_BYTES {
        return Err(KoreError::StorageError(format!(
            "q2_0 data too small: need {} bytes for {n} elements, got {}",
            nb * Q2_0_BLOCK_BYTES,
            data.len()
        )));
    }
    let mut out = vec![0.0f32; n];
    for b in 0..nb {
        let blk = &data[b * Q2_0_BLOCK_BYTES..];
        let d = fp16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for i in 0..16 {
            let byte = blk[2 + i];
            for j in 0..4 {
                let idx = b * 64 + i * 4 + j;
                if idx >= n {
                    break;
                }
                out[idx] = (((byte >> (2 * j)) & 3) as i32 - 1) as f32 * d;
            }
        }
    }
    Ok(out)
}

/// Packed GEMV: out[n] = sum_m w[m, n] * x[m], weights stored GGUF-style
/// [m, n] with m contiguous: `n` rows of `m` packed Q2_0 values each.
pub fn gemv_q2_0(w: &[u8], m: usize, n: usize, x: &[f32], out: &mut [f32]) -> Result<(), KoreError> {
    if m % Q2_0_BLOCK != 0 {
        return Err(KoreError::StorageError(format!("q2_0 in-dim {m} not multiple of 64")));
    }
    let bpr = m / Q2_0_BLOCK;
    let row_bytes = bpr * Q2_0_BLOCK_BYTES;
    if w.len() < n * row_bytes {
        return Err(KoreError::StorageError(format!(
            "q2_0 weight too small: need {} bytes for {m}x{n}, got {}",
            n * row_bytes,
            w.len()
        )));
    }
    for o in 0..n {
        let row = &w[o * row_bytes..(o + 1) * row_bytes];
        let mut acc = 0.0f32;
        for b in 0..bpr {
            let blk = &row[b * Q2_0_BLOCK_BYTES..];
            let d = fp16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            let mut sum = 0.0f32;
            let xb = &x[b * Q2_0_BLOCK..];
            for i in 0..16 {
                let byte = blk[2 + i];
                sum += ((byte & 3) as i32 - 1) as f32 * xb[i * 4];
                sum += (((byte >> 2) & 3) as i32 - 1) as f32 * xb[i * 4 + 1];
                sum += (((byte >> 4) & 3) as i32 - 1) as f32 * xb[i * 4 + 2];
                sum += (((byte >> 6) & 3) as i32 - 1) as f32 * xb[i * 4 + 3];
            }
            acc += d * sum;
        }
        out[o] = acc;
    }
    Ok(())
}

/// Convenience: GEMV returning a Tensor [1, n].
pub fn q2_0_matvec(w: &[u8], m: usize, n: usize, x: &[f32]) -> Result<Tensor, KoreError> {
    let mut out = vec![0.0f32; n];
    gemv_q2_0(w, m, n, x, &mut out)?;
    Ok(Tensor::from_f32(&out, &[1, n]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q2_0_dequant() {
        // scale = 2.0 fp16 = 0x4000; first byte = 0b11_10_01_00 -> {0,1,2,3} -> {-1,0,1,2}*2
        let mut blk = vec![0u8; 18];
        blk[0] = 0x00;
        blk[1] = 0x40;
        blk[2] = 0b11_10_01_00;
        let out = dequantize_q2_0(&blk, 64).unwrap();
        assert_eq!(out[0], -2.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 2.0);
        assert_eq!(out[3], 4.0);
        assert_eq!(out[4], 0.0); // zero byte -> all code 1 -> value 0
    }

    #[test]
    fn test_q2_0_gemv_identity() {
        // m=64, n=1: weights all code 0 (-1) with scale 1 -> out = -sum(x)
        let mut row = vec![0u8; 18];
        row[0] = 0x00;
        row[1] = 0x3C; // 1.0 fp16
        let x: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let mut out = [0.0f32; 1];
        gemv_q2_0(&row, 64, 1, &x, &mut out).unwrap();
        assert_eq!(out[0], -(0.0f32..64.0).sum::<f32>());
    }
}
