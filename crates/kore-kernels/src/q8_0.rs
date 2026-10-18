//! The legacy 8-bit block: file layout, and the device layout.
//!
//! Q8_0 is llama.cpp's oldest quantized format and still ships in real files --
//! bartowski's `Qwen3.6-35B-A3B-Q3_K_S` carries 105 Q8_0 tensors, including the
//! shared-expert FFN, `ssm_out` and the full-attention `k`/`v` projections.
//!
//! A block is 32 weights under one `f16` scale:
//!
//! ```text
//! d    f16        2 bytes
//! qs   [i8; 32]  32 bytes      value = qs[i] * d
//! ```
//!
//! Unlike the K-quants this module is **not** the authority on the file layout:
//! the CPU lane decodes Q8_0 through `oxillama-quant`, whose reference kernel and
//! SIMD kernels are validated against the same fixtures. What lives here is only
//! the *device* adaptation, so that the GPU kernel's block reads are aligned.

use kore_core::KoreError;

/// GGML type id for Q8_0.
pub const Q8_0_TYPE_ID: u32 = 8;
/// Weights per block.
pub const Q8_0_BLOCK_SIZE: usize = 32;
/// Bytes per block in a GGUF file.
pub const Q8_0_BLOCK_BYTES: usize = 34;
/// Bytes per block as the GPU kernel reads them.
pub const Q8_0_PACKED_BYTES: usize = 36;

fn err(msg: impl Into<String>) -> KoreError {
    KoreError::StorageError(msg.into())
}

/// Repack every 34-byte Q8_0 block into 36 bytes, quants first.
///
/// The GPU kernel reads its weight buffer as `array<u32>`, and 34 is not a
/// multiple of four: consecutive blocks would begin at alternating half-word
/// offsets, so a lane's four quants could straddle two words. Two zero bytes per
/// block (5.9%) buy word alignment, and moving the scale *behind* the quants puts
/// a lane's quad at plain word `quad` instead of making it dodge the scale.
///
/// This is the same kind of upload-time adaptation as
/// [`crate::k_quant::repack_q3_k`] and [`crate::q2_0::repack_q2_0`]; it is
/// applied at load to resident weights only, and the file's own packing stays
/// authoritative for the CPU lane.
pub fn repack_q8_0(data: &[u8]) -> Result<Vec<u8>, KoreError> {
    if !data.len().is_multiple_of(Q8_0_BLOCK_BYTES) {
        return Err(err(format!(
            "q8_0: payload of {} bytes is not a whole number of {Q8_0_BLOCK_BYTES}-byte blocks",
            data.len()
        )));
    }
    let blocks = data.len() / Q8_0_BLOCK_BYTES;
    let mut out = vec![0u8; blocks * Q8_0_PACKED_BYTES];
    for b in 0..blocks {
        let src = &data[b * Q8_0_BLOCK_BYTES..(b + 1) * Q8_0_BLOCK_BYTES];
        let dst = &mut out[b * Q8_0_PACKED_BYTES..(b + 1) * Q8_0_PACKED_BYTES];
        dst[..Q8_0_BLOCK_SIZE].copy_from_slice(&src[2..Q8_0_BLOCK_BYTES]);
        dst[Q8_0_BLOCK_SIZE..Q8_0_BLOCK_SIZE + 2].copy_from_slice(&src[..2]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The repack must be lossless: a block decoded from the packed bytes has to
    /// equal the same block decoded from the file bytes, to the last bit.
    #[test]
    fn repack_preserves_every_block() {
        let mut s: u32 = 0x9E37_79B9;
        let mut next = || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 16) as u8
        };
        let blocks = 7;
        let file: Vec<u8> = (0..blocks * Q8_0_BLOCK_BYTES).map(|_| next()).collect();

        let packed = repack_q8_0(&file).expect("repack");
        assert_eq!(packed.len(), blocks * Q8_0_PACKED_BYTES);

        for b in 0..blocks {
            let f = &file[b * Q8_0_BLOCK_BYTES..(b + 1) * Q8_0_BLOCK_BYTES];
            let p = &packed[b * Q8_0_PACKED_BYTES..(b + 1) * Q8_0_PACKED_BYTES];
            let d = half::f16::from_le_bytes([f[0], f[1]]).to_f32();
            for i in 0..Q8_0_BLOCK_SIZE {
                let want = (f[2 + i] as i8) as f32 * d;
                let got = (p[i] as i8) as f32 * half::f16::from_le_bytes([p[32], p[33]]).to_f32();
                assert_eq!(want.to_bits(), got.to_bits(), "block {b} weight {i}");
            }
        }
    }

    #[test]
    fn repack_rejects_a_partial_block() {
        assert!(repack_q8_0(&[0u8; Q8_0_BLOCK_BYTES - 1]).is_err());
    }
}
