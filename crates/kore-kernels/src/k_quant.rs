//! K-quant block geometry and reference dequantization (Q4_K / Q5_K / Q6_K).
//!
//! The K-quants are what real HuggingFace GGUFs actually ship. A file named
//! `Q4_K_M` is a *mix*: llama.cpp's recipe puts most matrices in Q4_K, some in
//! Q6_K, and some in Q5_K. (For example `unsloth/Ornith-1.0-9B-GGUF` is
//! 24 x Q5_K, 32 x Q6_K and the rest Q4_K.)
//!
//! This module is the *specification* for those layouts, expressed once so that
//! the packed compute kernels have something independent to be checked against.
//! `tests/k_quant_fixtures.rs` compares these functions bit-for-bit with
//! expected outputs produced by `gguf.quants`, the reference implementation
//! shipped with llama.cpp's Python tooling -- so a bug here cannot hide behind
//! a matching bug in the SIMD/WGSL kernels.
//!
//! All three types are super-blocks of 256 weights:
//!
//! | type | bytes/block | layout |
//! |------|-------------|--------|
//! | Q4_K | 144 | `d:f16, dmin:f16, scales[12], qs[128]` |
//! | Q5_K | 176 | `d:f16, dmin:f16, scales[12], qh[32], qs[128]` |
//! | Q6_K | 210 | `ql[128], qh[64], scales[16]:i8, d:f16` |
//!
//! Q4_K and Q5_K are affine: `w = d * sc[g] * q - dmin * mn[g]`, with a 6-bit
//! scale `sc` and a 6-bit min `mn` per 32-weight sub-block, both packed into the
//! same 12 `scales` bytes. Q5_K carries a 5th bit per weight in `qh`. Q6_K is
//! signed: `w = d * sc[s] * (q - 32)`, with `q` in `0..=63` and a signed 8-bit
//! scale per 16-weight sub-block.

use half::f16;
use kore_core::KoreError;

/// Weights per K-quant super-block.
pub const QK_K: usize = 256;

/// GGML type ids, matching [`crate::gguf`]-style readers (`gguf_min::type_block`).
pub const Q4_K_TYPE_ID: u32 = 12;
pub const Q5_K_TYPE_ID: u32 = 13;
pub const Q6_K_TYPE_ID: u32 = 14;

pub const Q4_K_BLOCK_BYTES: usize = 144;
pub const Q5_K_BLOCK_BYTES: usize = 176;
pub const Q6_K_BLOCK_BYTES: usize = 210;

/// Number of packed bytes per block for a K-quant type id, or `None` if the id
/// is not one this module knows.
pub const fn block_bytes(type_id: u32) -> Option<usize> {
    match type_id {
        Q4_K_TYPE_ID => Some(Q4_K_BLOCK_BYTES),
        Q5_K_TYPE_ID => Some(Q5_K_BLOCK_BYTES),
        Q6_K_TYPE_ID => Some(Q6_K_BLOCK_BYTES),
        _ => None,
    }
}

fn err(msg: impl Into<String>) -> KoreError {
    KoreError::StorageError(msg.into())
}

/// Number of blocks needed for `n` elements, validating that `data` is long
/// enough to actually hold them.
fn block_count(type_name: &str, data: &[u8], n: usize, block_bytes: usize) -> Result<usize, KoreError> {
    if block_bytes == 0 {
        return Err(err(format!("{type_name}: zero-size block")));
    }
    let nb = n.div_ceil(QK_K);
    let need = nb * block_bytes;
    if data.len() < need {
        return Err(err(format!(
            "{type_name}: payload too small: {} bytes < {need} needed for {n} elements \
             ({nb} blocks of {block_bytes} bytes)",
            data.len()
        )));
    }
    Ok(nb)
}

/// Unpack the eight scales and eight mins that Q4_K and Q5_K pack into 12 bytes.
///
/// Byte layout (this is `ggml`'s `get_scale_min_k4`):
///
/// ```text
///  0 EEAAAAAA    4 eeaaaaaa    8 eeeeEEEE
///  1 FFBBBBBB    5 ffbbbbbb    9 ffffFFFF
///  2 GGCCCCCC    6 ggcccccc   10 ggggGGGG
///  3 HHDDDDDD    7 hhdddddd   11 hhhhHHHH
/// ```
///
/// `A..H` are the six-bit scales and `a..h` the six-bit mins for the eight
/// 32-weight sub-blocks; the `E..H` / `e..h` in the high nibbles of bytes 0..4
/// and 8..12 are the *upper* two bits of those values.
pub fn unpack_scales_min(scales: &[u8]) -> ([f32; 8], [f32; 8]) {
    debug_assert!(scales.len() >= 12);
    let mut sc = [0.0f32; 8];
    let mut mn = [0.0f32; 8];

    // Sub-blocks 0..4: the low four bytes are scales, the next four are mins.
    for j in 0..4 {
        sc[j] = (scales[j] & 0x3F) as f32;
        mn[j] = (scales[j + 4] & 0x3F) as f32;
    }
    // Sub-blocks 4..8: one nibble each in bytes 8..12, plus two high bits
    // spliced in from bytes 0..4 (scales) and 4..8 (mins).
    for j in 4..8 {
        sc[j] = ((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 2) & 0x30)) as f32;
        mn[j] = (((scales[j + 4] >> 4) & 0x0F) | ((scales[j] >> 2) & 0x30)) as f32;
    }
    (sc, mn)
}

/// Dequantize `n` elements of packed Q4_K. `n` need not be a multiple of 256;
/// the tail block is truncated, which is how GGUF pads the final row.
pub fn dequantize_q4_k(data: &[u8], n: usize) -> Result<Vec<f32>, KoreError> {
    let nb = block_count("Q4_K", data, n, Q4_K_BLOCK_BYTES)?;
    let mut out = vec![0.0f32; nb * QK_K];

    for b in 0..nb {
        let blk = &data[b * Q4_K_BLOCK_BYTES..(b + 1) * Q4_K_BLOCK_BYTES];
        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let dmin = f16::from_le_bytes([blk[2], blk[3]]).to_f32();
        let (sc, mn) = unpack_scales_min(&blk[4..16]);
        let qs = &blk[16..Q4_K_BLOCK_BYTES];
        let base = b * QK_K;

        // Eight sub-blocks of 32 weights. Sub-block `g` reads the 32 bytes of
        // `qs` at `(g / 2) * 32`; even `g` takes the low nibble, odd `g` the
        // high nibble.
        for g in 0..8 {
            let row = &qs[(g / 2) * 32..(g / 2) * 32 + 32];
            let shift = ((g % 2) * 4) as u32;
            let d_sc = d * sc[g];
            let d_mn = dmin * mn[g];
            for (l, &byte) in row.iter().enumerate() {
                let q = ((byte >> shift) & 0x0F) as f32;
                out[base + g * 32 + l] = d_sc * q - d_mn;
            }
        }
    }

    out.truncate(n);
    Ok(out)
}

/// Dequantize `n` elements of packed Q5_K.
///
/// Identical to Q4_K except that each weight carries a fifth bit: `qh[l]` holds
/// one high bit per weight *position* `l`, with bit `g` belonging to sub-block
/// `g`.
pub fn dequantize_q5_k(data: &[u8], n: usize) -> Result<Vec<f32>, KoreError> {
    let nb = block_count("Q5_K", data, n, Q5_K_BLOCK_BYTES)?;
    let mut out = vec![0.0f32; nb * QK_K];

    for b in 0..nb {
        let blk = &data[b * Q5_K_BLOCK_BYTES..(b + 1) * Q5_K_BLOCK_BYTES];
        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let dmin = f16::from_le_bytes([blk[2], blk[3]]).to_f32();
        let (sc, mn) = unpack_scales_min(&blk[4..16]);
        let qh = &blk[16..48];
        let qs = &blk[48..Q5_K_BLOCK_BYTES];
        let base = b * QK_K;

        for g in 0..8 {
            let row = &qs[(g / 2) * 32..(g / 2) * 32 + 32];
            let shift = ((g % 2) * 4) as u32;
            let d_sc = d * sc[g];
            let d_mn = dmin * mn[g];
            for (l, &byte) in row.iter().enumerate() {
                let lo = (byte >> shift) & 0x0F;
                let hi = (qh[l] >> g) & 0x01;
                out[base + g * 32 + l] = d_sc * (lo | (hi << 4)) as f32 - d_mn;
            }
        }
    }

    out.truncate(n);
    Ok(out)
}

/// Dequantize `n` elements of packed Q6_K.
pub fn dequantize_q6_k(data: &[u8], n: usize) -> Result<Vec<f32>, KoreError> {
    let nb = block_count("Q6_K", data, n, Q6_K_BLOCK_BYTES)?;
    let mut out = vec![0.0f32; nb * QK_K];

    for b in 0..nb {
        let blk = &data[b * Q6_K_BLOCK_BYTES..(b + 1) * Q6_K_BLOCK_BYTES];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let scales = &blk[192..208];
        let d = f16::from_le_bytes([blk[208], blk[209]]).to_f32();
        let base = b * QK_K;

        for blk in 0..2 {
            // `ql` holds 128 bytes per block, two nibbles each -> 256 weights.
            // The 64-byte half-block alternates low/high nibble, and `qh`
            // supplies two high bits per weight in 32-byte halves.
            let ql_blk = &ql[blk * 64..blk * 64 + 64];
            let qh_blk = &qh[blk * 32..blk * 32 + 32];
            for k in 0..128 {
                let i = blk * 128 + k;
                let sc_i = scales[i / 16] as i8 as f32;
                let lo = (ql_blk[k % 64] >> (4 * ((k / 64) % 2))) & 0x0F;
                let hi = (qh_blk[k % 32] >> (((k / 32) % 4) * 2)) & 0x03;
                let q = (lo | (hi << 4)) as i32 - 32;
                out[base + i] = d * sc_i * q as f32;
            }
        }
    }

    out.truncate(n);
    Ok(out)
}

/// Dequantize `n` elements of any K-quant by GGML type id.
///
/// Returns an error for ids this module does not implement (notably Q2_K and
/// Q3_K, which are sized by the reader but have no kernel anywhere in KORE yet).
pub fn dequantize(type_id: u32, data: &[u8], n: usize) -> Result<Vec<f32>, KoreError> {
    match type_id {
        Q4_K_TYPE_ID => dequantize_q4_k(data, n),
        Q5_K_TYPE_ID => dequantize_q5_k(data, n),
        Q6_K_TYPE_ID => dequantize_q6_k(data, n),
        other => Err(err(format!(
            "k_quant: no dequantizer for GGML type {other} (only Q4_K=12, Q5_K=13, Q6_K=14)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_unpack_matches_ggml_bit_layout() {
        // Construct the documented 12-byte layout by hand and check the decode.
        //   sc[0] = A = 0b111111, sc[4] = low nibble 0b1010 | high bits 0b11
        //   mn[0] = a = 0b000001, mn[4] = low nibble 0b0101 | high bits 0b10
        let mut s = [0u8; 12];
        s[0] = 0b11_111111; // A = 63, plus EE = 0b11
        s[4] = 0b00_000001; // a = 1
        s[8] = 0b0010_1010; // sc[4] low nibble = 0b1010, mn[4] low nibble = 0b0010 >> ...
        // Bytes 8..12 hold both the low nibbles and, in their high nibbles, the
        // mins' upper bits; the scales' upper bits live in bytes 0..4 (bits 6..8).
        let (sc, mn) = unpack_scales_min(&s);
        assert_eq!(sc[0], 63.0);
        assert_eq!(mn[0], 1.0);
        // sc[4] = (s[8] & 0x0F) | ((s[0] >> 2) & 0x30) = 0b1010 | 0b110000 = 58
        assert_eq!(sc[4], (0b1010 | 0b110000) as f32);
        // mn[4] = ((s[8] >> 4) & 0x0F) | ((s[4] >> 2) & 0x30) = 0b0010 | 0 = 2
        assert_eq!(mn[4], 2.0);
    }

    #[test]
    fn zero_block_dequantizes_to_zero() {
        for (id, bytes) in [
            (Q4_K_TYPE_ID, Q4_K_BLOCK_BYTES),
            (Q5_K_TYPE_ID, Q5_K_BLOCK_BYTES),
            (Q6_K_TYPE_ID, Q6_K_BLOCK_BYTES),
        ] {
            let out = dequantize(id, &vec![0u8; bytes], QK_K).unwrap();
            assert!(out.iter().all(|v| *v == 0.0), "type {id} zero block nonzero");
        }
    }

    #[test]
    fn short_payload_is_an_error_not_a_panic() {
        let e = dequantize_q4_k(&[0u8; 100], QK_K).unwrap_err();
        assert!(format!("{e}").contains("payload too small"));
        let e = dequantize(11, &[0u8; 256], QK_K).unwrap_err(); // Q3_K
        assert!(format!("{e}").contains("no dequantizer"));
    }

    #[test]
    fn nonzero_f16_scale_scales_output() {
        // Q6_K: d is the final f16. 1.0 == 0x3C00. With an all-zero rest only
        // scale sub-block 0 is nonzero, so the first 16 weights are d*1*(0-32)
        // and everything else is d*0*(...) == 0.
        let mut blk = [0u8; Q6_K_BLOCK_BYTES];
        blk[208] = 0x00;
        blk[209] = 0x3C; // f16 1.0
        blk[192] = 1; // scales[0] = 1
        let out = dequantize_q6_k(&blk, QK_K).unwrap();
        for (i, v) in out.iter().enumerate() {
            let want = if i < 16 { -32.0 } else { 0.0 };
            assert_eq!(*v, want, "index {i}");
        }
    }

    #[test]
    fn q6_k_reads_all_six_bits() {
        // qh supplies two high bits per weight; verify the bit plane selection
        // by giving a single weight the value 63 (all six bits set) and
        // checking it lands where the layout says.
        let mut blk = [0u8; Q6_K_BLOCK_BYTES];
        blk[208] = 0x00;
        blk[209] = 0x3C; // d = 1.0
        blk[192] = 1; // scales[0] = 1 (applies to indices 0..16)
        // All 63 for the first 16 weights: ql nibbles 0xF, qh bits 0b11.
        blk[..8].fill(0xFF); // ql low-nibble plane for indices 0..8
        // qh[0..8] carry the high bits for the 0..64 weight range; bit pairs
        // 0b11 for plane2 = 0 (indices 0..8) and plane2 = 1 (indices 32..40).
        for b in blk[128..136].iter_mut() {
            *b = 0b0000_0011;
        }
        let out = dequantize_q6_k(&blk, QK_K).unwrap();
        // First 8 weights: ql 0xF, qh bits = 0b11 -> q = 63 -> 63 - 32 = 31.
        for (i, v) in out.iter().enumerate().take(8) {
            assert_eq!(*v, 31.0, "index {i}");
        }
    }
}
