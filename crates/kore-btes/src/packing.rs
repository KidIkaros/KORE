//! Packing utilities for ternary and quaternary data.

use crate::encoder::{Trit, Quat, encode_trits, decode_trits, pack_quats, unpack_quats};

/// Pack a slice of trits into bytes using base-243 encoding.
/// 5 trits per byte, ~95% storage efficiency.
pub fn pack_ternary(trits: &[Trit]) -> Vec<u8> {
    let nbytes = (trits.len() + 4) / 5;
    let mut packed = Vec::with_capacity(nbytes);

    for chunk in trits.chunks(5) {
        let mut block = [Trit::Zero; 5];
        for (i, &t) in chunk.iter().enumerate() {
            block[i] = t;
        }
        packed.push(encode_trits(&block));
    }

    packed
}

/// Unpack bytes into trits. Returns exactly `num_trits` values.
pub fn unpack_ternary(packed: &[u8], num_trits: usize) -> Vec<Trit> {
    let mut trits = Vec::with_capacity(num_trits);
    for &byte in packed {
        let block = decode_trits(byte);
        for &t in &block {
            if trits.len() >= num_trits {
                break;
            }
            trits.push(t);
        }
    }
    trits.truncate(num_trits);
    trits
}

/// Pack a slice of quaternary values into bytes.
/// 4 values per byte (2 bits each), 4x compression.
pub fn pack_quaternary(quats: &[Quat]) -> Vec<u8> {
    let nbytes = (quats.len() + 3) / 4;
    let mut packed = Vec::with_capacity(nbytes);

    for chunk in quats.chunks(4) {
        let mut block = [Quat::Pos1; 4]; // default to +1 (index 2)
        for (i, &q) in chunk.iter().enumerate() {
            block[i] = q;
        }
        packed.push(pack_quats(&block));
    }

    packed
}

/// Unpack bytes into quaternary values. Returns exactly `num_quats` values.
pub fn unpack_quaternary(packed: &[u8], num_quats: usize) -> Vec<Quat> {
    let mut quats = Vec::with_capacity(num_quats);
    for &byte in packed {
        let block = unpack_quats(byte);
        for &q in &block {
            if quats.len() >= num_quats {
                break;
            }
            quats.push(q);
        }
    }
    quats.truncate(num_quats);
    quats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_pack_roundtrip() {
        let trits = vec![
            Trit::Pos, Trit::Neg, Trit::Zero, Trit::Pos, Trit::Neg,
            Trit::Zero, Trit::Pos, Trit::Pos,
        ];
        let packed = pack_ternary(&trits);
        let unpacked = unpack_ternary(&packed, trits.len());
        assert_eq!(trits, unpacked);
    }

    #[test]
    fn test_quaternary_pack_roundtrip() {
        let quats = vec![
            Quat::Neg3, Quat::Neg1, Quat::Pos1, Quat::Pos3,
            Quat::Pos1, Quat::Neg3,
        ];
        let packed = pack_quaternary(&quats);
        let unpacked = unpack_quaternary(&packed, quats.len());
        assert_eq!(quats, unpacked);
    }

    #[test]
    fn test_ternary_compression_ratio() {
        // 100 trits should pack into 20 bytes
        let trits = vec![Trit::Pos; 100];
        let packed = pack_ternary(&trits);
        assert_eq!(packed.len(), 20);
        // Naive: 100 bytes. Packed: 20 bytes. 5x compression.
    }

    #[test]
    fn test_quaternary_compression_ratio() {
        // 100 quats should pack into 25 bytes
        let quats = vec![Quat::Pos1; 100];
        let packed = pack_quaternary(&quats);
        assert_eq!(packed.len(), 25);
        // Naive: 100 bytes (or 400 for f32). Packed: 25 bytes. 16x vs f32.
    }
}
