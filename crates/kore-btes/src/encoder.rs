//! Binary ↔ Ternary ↔ Quaternary conversion.
//!
//! Ternary: balanced {-1, 0, +1}, packed 5 trits per byte (base-243).
//! Quaternary: {-3, -1, +1, +3}, packed 4 values per byte (2 bits each).

/// Balanced ternary digit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum Trit {
    Neg = -1,
    Zero = 0,
    Pos = 1,
}

/// Quaternary digit (2-bit).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Quat {
    Neg3 = 0, // maps to -3
    Neg1 = 1, // maps to -1
    Pos1 = 2, // maps to +1
    Pos3 = 3, // maps to +3
}

impl Quat {
    /// Convert to float value.
    pub fn to_f32(self) -> f32 {
        match self {
            Quat::Neg3 => -3.0,
            Quat::Neg1 => -1.0,
            Quat::Pos1 => 1.0,
            Quat::Pos3 => 3.0,
        }
    }

    /// Convert from 2-bit index.
    pub fn from_index(idx: u8) -> Self {
        match idx & 0x3 {
            0 => Quat::Neg3,
            1 => Quat::Neg1,
            2 => Quat::Pos1,
            3 => Quat::Pos3,
            _ => unreachable!(),
        }
    }
}

/// Lookup table for base-243 ternary decoding.
/// Each byte (0..=242) maps to 5 trits.
pub const TERNARY_DECODE_LUT: [[Trit; 5]; 243] = {
    let mut lut = [[Trit::Zero; 5]; 243];
    let mut i = 0u16;
    while i < 243 {
        let mut val = i;
        let mut j = 0;
        while j < 5 {
            let rem = val % 3;
            lut[i as usize][j] = match rem {
                0 => Trit::Neg,
                1 => Trit::Zero,
                2 => Trit::Pos,
                _ => Trit::Zero,
            };
            val /= 3;
            j += 1;
        }
        i += 1;
    }
    lut
};

/// Encode 5 trits into a single byte (base-243).
pub fn encode_trits(trits: &[Trit; 5]) -> u8 {
    let mut val: u8 = 0;
    let mut multiplier: u8 = 1;
    for &t in trits {
        let digit = match t {
            Trit::Neg => 0,
            Trit::Zero => 1,
            Trit::Pos => 2,
        };
        val += digit * multiplier;
        multiplier = multiplier.wrapping_mul(3);
    }
    val
}

/// Decode a byte into 5 trits (base-243).
pub fn decode_trits(byte: u8) -> [Trit; 5] {
    if byte < 243 {
        TERNARY_DECODE_LUT[byte as usize]
    } else {
        [Trit::Zero; 5]
    }
}

/// Pack 4 quaternary values into a single byte.
pub fn pack_quats(quats: &[Quat; 4]) -> u8 {
    (quats[0] as u8) | ((quats[1] as u8) << 2) | ((quats[2] as u8) << 4) | ((quats[3] as u8) << 6)
}

/// Unpack a byte into 4 quaternary values.
pub fn unpack_quats(byte: u8) -> [Quat; 4] {
    [
        Quat::from_index(byte & 0x3),
        Quat::from_index((byte >> 2) & 0x3),
        Quat::from_index((byte >> 4) & 0x3),
        Quat::from_index((byte >> 6) & 0x3),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trit_encode_decode_roundtrip() {
        let trits = [Trit::Pos, Trit::Neg, Trit::Zero, Trit::Pos, Trit::Neg];
        let byte = encode_trits(&trits);
        let decoded = decode_trits(byte);
        assert_eq!(trits, decoded);
    }

    #[test]
    fn test_all_zeros() {
        let trits = [Trit::Zero; 5];
        let byte = encode_trits(&trits);
        // All zeros → 1+3+9+27+81 = 121
        assert_eq!(byte, 121);
        assert_eq!(decode_trits(byte), trits);
    }

    #[test]
    fn test_quat_pack_unpack() {
        let quats = [Quat::Neg3, Quat::Neg1, Quat::Pos1, Quat::Pos3];
        let byte = pack_quats(&quats);
        let unpacked = unpack_quats(byte);
        assert_eq!(quats, unpacked);
    }

    #[test]
    fn test_quat_to_f32() {
        assert_eq!(Quat::Neg3.to_f32(), -3.0);
        assert_eq!(Quat::Neg1.to_f32(), -1.0);
        assert_eq!(Quat::Pos1.to_f32(), 1.0);
        assert_eq!(Quat::Pos3.to_f32(), 3.0);
    }

    #[test]
    fn test_all_byte_values_decode() {
        for b in 0..243u8 {
            let trits = decode_trits(b);
            let re_encoded = encode_trits(&trits);
            assert_eq!(b, re_encoded, "Roundtrip failed for byte {}", b);
        }
    }
}
