//! Virtual Ternary ALU — 64-trit parallel arithmetic via bit-slicing.
//!
//! Ported from `btes/src/vtalu.c`. Performs balanced ternary arithmetic
//! on 64 trits simultaneously using only bitwise operations (no branches).
//!
//! ## Bit-Slice Encoding
//!
//! | plane_a | plane_b | Trit |
//! |---------|---------|------|
//! |    0    |    0    |   0  |
//! |    0    |    1    |  +1  |
//! |    1    |    0    |  -1  |
//! |    1    |    1    | invalid |

/// 64 balanced ternary trits packed into two `u64` bit-planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TernaryWord64 {
    /// Negative plane: bit set means trit is -1 (when plane_b is 0).
    pub plane_a: u64,
    /// Positive plane: bit set means trit is +1 (when plane_a is 0).
    pub plane_b: u64,
}

// ============================================================================
// Constructors & Validation
// ============================================================================

impl TernaryWord64 {
    /// All 64 trits set to zero.
    pub const ZERO: Self = Self { plane_a: 0, plane_b: 0 };

    /// Create a word with all trits set to zero.
    #[inline]
    pub fn zero() -> Self {
        Self::ZERO
    }

    /// Create a word with all 64 trits set to the given value (-1, 0, or +1).
    #[inline]
    pub fn fill(trit: i8) -> Self {
        match trit {
            1 => Self { plane_a: 0, plane_b: u64::MAX },
            -1 => Self { plane_a: u64::MAX, plane_b: 0 },
            _ => Self::ZERO,
        }
    }

    /// Count invalid trits (positions where both planes are 1).
    #[inline]
    pub fn count_invalid(self) -> u32 {
        (self.plane_a & self.plane_b).count_ones()
    }

    /// Clear invalid states to zero.
    #[inline]
    pub fn sanitize(self) -> Self {
        let invalid = self.plane_a & self.plane_b;
        Self {
            plane_a: self.plane_a & !invalid,
            plane_b: self.plane_b & !invalid,
        }
    }

    /// Check if all trits are valid (no position has both planes set).
    #[inline]
    pub fn is_valid(self) -> bool {
        (self.plane_a & self.plane_b) == 0
    }

    /// Count non-zero trits.
    #[inline]
    pub fn popcount(self) -> u32 {
        (self.plane_a | self.plane_b).count_ones()
    }
}

// ============================================================================
// Unary Operations
// ============================================================================

impl TernaryWord64 {
    /// Negate all trits: -1 <-> +1, 0 -> 0. Swap planes.
    #[inline]
    pub fn negate(self) -> Self {
        Self {
            plane_a: self.plane_b,
            plane_b: self.plane_a,
        }
    }

    /// Absolute value: -1 -> +1, +1 -> +1, 0 -> 0.
    #[inline]
    pub fn abs(self) -> Self {
        Self {
            plane_a: 0,
            plane_b: self.plane_a | self.plane_b,
        }
    }

    /// Sign function (identity for valid trits, sanitizes invalid).
    #[inline]
    pub fn sign(self) -> Self {
        self.sanitize()
    }
}

// ============================================================================
// Binary Logic Operations (Element-wise)
// ============================================================================

impl TernaryWord64 {
    /// Ternary MIN: result is -1 if either is -1, else 0 if either is 0, else +1.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        let pa = self.plane_a | other.plane_a;
        Self {
            plane_a: pa,
            plane_b: self.plane_b & other.plane_b & !pa,
        }
    }

    /// Ternary MAX: result is +1 if either is +1, else 0 if either is 0, else -1.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        let pb = self.plane_b | other.plane_b;
        Self {
            plane_a: self.plane_a & other.plane_a & !pb,
            plane_b: pb,
        }
    }

    /// Consensus: returns value if both agree, else 0.
    #[inline]
    pub fn consensus(self, other: Self) -> Self {
        let agree = !(self.plane_a ^ other.plane_a) & !(self.plane_b ^ other.plane_b);
        Self {
            plane_a: self.plane_a & agree,
            plane_b: self.plane_b & agree,
        }
    }
}

// ============================================================================
// Arithmetic Operations
// ============================================================================

impl TernaryWord64 {
    /// Balanced ternary addition without carry propagation.
    /// Returns (sum, carry) where carry needs to be shifted left and added.
    #[inline]
    pub fn add_no_carry(self, other: Self) -> (Self, Self) {
        let a_neg = self.plane_a & !self.plane_b;
        let a_zero = !self.plane_a & !self.plane_b;
        let a_pos = !self.plane_a & self.plane_b;

        let b_neg = other.plane_a & !other.plane_b;
        let b_zero = !other.plane_a & !other.plane_b;
        let b_pos = !other.plane_a & other.plane_b;

        // Sum: -1 when (a=-1,b=0)|(a=0,b=-1)|(a=+1,b=+1)
        //      +1 when (a=+1,b=0)|(a=0,b=+1)|(a=-1,b=-1)
        let sum_neg = (a_neg & b_zero) | (a_zero & b_neg) | (a_pos & b_pos);
        let sum_pos = (a_pos & b_zero) | (a_zero & b_pos) | (a_neg & b_neg);

        let sum = Self { plane_a: sum_neg, plane_b: sum_pos };

        // Carry: -1 when (a=-1,b=-1), +1 when (a=+1,b=+1)
        let carry = Self {
            plane_a: a_neg & b_neg,
            plane_b: a_pos & b_pos,
        };

        (sum, carry)
    }

    /// Full balanced ternary addition with carry propagation.
    /// Returns (sum, final_carry).
    pub fn add(self, other: Self, carry_in: i8) -> (Self, i8) {
        let (mut sum, mut carry) = self.add_no_carry(other);

        // Add initial carry_in at LSB
        if carry_in != 0 {
            let carry_word = if carry_in > 0 {
                Self { plane_a: 0, plane_b: 1 }
            } else {
                Self { plane_a: 1, plane_b: 0 }
            };
            let (s2, c2) = sum.add_no_carry(carry_word);
            sum = s2;
            carry.plane_a |= c2.plane_a;
            carry.plane_b |= c2.plane_b;
        }

        // Ripple carry propagation
        for _ in 0..64 {
            if carry.plane_a == 0 && carry.plane_b == 0 {
                break;
            }
            let shifted = Self {
                plane_a: carry.plane_a << 1,
                plane_b: carry.plane_b << 1,
            };
            let (s, c) = sum.add_no_carry(shifted);
            sum = s;
            carry = c;
        }

        // Extract final carry from MSB overflow
        let carry_out = if carry.plane_a & (1u64 << 63) != 0 {
            -1
        } else if carry.plane_b & (1u64 << 63) != 0 {
            1
        } else {
            0
        };

        (sum, carry_out)
    }

    /// Subtraction: a - b = a + (-b).
    #[inline]
    pub fn sub(self, other: Self, borrow_in: i8) -> (Self, i8) {
        self.add(other.negate(), -borrow_in)
    }

    /// Element-wise ternary multiplication.
    /// Product is 0 if either is 0, +1 if signs match, -1 if signs differ.
    #[inline]
    pub fn mul_elementwise(self, other: Self) -> Self {
        let nz_a = self.plane_a | self.plane_b;
        let nz_b = other.plane_a | other.plane_b;
        let nz = nz_a & nz_b;

        let signs_match = (self.plane_a & other.plane_a) | (self.plane_b & other.plane_b);

        Self {
            plane_a: nz & !signs_match,
            plane_b: nz & signs_match,
        }
    }
}

// ============================================================================
// Comparison Operations
// ============================================================================

impl TernaryWord64 {
    /// Element-wise equality: +1 where equal, 0 where different.
    #[inline]
    pub fn eq_trit(self, other: Self) -> Self {
        let equal = !(self.plane_a ^ other.plane_a) & !(self.plane_b ^ other.plane_b);
        Self { plane_a: 0, plane_b: equal }
    }

    /// Element-wise comparison: sign of (a - b) per position.
    #[inline]
    pub fn cmp_trit(self, other: Self) -> Self {
        let (diff, _) = self.sub(other, 0);
        diff.sign()
    }
}

// ============================================================================
// Conversion Utilities
// ============================================================================

impl TernaryWord64 {
    /// Get a single trit at the given index (0-63, LSB first).
    #[inline]
    pub fn get_trit(self, index: usize) -> i8 {
        if index >= 64 { return 0; }
        let mask = 1u64 << index;
        let a = (self.plane_a & mask) != 0;
        let b = (self.plane_b & mask) != 0;
        match (a, b) {
            (false, false) => 0,
            (false, true) => 1,
            (true, false) => -1,
            (true, true) => 0, // invalid → 0
        }
    }

    /// Set a single trit at the given index.
    #[inline]
    pub fn set_trit(&mut self, index: usize, trit: i8) {
        if index >= 64 { return; }
        let mask = 1u64 << index;
        let clear = !mask;
        self.plane_a &= clear;
        self.plane_b &= clear;
        match trit {
            1 => self.plane_b |= mask,
            -1 => self.plane_a |= mask,
            _ => {} // 0: both cleared
        }
    }

    /// Create from a slice of trits (up to 64).
    pub fn from_trits(trits: &[i8]) -> Self {
        let mut result = Self::ZERO;
        let count = trits.len().min(64);
        for i in 0..count {
            result.set_trit(i, trits[i]);
        }
        result
    }

    /// Convert to an array of 64 trits.
    pub fn to_trits(self) -> [i8; 64] {
        let mut trits = [0i8; 64];
        for i in 0..64 {
            trits[i] = self.get_trit(i);
        }
        trits
    }

    /// Dot product: sum of element-wise products (returns scalar i64).
    /// Useful for ternary matmul inner loops.
    pub fn dot(self, other: Self) -> i64 {
        let product = self.mul_elementwise(other);
        let pos_count = product.plane_b.count_ones() as i64;
        let neg_count = product.plane_a.count_ones() as i64;
        pos_count - neg_count
    }
}

// ============================================================================
// Std trait impls
// ============================================================================

impl std::ops::Neg for TernaryWord64 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self { self.negate() }
}

impl std::ops::BitAnd for TernaryWord64 {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self { self.mul_elementwise(rhs) }
}

impl Default for TernaryWord64 {
    fn default() -> Self { Self::ZERO }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        let z = TernaryWord64::zero();
        assert_eq!(z.plane_a, 0);
        assert_eq!(z.plane_b, 0);
        assert!(z.is_valid());
    }

    #[test]
    fn test_fill() {
        let pos = TernaryWord64::fill(1);
        assert_eq!(pos.plane_a, 0);
        assert_eq!(pos.plane_b, u64::MAX);

        let neg = TernaryWord64::fill(-1);
        assert_eq!(neg.plane_a, u64::MAX);
        assert_eq!(neg.plane_b, 0);
    }

    #[test]
    fn test_negate() {
        let pos = TernaryWord64::fill(1);
        let neg = pos.negate();
        assert_eq!(neg, TernaryWord64::fill(-1));
        assert_eq!(neg.negate(), pos);

        let z = TernaryWord64::zero();
        assert_eq!(z.negate(), z);
    }

    #[test]
    fn test_abs() {
        let neg = TernaryWord64::fill(-1);
        assert_eq!(neg.abs(), TernaryWord64::fill(1));

        let pos = TernaryWord64::fill(1);
        assert_eq!(pos.abs(), pos);

        let z = TernaryWord64::zero();
        assert_eq!(z.abs(), z);
    }

    #[test]
    fn test_get_set_trit() {
        let mut w = TernaryWord64::zero();
        w.set_trit(0, 1);
        w.set_trit(1, -1);
        w.set_trit(63, 1);

        assert_eq!(w.get_trit(0), 1);
        assert_eq!(w.get_trit(1), -1);
        assert_eq!(w.get_trit(2), 0);
        assert_eq!(w.get_trit(63), 1);
    }

    #[test]
    fn test_from_to_trits() {
        let input: Vec<i8> = vec![1, -1, 0, 1, -1, 0, 0, 1];
        let w = TernaryWord64::from_trits(&input);
        let trits = w.to_trits();

        for (i, &expected) in input.iter().enumerate() {
            assert_eq!(trits[i], expected, "trit {} mismatch", i);
        }
        for i in input.len()..64 {
            assert_eq!(trits[i], 0, "trit {} should be 0", i);
        }
    }

    #[test]
    fn test_mul_elementwise() {
        // (-1) * (-1) = +1, (-1) * (+1) = -1, (+1) * (+1) = +1, 0 * x = 0
        let a = TernaryWord64::from_trits(&[-1, -1, 1, 0, 1]);
        let b = TernaryWord64::from_trits(&[-1, 1, 1, 1, 0]);
        let c = a.mul_elementwise(b);
        let trits = c.to_trits();

        assert_eq!(trits[0], 1);   // (-1)*(-1)
        assert_eq!(trits[1], -1);  // (-1)*(+1)
        assert_eq!(trits[2], 1);   // (+1)*(+1)
        assert_eq!(trits[3], 0);   // (0)*(+1)
        assert_eq!(trits[4], 0);   // (+1)*(0)
    }

    #[test]
    fn test_add_simple() {
        // +1 + (-1) = 0
        let a = TernaryWord64::from_trits(&[1]);
        let b = TernaryWord64::from_trits(&[-1]);
        let (sum, carry) = a.add(b, 0);
        assert_eq!(sum.get_trit(0), 0);
        assert_eq!(carry, 0);
    }

    #[test]
    fn test_add_carry() {
        // +1 + +1 = -1 with carry +1 (i.e., result is 2 in balanced ternary = 1*3 + (-1))
        let a = TernaryWord64::from_trits(&[1]);
        let b = TernaryWord64::from_trits(&[1]);
        let (sum, _) = a.add(b, 0);
        assert_eq!(sum.get_trit(0), -1); // LSB
        assert_eq!(sum.get_trit(1), 1);  // carry propagated
    }

    #[test]
    fn test_sub() {
        let a = TernaryWord64::from_trits(&[1, 0, -1]);
        let b = TernaryWord64::from_trits(&[1, 0, -1]);
        let (diff, _) = a.sub(b, 0);
        assert_eq!(diff.get_trit(0), 0);
        assert_eq!(diff.get_trit(1), 0);
        assert_eq!(diff.get_trit(2), 0);
    }

    #[test]
    fn test_eq_trit() {
        let a = TernaryWord64::from_trits(&[1, -1, 0, 1]);
        let b = TernaryWord64::from_trits(&[1, 0, 0, -1]);
        let eq = a.eq_trit(b);

        assert_eq!(eq.get_trit(0), 1);  // both +1
        assert_eq!(eq.get_trit(1), 0);  // -1 vs 0
        assert_eq!(eq.get_trit(2), 1);  // both 0
        assert_eq!(eq.get_trit(3), 0);  // +1 vs -1
    }

    #[test]
    fn test_dot_product() {
        // [1, -1, 1] . [1, 1, -1] = 1*1 + (-1)*1 + 1*(-1) = 1 - 1 - 1 = -1
        let a = TernaryWord64::from_trits(&[1, -1, 1]);
        let b = TernaryWord64::from_trits(&[1, 1, -1]);
        assert_eq!(a.dot(b), -1);
    }

    #[test]
    fn test_dot_product_orthogonal() {
        // [1, -1, 0] . [0, 0, 1] = 0
        let a = TernaryWord64::from_trits(&[1, -1, 0]);
        let b = TernaryWord64::from_trits(&[0, 0, 1]);
        assert_eq!(a.dot(b), 0);
    }

    #[test]
    fn test_min_max() {
        let a = TernaryWord64::from_trits(&[-1, 0, 1, 0]);
        let b = TernaryWord64::from_trits(&[0, 1, 0, -1]);

        let mn = a.min(b);
        assert_eq!(mn.get_trit(0), -1);
        assert_eq!(mn.get_trit(1), 0);
        assert_eq!(mn.get_trit(2), 0);
        assert_eq!(mn.get_trit(3), -1);

        let mx = a.max(b);
        assert_eq!(mx.get_trit(0), 0);
        assert_eq!(mx.get_trit(1), 1);
        assert_eq!(mx.get_trit(2), 1);
        assert_eq!(mx.get_trit(3), 0);
    }

    #[test]
    fn test_consensus() {
        let a = TernaryWord64::from_trits(&[1, -1, 0, 1]);
        let b = TernaryWord64::from_trits(&[1, 0, 0, 1]);
        let c = a.consensus(b);

        assert_eq!(c.get_trit(0), 1);  // agree
        assert_eq!(c.get_trit(1), 0);  // disagree
        assert_eq!(c.get_trit(2), 0);  // agree (both 0)
        assert_eq!(c.get_trit(3), 1);  // agree
    }

    #[test]
    fn test_sanitize() {
        let invalid = TernaryWord64 { plane_a: 0b11, plane_b: 0b11 };
        assert_eq!(invalid.count_invalid(), 2);
        let clean = invalid.sanitize();
        assert!(clean.is_valid());
        assert_eq!(clean.plane_a, 0);
        assert_eq!(clean.plane_b, 0);
    }

    #[test]
    fn test_popcount() {
        let w = TernaryWord64::from_trits(&[1, -1, 0, 1, -1]);
        assert_eq!(w.popcount(), 4); // 4 non-zero trits
    }

    #[test]
    fn test_neg_trait() {
        let pos = TernaryWord64::fill(1);
        assert_eq!(-pos, TernaryWord64::fill(-1));
    }
}
