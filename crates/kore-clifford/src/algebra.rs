//! Clifford algebra Cl(p,q) definition and Cayley table generation.
//!
//! The algebra is defined by its signature (p, q) where:
//! - p basis vectors square to +1
//! - q basis vectors square to -1
//! - Total dimension n = p + q
//! - Algebra has 2^n basis blades

/// Sign result from multiplying two basis blades.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sign {
    Pos,
    Neg,
    Zero,
}

impl Sign {
    pub fn as_f32(self) -> f32 {
        match self {
            Sign::Pos => 1.0,
            Sign::Neg => -1.0,
            Sign::Zero => 0.0,
        }
    }

    pub fn flip(self) -> Self {
        match self {
            Sign::Pos => Sign::Neg,
            Sign::Neg => Sign::Pos,
            Sign::Zero => Sign::Zero,
        }
    }
}

impl std::ops::Mul for Sign {
    type Output = Sign;
    fn mul(self, rhs: Sign) -> Sign {
        match (self, rhs) {
            (Sign::Zero, _) | (_, Sign::Zero) => Sign::Zero,
            (Sign::Pos, s) | (s, Sign::Pos) => s,
            (Sign::Neg, Sign::Neg) => Sign::Pos,
        }
    }
}

/// Entry in the Cayley (multiplication) table.
#[derive(Debug, Clone, Copy)]
pub struct CayleyEntry {
    /// Resulting basis blade index.
    pub blade: usize,
    /// Sign of the product.
    pub sign: Sign,
}

/// A Clifford algebra Cl(p,q).
///
/// Stores the precomputed Cayley table for the geometric product.
/// The algebra has 2^(p+q) basis blades.
#[derive(Debug, Clone)]
pub struct CliffordAlgebra {
    /// Number of basis vectors squaring to +1.
    pub p: usize,
    /// Number of basis vectors squaring to -1.
    pub q: usize,
    /// Total number of basis vectors.
    pub n: usize,
    /// Total number of basis blades (2^n).
    pub dim: usize,
    /// Cayley table: cayley[i][j] = result of blade_i * blade_j.
    pub cayley: Vec<Vec<CayleyEntry>>,
    /// Grade of each basis blade.
    pub grades: Vec<usize>,
}

impl CliffordAlgebra {
    /// Create a new Clifford algebra Cl(p,q) and precompute the Cayley table.
    pub fn new(p: usize, q: usize) -> Self {
        let n = p + q;
        let dim = 1 << n; // 2^n

        // Compute grades (popcount of blade index)
        let grades: Vec<usize> = (0..dim).map(|i| (i as u32).count_ones() as usize).collect();

        // Build Cayley table
        let mut cayley = vec![vec![CayleyEntry { blade: 0, sign: Sign::Zero }; dim]; dim];

        for (i, c_row) in cayley.iter_mut().enumerate() {
            for (j, entry) in c_row.iter_mut().enumerate() {
                *entry = Self::multiply_blades(i, j, p, q, n);
            }
        }

        Self { p, q, n, dim, cayley, grades }
    }

    /// Multiply two basis blades represented as bitmasks.
    ///
    /// Each bit in the bitmask represents a basis vector.
    /// e.g., for Cl(3,0): e1=0b001, e2=0b010, e3=0b100, e12=0b011, etc.
    fn multiply_blades(a: usize, b: usize, p: usize, _q: usize, n: usize) -> CayleyEntry {
        let result_blade = a ^ b; // XOR gives the resulting blade

        // Count sign flips from reordering basis vectors
        let mut sign = Sign::Pos;

        // For each bit in b, count how many bits in a are to the right
        // (these need to be swapped past, each swap flips sign)
        for i in 0..n {
            if (b >> i) & 1 == 1 {
                // Count bits in a that are at positions > i
                for j in (i + 1)..n {
                    if (a >> j) & 1 == 1 {
                        sign = sign.flip();
                    }
                }
            }
        }

        // Handle squares of basis vectors
        // Shared bits (a & b) represent basis vectors that appear twice
        let shared = a & b;
        for i in 0..n {
            if (shared >> i) & 1 == 1 {
                if i >= p {
                    // This basis vector squares to -1 (it's in the q part)
                    sign = sign.flip();
                }
                // Basis vectors squaring to +1 don't change sign

                // Count bits in a that are between this shared bit and its pair
                // (additional swaps needed to bring the pair together)
                for j in (i + 1)..n {
                    if (a >> j) & 1 == 1 && (shared >> j) & 1 == 0 {
                        // Non-shared bit in a between shared bits
                    }
                }
            }
        }

        CayleyEntry { blade: result_blade, sign }
    }

    /// Get the grade of a basis blade.
    pub fn grade(&self, blade: usize) -> usize {
        self.grades[blade]
    }

    /// Get all blade indices of a specific grade.
    pub fn blades_of_grade(&self, grade: usize) -> Vec<usize> {
        (0..self.dim)
            .filter(|&i| self.grades[i] == grade)
            .collect()
    }

    /// Name of a basis blade (for display).
    pub fn blade_name(&self, blade: usize) -> String {
        if blade == 0 {
            return "1".to_string();
        }
        let mut parts = Vec::new();
        for i in 0..self.n {
            if (blade >> i) & 1 == 1 {
                parts.push(format!("e{}", i + 1));
            }
        }
        parts.join("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cl2_0() {
        // Cl(2,0): 2D Euclidean algebra
        // Basis: {1, e1, e2, e12}
        let alg = CliffordAlgebra::new(2, 0);
        assert_eq!(alg.dim, 4);
        assert_eq!(alg.n, 2);

        // Grades
        assert_eq!(alg.grade(0b00), 0); // scalar
        assert_eq!(alg.grade(0b01), 1); // e1
        assert_eq!(alg.grade(0b10), 1); // e2
        assert_eq!(alg.grade(0b11), 2); // e12

        // e1 * e1 = +1 (p=2, so first 2 basis vectors square to +1)
        let e1e1 = &alg.cayley[0b01][0b01];
        assert_eq!(e1e1.blade, 0); // scalar
        assert_eq!(e1e1.sign, Sign::Pos);

        // e1 * e2 = e12
        let e1e2 = &alg.cayley[0b01][0b10];
        assert_eq!(e1e2.blade, 0b11); // e12
        assert_eq!(e1e2.sign, Sign::Pos);

        // e2 * e1 = -e12 (anticommutative)
        let e2e1 = &alg.cayley[0b10][0b01];
        assert_eq!(e2e1.blade, 0b11);
        assert_eq!(e2e1.sign, Sign::Neg);
    }

    #[test]
    fn test_cl3_0() {
        // Cl(3,0): 3D Euclidean algebra (8 basis blades)
        let alg = CliffordAlgebra::new(3, 0);
        assert_eq!(alg.dim, 8);

        // Grade counts: 1 scalar, 3 vectors, 3 bivectors, 1 trivector
        assert_eq!(alg.blades_of_grade(0).len(), 1);
        assert_eq!(alg.blades_of_grade(1).len(), 3);
        assert_eq!(alg.blades_of_grade(2).len(), 3);
        assert_eq!(alg.blades_of_grade(3).len(), 1);
    }

    #[test]
    fn test_cl0_1() {
        // Cl(0,1): Complex numbers
        // e1^2 = -1 (like imaginary unit i)
        let alg = CliffordAlgebra::new(0, 1);
        assert_eq!(alg.dim, 2);

        let e1e1 = &alg.cayley[1][1];
        assert_eq!(e1e1.blade, 0); // scalar
        assert_eq!(e1e1.sign, Sign::Neg); // e1^2 = -1
    }

    #[test]
    fn test_blade_names() {
        let alg = CliffordAlgebra::new(3, 0);
        assert_eq!(alg.blade_name(0b000), "1");
        assert_eq!(alg.blade_name(0b001), "e1");
        assert_eq!(alg.blade_name(0b010), "e2");
        assert_eq!(alg.blade_name(0b011), "e1e2");
        assert_eq!(alg.blade_name(0b111), "e1e2e3");
    }

    #[test]
    fn test_sign_arithmetic() {
        assert_eq!(Sign::Pos * Sign::Pos, Sign::Pos);
        assert_eq!(Sign::Pos * Sign::Neg, Sign::Neg);
        assert_eq!(Sign::Neg * Sign::Neg, Sign::Pos);
        assert_eq!(Sign::Zero * Sign::Pos, Sign::Zero);
    }
}
