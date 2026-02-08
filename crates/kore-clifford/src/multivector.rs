//! Multivector — the fundamental element of a Clifford algebra.
//!
//! A multivector is a linear combination of basis blades:
//! M = a₀·1 + a₁·e₁ + a₂·e₂ + a₃·e₁₂ + ...

use crate::algebra::CliffordAlgebra;

/// A multivector in a Clifford algebra.
///
/// Stores one f32 coefficient per basis blade.
#[derive(Debug, Clone)]
pub struct Multivector {
    /// Coefficients for each basis blade (length = 2^n).
    pub coeffs: Vec<f32>,
    /// Reference to the algebra (shared).
    dim: usize,
}

impl Multivector {
    /// Create a zero multivector for an algebra of given dimension.
    pub fn zero(alg: &CliffordAlgebra) -> Self {
        Self {
            coeffs: vec![0.0; alg.dim],
            dim: alg.dim,
        }
    }

    /// Create a scalar multivector.
    pub fn scalar(alg: &CliffordAlgebra, value: f32) -> Self {
        let mut mv = Self::zero(alg);
        mv.coeffs[0] = value;
        mv
    }

    /// Create a multivector from a vector (grade-1 components).
    pub fn vector(alg: &CliffordAlgebra, components: &[f32]) -> Self {
        let mut mv = Self::zero(alg);
        let grade1 = alg.blades_of_grade(1);
        for (i, &blade) in grade1.iter().enumerate() {
            if i < components.len() {
                mv.coeffs[blade] = components[i];
            }
        }
        mv
    }

    /// Create a multivector from all coefficients.
    pub fn from_coeffs(coeffs: Vec<f32>) -> Self {
        let dim = coeffs.len();
        Self { coeffs, dim }
    }

    /// Get the scalar (grade-0) part.
    pub fn scalar_part(&self) -> f32 {
        self.coeffs[0]
    }

    /// Get the grade-k part of this multivector.
    pub fn grade_project(&self, alg: &CliffordAlgebra, grade: usize) -> Multivector {
        let mut result = Self::zero(alg);
        for (i, &coeff) in self.coeffs.iter().enumerate() {
            if alg.grade(i) == grade {
                result.coeffs[i] = coeff;
            }
        }
        result
    }

    /// Check if this multivector is approximately zero.
    pub fn is_zero(&self, eps: f32) -> bool {
        self.coeffs.iter().all(|&c| c.abs() < eps)
    }

    /// Squared norm: ||M||² = <M * M̃>₀ (scalar part of M times its reverse).
    pub fn norm_squared(&self, alg: &CliffordAlgebra) -> f32 {
        let rev = self.reverse(alg);
        let product = crate::products::geometric(alg, self, &rev);
        product.scalar_part()
    }

    /// Norm: ||M|| = sqrt(|<M * M̃>₀|).
    pub fn norm(&self, alg: &CliffordAlgebra) -> f32 {
        self.norm_squared(alg).abs().sqrt()
    }

    /// Reverse: reverses the order of basis vectors in each blade.
    /// For a grade-k blade: rev = (-1)^(k(k-1)/2) * blade
    pub fn reverse(&self, alg: &CliffordAlgebra) -> Multivector {
        let mut result = self.clone();
        for (i, coeff) in result.coeffs.iter_mut().enumerate() {
            let k = alg.grade(i);
            let sign = if (k * (k.wrapping_sub(1)) / 2) % 2 == 0 { 1.0 } else { -1.0 };
            *coeff *= sign;
        }
        result
    }

    /// Grade involution: negates odd-grade components.
    pub fn involute(&self, alg: &CliffordAlgebra) -> Multivector {
        let mut result = self.clone();
        for (i, coeff) in result.coeffs.iter_mut().enumerate() {
            if alg.grade(i) % 2 == 1 {
                *coeff = -*coeff;
            }
        }
        result
    }

    /// Conjugate: reverse followed by grade involution.
    pub fn conjugate(&self, alg: &CliffordAlgebra) -> Multivector {
        self.reverse(alg).involute(alg)
    }

    /// Dimension of the algebra.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

// Arithmetic: Add, Sub, scalar Mul
impl std::ops::Add for &Multivector {
    type Output = Multivector;
    fn add(self, rhs: &Multivector) -> Multivector {
        let coeffs: Vec<f32> = self.coeffs.iter()
            .zip(rhs.coeffs.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Multivector::from_coeffs(coeffs)
    }
}

impl std::ops::Sub for &Multivector {
    type Output = Multivector;
    fn sub(self, rhs: &Multivector) -> Multivector {
        let coeffs: Vec<f32> = self.coeffs.iter()
            .zip(rhs.coeffs.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        Multivector::from_coeffs(coeffs)
    }
}

impl std::ops::Mul<f32> for &Multivector {
    type Output = Multivector;
    fn mul(self, scalar: f32) -> Multivector {
        let coeffs: Vec<f32> = self.coeffs.iter().map(|&c| c * scalar).collect();
        Multivector::from_coeffs(coeffs)
    }
}

impl std::ops::Neg for &Multivector {
    type Output = Multivector;
    fn neg(self) -> Multivector {
        self * (-1.0)
    }
}

impl std::fmt::Display for Multivector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for (i, &c) in self.coeffs.iter().enumerate() {
            if c.abs() < 1e-7 {
                continue;
            }
            if !first && c > 0.0 {
                write!(f, " + ")?;
            } else if !first && c < 0.0 {
                write!(f, " - ")?;
            }
            if i == 0 {
                write!(f, "{:.4}", c)?;
            } else {
                write!(f, "{:.4}·e", c.abs())?;
                for bit in 0..self.dim.trailing_zeros() + 1 {
                    if (i >> bit) & 1 == 1 {
                        write!(f, "{}", bit + 1)?;
                    }
                }
            }
            first = false;
        }
        if first {
            write!(f, "0")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        let alg = CliffordAlgebra::new(3, 0);
        let mv = Multivector::zero(&alg);
        assert!(mv.is_zero(1e-7));
        assert_eq!(mv.dim(), 8);
    }

    #[test]
    fn test_scalar() {
        let alg = CliffordAlgebra::new(2, 0);
        let mv = Multivector::scalar(&alg, 3.14);
        assert!((mv.scalar_part() - 3.14).abs() < 1e-6);
    }

    #[test]
    fn test_vector() {
        let alg = CliffordAlgebra::new(3, 0);
        let v = Multivector::vector(&alg, &[1.0, 2.0, 3.0]);
        assert_eq!(v.coeffs[0b001], 1.0); // e1
        assert_eq!(v.coeffs[0b010], 2.0); // e2
        assert_eq!(v.coeffs[0b100], 3.0); // e3
    }

    #[test]
    fn test_add_sub() {
        let alg = CliffordAlgebra::new(2, 0);
        let a = Multivector::vector(&alg, &[1.0, 2.0]);
        let b = Multivector::vector(&alg, &[3.0, 4.0]);
        let c = &a + &b;
        assert_eq!(c.coeffs[0b01], 4.0);
        assert_eq!(c.coeffs[0b10], 6.0);

        let d = &a - &b;
        assert_eq!(d.coeffs[0b01], -2.0);
    }

    #[test]
    fn test_scalar_mul() {
        let alg = CliffordAlgebra::new(2, 0);
        let a = Multivector::vector(&alg, &[1.0, 2.0]);
        let b = &a * 3.0;
        assert_eq!(b.coeffs[0b01], 3.0);
        assert_eq!(b.coeffs[0b10], 6.0);
    }

    #[test]
    fn test_reverse() {
        let alg = CliffordAlgebra::new(2, 0);
        // Grade 0: sign = +1
        // Grade 1: sign = +1 (k(k-1)/2 = 0)
        // Grade 2: sign = -1 (k(k-1)/2 = 1)
        let mut mv = Multivector::zero(&alg);
        mv.coeffs[0] = 1.0;    // scalar
        mv.coeffs[0b01] = 2.0; // e1
        mv.coeffs[0b11] = 3.0; // e12

        let rev = mv.reverse(&alg);
        assert_eq!(rev.coeffs[0], 1.0);     // unchanged
        assert_eq!(rev.coeffs[0b01], 2.0);  // unchanged
        assert_eq!(rev.coeffs[0b11], -3.0); // flipped
    }

    #[test]
    fn test_grade_project() {
        let alg = CliffordAlgebra::new(3, 0);
        let mut mv = Multivector::zero(&alg);
        mv.coeffs[0] = 1.0;      // grade 0
        mv.coeffs[0b001] = 2.0;  // grade 1
        mv.coeffs[0b011] = 3.0;  // grade 2

        let g1 = mv.grade_project(&alg, 1);
        assert_eq!(g1.coeffs[0], 0.0);
        assert_eq!(g1.coeffs[0b001], 2.0);
        assert_eq!(g1.coeffs[0b011], 0.0);
    }
}
