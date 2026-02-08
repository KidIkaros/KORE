//! Clifford algebra products: geometric, inner, outer.

use crate::algebra::CliffordAlgebra;
use crate::multivector::Multivector;

/// Geometric product: the fundamental product of Clifford algebra.
///
/// For two multivectors A and B:
/// (A * B)_k = Σ_ij cayley[i][j].sign * A_i * B_j  (where cayley[i][j].blade == k)
pub fn geometric(alg: &CliffordAlgebra, a: &Multivector, b: &Multivector) -> Multivector {
    let mut result = Multivector::zero(alg);

    for i in 0..alg.dim {
        if a.coeffs[i].abs() < 1e-10 {
            continue;
        }
        for j in 0..alg.dim {
            if b.coeffs[j].abs() < 1e-10 {
                continue;
            }
            let entry = &alg.cayley[i][j];
            let val = entry.sign.as_f32() * a.coeffs[i] * b.coeffs[j];
            result.coeffs[entry.blade] += val;
        }
    }

    result
}

/// Inner product (left contraction): extracts the "dot product" part.
///
/// <A>_r ⌋ <B>_s = <A*B>_{s-r}  when s >= r, else 0.
/// For vectors: a · b = <ab>_0 (scalar part of geometric product).
pub fn inner(alg: &CliffordAlgebra, a: &Multivector, b: &Multivector) -> Multivector {
    let mut result = Multivector::zero(alg);

    for i in 0..alg.dim {
        if a.coeffs[i].abs() < 1e-10 {
            continue;
        }
        let grade_a = alg.grade(i);

        for j in 0..alg.dim {
            if b.coeffs[j].abs() < 1e-10 {
                continue;
            }
            let grade_b = alg.grade(j);
            let entry = &alg.cayley[i][j];
            let result_grade = alg.grade(entry.blade);

            // Inner product: keep only terms where result grade = |grade_b - grade_a|
            if grade_b >= grade_a && result_grade == grade_b - grade_a {
                let val = entry.sign.as_f32() * a.coeffs[i] * b.coeffs[j];
                result.coeffs[entry.blade] += val;
            }
        }
    }

    result
}

/// Outer (wedge) product: extracts the "cross product" part.
///
/// <A>_r ∧ <B>_s = <A*B>_{r+s}
pub fn outer(alg: &CliffordAlgebra, a: &Multivector, b: &Multivector) -> Multivector {
    let mut result = Multivector::zero(alg);

    for i in 0..alg.dim {
        if a.coeffs[i].abs() < 1e-10 {
            continue;
        }
        let grade_a = alg.grade(i);

        for j in 0..alg.dim {
            if b.coeffs[j].abs() < 1e-10 {
                continue;
            }
            let grade_b = alg.grade(j);
            let entry = &alg.cayley[i][j];
            let result_grade = alg.grade(entry.blade);

            // Outer product: keep only terms where result grade = grade_a + grade_b
            if result_grade == grade_a + grade_b {
                let val = entry.sign.as_f32() * a.coeffs[i] * b.coeffs[j];
                result.coeffs[entry.blade] += val;
            }
        }
    }

    result
}

/// Scalar product: <AB>_0 (grade-0 part of geometric product).
pub fn scalar_product(alg: &CliffordAlgebra, a: &Multivector, b: &Multivector) -> f32 {
    let mut result = 0.0f32;

    for i in 0..alg.dim {
        if a.coeffs[i].abs() < 1e-10 {
            continue;
        }
        for j in 0..alg.dim {
            if b.coeffs[j].abs() < 1e-10 {
                continue;
            }
            let entry = &alg.cayley[i][j];
            if entry.blade == 0 {
                result += entry.sign.as_f32() * a.coeffs[i] * b.coeffs[j];
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geometric_scalars() {
        let alg = CliffordAlgebra::new(2, 0);
        let a = Multivector::scalar(&alg, 3.0);
        let b = Multivector::scalar(&alg, 4.0);
        let c = geometric(&alg, &a, &b);
        assert!((c.scalar_part() - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_geometric_vector_self() {
        // In Cl(2,0): e1 * e1 = +1
        let alg = CliffordAlgebra::new(2, 0);
        let e1 = Multivector::vector(&alg, &[1.0, 0.0]);
        let result = geometric(&alg, &e1, &e1);
        assert!((result.scalar_part() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_geometric_anticommutative() {
        // e1 * e2 = e12, e2 * e1 = -e12
        let alg = CliffordAlgebra::new(2, 0);
        let e1 = Multivector::vector(&alg, &[1.0, 0.0]);
        let e2 = Multivector::vector(&alg, &[0.0, 1.0]);

        let e1e2 = geometric(&alg, &e1, &e2);
        let e2e1 = geometric(&alg, &e2, &e1);

        assert!((e1e2.coeffs[0b11] - 1.0).abs() < 1e-6);
        assert!((e2e1.coeffs[0b11] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_inner_product_vectors() {
        // Inner product of vectors = dot product
        let alg = CliffordAlgebra::new(3, 0);
        let a = Multivector::vector(&alg, &[1.0, 2.0, 3.0]);
        let b = Multivector::vector(&alg, &[4.0, 5.0, 6.0]);

        let dot = inner(&alg, &a, &b);
        // 1*4 + 2*5 + 3*6 = 32
        assert!((dot.scalar_part() - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_outer_product_vectors() {
        // Outer product of vectors = bivector
        let alg = CliffordAlgebra::new(3, 0);
        let e1 = Multivector::vector(&alg, &[1.0, 0.0, 0.0]);
        let e2 = Multivector::vector(&alg, &[0.0, 1.0, 0.0]);

        let wedge = outer(&alg, &e1, &e2);
        // e1 ∧ e2 = e12
        assert!((wedge.coeffs[0b011] - 1.0).abs() < 1e-6);
        assert!(wedge.scalar_part().abs() < 1e-6); // no scalar part
    }

    #[test]
    fn test_geometric_equals_inner_plus_outer_for_vectors() {
        // For vectors: a*b = a·b + a∧b
        let alg = CliffordAlgebra::new(3, 0);
        let a = Multivector::vector(&alg, &[1.0, 2.0, 3.0]);
        let b = Multivector::vector(&alg, &[4.0, 5.0, 6.0]);

        let geo = geometric(&alg, &a, &b);
        let inn = inner(&alg, &a, &b);
        let out = outer(&alg, &a, &b);
        let sum = &inn + &out;

        for i in 0..alg.dim {
            assert!(
                (geo.coeffs[i] - sum.coeffs[i]).abs() < 1e-5,
                "Mismatch at blade {}: geo={}, sum={}",
                i, geo.coeffs[i], sum.coeffs[i]
            );
        }
    }

    #[test]
    fn test_complex_numbers() {
        // Cl(0,1) ≅ complex numbers: e1^2 = -1
        let alg = CliffordAlgebra::new(0, 1);

        // z1 = 2 + 3i (scalar + e1)
        let mut z1 = Multivector::zero(&alg);
        z1.coeffs[0] = 2.0;
        z1.coeffs[1] = 3.0;

        // z2 = 1 + 2i
        let mut z2 = Multivector::zero(&alg);
        z2.coeffs[0] = 1.0;
        z2.coeffs[1] = 2.0;

        // z1 * z2 = (2+3i)(1+2i) = 2+4i+3i+6i² = 2+7i-6 = -4+7i
        let product = geometric(&alg, &z1, &z2);
        assert!((product.coeffs[0] - (-4.0)).abs() < 1e-5);
        assert!((product.coeffs[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_scalar_product() {
        let alg = CliffordAlgebra::new(3, 0);
        let a = Multivector::vector(&alg, &[1.0, 0.0, 0.0]);
        let b = Multivector::vector(&alg, &[1.0, 0.0, 0.0]);
        let sp = scalar_product(&alg, &a, &b);
        assert!((sp - 1.0).abs() < 1e-6);
    }
}
