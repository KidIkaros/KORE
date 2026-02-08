//! Higher-level operations on multivectors: dual, sandwich product, rotor.

use crate::algebra::CliffordAlgebra;
use crate::multivector::Multivector;
use crate::products::geometric;

/// Dual: M* = M · I^{-1} where I is the pseudoscalar.
///
/// The pseudoscalar is the highest-grade blade (e.g., e123 in Cl(3,0)).
pub fn dual(alg: &CliffordAlgebra, mv: &Multivector) -> Multivector {
    let pseudo_idx = alg.dim - 1; // highest blade index = all bits set
    let mut pseudo = Multivector::zero(alg);
    pseudo.coeffs[pseudo_idx] = 1.0;

    // Compute pseudoscalar inverse: I^{-1} = rev(I) / (I * rev(I))
    let pseudo_rev = pseudo.reverse(alg);
    let norm_sq = geometric(alg, &pseudo, &pseudo_rev).scalar_part();

    if norm_sq.abs() < 1e-10 {
        return Multivector::zero(alg); // degenerate algebra
    }

    let pseudo_inv_coeffs: Vec<f32> = pseudo_rev.coeffs.iter().map(|&c| c / norm_sq).collect();
    let pseudo_inv = Multivector::from_coeffs(pseudo_inv_coeffs);

    geometric(alg, mv, &pseudo_inv)
}

/// Sandwich product: R * M * R̃ (used for rotations/reflections).
///
/// R is typically a rotor (even-grade multivector with |R|=1).
pub fn sandwich(alg: &CliffordAlgebra, rotor: &Multivector, mv: &Multivector) -> Multivector {
    let rev = rotor.reverse(alg);
    let temp = geometric(alg, rotor, mv);
    geometric(alg, &temp, &rev)
}

/// Create a rotor from a bivector angle: R = exp(-B/2)
///
/// For a unit bivector B with B² = -1:
/// R = cos(θ/2) - sin(θ/2) * B̂
///
/// This implements the Taylor series for small algebras.
pub fn rotor_from_bivector(alg: &CliffordAlgebra, bivector: &Multivector, angle: f32) -> Multivector {
    let half_angle = angle / 2.0;

    // Compute B² to determine if it's a simple bivector
    let b_sq = geometric(alg, bivector, bivector);
    let b_sq_scalar = b_sq.scalar_part();

    if b_sq_scalar < 0.0 {
        // B² < 0: standard rotation (Euclidean case)
        let b_norm = (-b_sq_scalar).sqrt();
        let cos_ha = (half_angle * b_norm).cos();
        let sin_ha = if b_norm.abs() < 1e-10 {
            half_angle
        } else {
            (half_angle * b_norm).sin() / b_norm
        };

        // R = cos(θ/2) - sin(θ/2) * B
        let scalar = Multivector::scalar(alg, cos_ha);
        let biv_part = bivector * (-sin_ha);
        &scalar + &biv_part
    } else {
        // B² >= 0: hyperbolic case or degenerate
        // Use Taylor series: exp(-B*θ/2) ≈ 1 - B*θ/2 + B²*θ²/8 - ...
        let mut result = Multivector::scalar(alg, 1.0);
        let mut term = Multivector::scalar(alg, 1.0);
        let neg_half_b = bivector * (-half_angle);

        for k in 1..=8 {
            term = geometric(alg, &term, &neg_half_b);
            let factor = 1.0 / factorial(k) as f32;
            let scaled = &term * factor;
            result = &result + &scaled;
        }
        result
    }
}

fn factorial(n: usize) -> usize {
    (1..=n).product()
}

/// Reflect a vector through a hyperplane defined by its normal vector.
///
/// reflection(n, v) = -n * v * n^{-1}
pub fn reflect(alg: &CliffordAlgebra, normal: &Multivector, v: &Multivector) -> Multivector {
    let n_sq = geometric(alg, normal, normal).scalar_part();
    if n_sq.abs() < 1e-10 {
        return Multivector::zero(alg);
    }

    let n_inv_coeffs: Vec<f32> = normal.coeffs.iter().map(|&c| c / n_sq).collect();
    let n_inv = Multivector::from_coeffs(n_inv_coeffs);

    let temp = geometric(alg, normal, v);
    let result = geometric(alg, &temp, &n_inv);
    &result * (-1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sandwich_identity() {
        // Sandwich with scalar 1 is identity
        let alg = CliffordAlgebra::new(3, 0);
        let rotor = Multivector::scalar(&alg, 1.0);
        let v = Multivector::vector(&alg, &[1.0, 2.0, 3.0]);

        let result = sandwich(&alg, &rotor, &v);
        for i in 0..alg.dim {
            assert!(
                (result.coeffs[i] - v.coeffs[i]).abs() < 1e-5,
                "Mismatch at {}: {} vs {}",
                i, result.coeffs[i], v.coeffs[i]
            );
        }
    }

    #[test]
    fn test_reflection() {
        // Reflect (1,0,0) through the e1 normal → (-1,0,0)
        let alg = CliffordAlgebra::new(3, 0);
        let normal = Multivector::vector(&alg, &[1.0, 0.0, 0.0]);
        let v = Multivector::vector(&alg, &[1.0, 0.0, 0.0]);

        let reflected = reflect(&alg, &normal, &v);
        assert!((reflected.coeffs[0b001] - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_reflection_perpendicular() {
        // Reflect (0,1,0) through e1 normal → (0,1,0) (perpendicular, unchanged)
        let alg = CliffordAlgebra::new(3, 0);
        let normal = Multivector::vector(&alg, &[1.0, 0.0, 0.0]);
        let v = Multivector::vector(&alg, &[0.0, 1.0, 0.0]);

        let reflected = reflect(&alg, &normal, &v);
        assert!((reflected.coeffs[0b010] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rotor_rotation_90deg() {
        // Rotate e1 by 90° in the e12 plane → e2
        let alg = CliffordAlgebra::new(2, 0);

        // Bivector for e12 plane
        let mut bivector = Multivector::zero(&alg);
        bivector.coeffs[0b11] = 1.0; // e12

        let angle = std::f32::consts::FRAC_PI_2; // 90°
        let rotor = rotor_from_bivector(&alg, &bivector, angle);

        let e1 = Multivector::vector(&alg, &[1.0, 0.0]);
        let rotated = sandwich(&alg, &rotor, &e1);

        // Should be approximately e2
        assert!(rotated.coeffs[0b01].abs() < 0.1, "e1 component: {}", rotated.coeffs[0b01]);
        assert!((rotated.coeffs[0b10].abs() - 1.0).abs() < 0.1, "e2 component: {}", rotated.coeffs[0b10]);
    }

    #[test]
    fn test_dual_cl3() {
        let alg = CliffordAlgebra::new(3, 0);
        let e1 = Multivector::vector(&alg, &[1.0, 0.0, 0.0]);
        let d = dual(&alg, &e1);

        // dual(e1) in Cl(3,0) should be a bivector (grade 2)
        let grade = d.grade_project(&alg, 2);
        assert!(!grade.is_zero(1e-5));
    }
}
