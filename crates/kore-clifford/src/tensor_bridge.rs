//! Tensor bridge: connect Clifford algebra to kore-core Tensor.
//!
//! `MultivectorTensor` stores a batch of multivectors as a single Tensor
//! with shape `[batch..., algebra_dim]`, enabling batched geometric algebra
//! operations that integrate with the ML pipeline (autograd, GPU, etc.).

use kore_core::{KoreError, Tensor};
use crate::algebra::CliffordAlgebra;
use crate::multivector::Multivector;

/// A batched multivector stored as a Tensor.
///
/// The last dimension corresponds to the algebra's basis blade coefficients.
/// For Cl(3,0) with algebra_dim=8, a batch of N multivectors has shape `[N, 8]`.
#[derive(Debug, Clone)]
pub struct MultivectorTensor {
    /// Underlying tensor with shape `[batch..., algebra_dim]`.
    pub data: Tensor,
    /// The Clifford algebra this multivector lives in.
    pub algebra: CliffordAlgebra,
}

impl MultivectorTensor {
    /// Create a batch of zero multivectors.
    ///
    /// `batch_shape`: e.g. `[32]` for a batch of 32, or `[4, 8]` for 4×8.
    pub fn zeros(alg: &CliffordAlgebra, batch_shape: &[usize]) -> Self {
        let mut shape = batch_shape.to_vec();
        shape.push(alg.dim);
        Self {
            data: Tensor::zeros(&shape, kore_core::DType::F32),
            algebra: alg.clone(),
        }
    }

    /// Create from an existing tensor. Last dim must equal algebra dim.
    pub fn from_tensor(alg: &CliffordAlgebra, data: Tensor) -> Result<Self, KoreError> {
        let dims = data.shape().dims();
        if dims.is_empty() {
            return Err(KoreError::ShapeMismatch {
                expected: vec![alg.dim],
                got: dims.to_vec(),
            });
        }
        let last = *dims.last().unwrap();
        if last != alg.dim {
            return Err(KoreError::ShapeMismatch {
                expected: vec![alg.dim],
                got: vec![last],
            });
        }
        Ok(Self { data, algebra: alg.clone() })
    }

    /// Create a batch of scalar multivectors from a tensor of scalars.
    ///
    /// `scalars`: shape `[batch...]` → result shape `[batch..., algebra_dim]`.
    pub fn from_scalars(alg: &CliffordAlgebra, scalars: &Tensor) -> Result<Self, KoreError> {
        let s_dims = scalars.shape().dims();
        let batch_numel = scalars.numel();
        let s_data = scalars.contiguous();
        let s_slice = s_data.as_f32_slice()
            .ok_or_else(|| KoreError::UnsupportedDType(scalars.dtype()))?;

        let mut out = vec![0.0f32; batch_numel * alg.dim];
        for i in 0..batch_numel {
            out[i * alg.dim] = s_slice[i]; // scalar component is index 0
        }

        let mut shape = s_dims.to_vec();
        shape.push(alg.dim);
        Ok(Self {
            data: Tensor::from_f32(&out, &shape),
            algebra: alg.clone(),
        })
    }

    /// Create a batch of grade-1 vectors from a tensor of vector components.
    ///
    /// `vectors`: shape `[batch..., n]` where n = p+q.
    /// Result shape `[batch..., algebra_dim]`.
    pub fn from_vectors(alg: &CliffordAlgebra, vectors: &Tensor) -> Result<Self, KoreError> {
        let v_dims = vectors.shape().dims();
        if v_dims.is_empty() {
            return Err(KoreError::ShapeMismatch {
                expected: vec![alg.n],
                got: v_dims.to_vec(),
            });
        }
        let last = *v_dims.last().unwrap();
        if last != alg.n {
            return Err(KoreError::ShapeMismatch {
                expected: vec![alg.n],
                got: vec![last],
            });
        }

        let batch_numel: usize = v_dims[..v_dims.len() - 1].iter().product();
        let v_data = vectors.contiguous();
        let v_slice = v_data.as_f32_slice()
            .ok_or_else(|| KoreError::UnsupportedDType(vectors.dtype()))?;

        let grade1_blades = alg.blades_of_grade(1);
        let mut out = vec![0.0f32; batch_numel * alg.dim];

        for b in 0..batch_numel {
            for (vi, &blade_idx) in grade1_blades.iter().enumerate() {
                if vi < last {
                    out[b * alg.dim + blade_idx] = v_slice[b * last + vi];
                }
            }
        }

        let mut shape = v_dims[..v_dims.len() - 1].to_vec();
        shape.push(alg.dim);
        Ok(Self {
            data: Tensor::from_f32(&out, &shape),
            algebra: alg.clone(),
        })
    }

    /// Extract the scalar (grade-0) part as a tensor.
    ///
    /// Returns shape `[batch...]`.
    pub fn scalar_part(&self) -> Tensor {
        let dims = self.data.shape().dims();
        let batch_numel: usize = dims[..dims.len() - 1].iter().product();
        let d = self.algebra.dim;
        let data = self.data.contiguous();
        let slice = data.as_f32_slice().unwrap();

        let mut out = vec![0.0f32; batch_numel];
        for b in 0..batch_numel {
            out[b] = slice[b * d]; // index 0 = scalar blade
        }

        let batch_shape = &dims[..dims.len() - 1];
        if batch_shape.is_empty() {
            Tensor::from_f32(&out, &[1])
        } else {
            Tensor::from_f32(&out, batch_shape)
        }
    }

    /// Extract grade-k components as a tensor.
    ///
    /// Returns a new `MultivectorTensor` with only grade-k blades non-zero.
    pub fn grade_project(&self, grade: usize) -> Self {
        let dims = self.data.shape().dims();
        let batch_numel: usize = dims[..dims.len() - 1].iter().product();
        let d = self.algebra.dim;
        let data = self.data.contiguous();
        let slice = data.as_f32_slice().unwrap();

        let mut out = vec![0.0f32; batch_numel * d];
        for b in 0..batch_numel {
            for i in 0..d {
                if self.algebra.grade(i) == grade {
                    out[b * d + i] = slice[b * d + i];
                }
            }
        }

        Self {
            data: Tensor::from_f32(&out, dims),
            algebra: self.algebra.clone(),
        }
    }

    /// Batched geometric product: self ⊗ other.
    ///
    /// Both must have the same algebra and compatible batch shapes.
    /// Uses the precomputed Cayley table for the product.
    pub fn geometric_product(&self, other: &MultivectorTensor) -> Result<Self, KoreError> {
        let a_dims = self.data.shape().dims();
        let b_dims = other.data.shape().dims();
        if a_dims != b_dims {
            return Err(KoreError::ShapeMismatch {
                expected: a_dims.to_vec(),
                got: b_dims.to_vec(),
            });
        }

        let d = self.algebra.dim;
        let batch_numel: usize = a_dims[..a_dims.len() - 1].iter().product();

        let a_data = self.data.contiguous();
        let b_data = other.data.contiguous();
        let a_slice = a_data.as_f32_slice().unwrap();
        let b_slice = b_data.as_f32_slice().unwrap();

        let mut out = vec![0.0f32; batch_numel * d];

        for batch in 0..batch_numel {
            let a_off = batch * d;
            let b_off = batch * d;
            let c_off = batch * d;

            for i in 0..d {
                let ai = a_slice[a_off + i];
                if ai.abs() < 1e-10 { continue; }
                for j in 0..d {
                    let bj = b_slice[b_off + j];
                    if bj.abs() < 1e-10 { continue; }
                    let entry = &self.algebra.cayley[i][j];
                    out[c_off + entry.blade] += entry.sign.as_f32() * ai * bj;
                }
            }
        }

        Ok(Self {
            data: Tensor::from_f32(&out, a_dims),
            algebra: self.algebra.clone(),
        })
    }

    /// Batched inner product (left contraction).
    pub fn inner_product(&self, other: &MultivectorTensor) -> Result<Self, KoreError> {
        let a_dims = self.data.shape().dims();
        let b_dims = other.data.shape().dims();
        if a_dims != b_dims {
            return Err(KoreError::ShapeMismatch {
                expected: a_dims.to_vec(),
                got: b_dims.to_vec(),
            });
        }

        let d = self.algebra.dim;
        let batch_numel: usize = a_dims[..a_dims.len() - 1].iter().product();

        let a_data = self.data.contiguous();
        let b_data = other.data.contiguous();
        let a_slice = a_data.as_f32_slice().unwrap();
        let b_slice = b_data.as_f32_slice().unwrap();

        let mut out = vec![0.0f32; batch_numel * d];

        for batch in 0..batch_numel {
            let a_off = batch * d;
            let b_off = batch * d;
            let c_off = batch * d;

            for i in 0..d {
                let ai = a_slice[a_off + i];
                if ai.abs() < 1e-10 { continue; }
                let grade_a = self.algebra.grade(i);

                for j in 0..d {
                    let bj = b_slice[b_off + j];
                    if bj.abs() < 1e-10 { continue; }
                    let grade_b = self.algebra.grade(j);
                    let entry = &self.algebra.cayley[i][j];
                    let result_grade = self.algebra.grade(entry.blade);

                    if grade_b >= grade_a && result_grade == grade_b - grade_a {
                        out[c_off + entry.blade] += entry.sign.as_f32() * ai * bj;
                    }
                }
            }
        }

        Ok(Self {
            data: Tensor::from_f32(&out, a_dims),
            algebra: self.algebra.clone(),
        })
    }

    /// Batched outer (wedge) product.
    pub fn outer_product(&self, other: &MultivectorTensor) -> Result<Self, KoreError> {
        let a_dims = self.data.shape().dims();
        let b_dims = other.data.shape().dims();
        if a_dims != b_dims {
            return Err(KoreError::ShapeMismatch {
                expected: a_dims.to_vec(),
                got: b_dims.to_vec(),
            });
        }

        let d = self.algebra.dim;
        let batch_numel: usize = a_dims[..a_dims.len() - 1].iter().product();

        let a_data = self.data.contiguous();
        let b_data = other.data.contiguous();
        let a_slice = a_data.as_f32_slice().unwrap();
        let b_slice = b_data.as_f32_slice().unwrap();

        let mut out = vec![0.0f32; batch_numel * d];

        for batch in 0..batch_numel {
            let a_off = batch * d;
            let b_off = batch * d;
            let c_off = batch * d;

            for i in 0..d {
                let ai = a_slice[a_off + i];
                if ai.abs() < 1e-10 { continue; }
                let grade_a = self.algebra.grade(i);

                for j in 0..d {
                    let bj = b_slice[b_off + j];
                    if bj.abs() < 1e-10 { continue; }
                    let grade_b = self.algebra.grade(j);
                    let entry = &self.algebra.cayley[i][j];
                    let result_grade = self.algebra.grade(entry.blade);

                    if result_grade == grade_a + grade_b {
                        out[c_off + entry.blade] += entry.sign.as_f32() * ai * bj;
                    }
                }
            }
        }

        Ok(Self {
            data: Tensor::from_f32(&out, a_dims),
            algebra: self.algebra.clone(),
        })
    }

    /// Batched reverse operation.
    pub fn reverse(&self) -> Self {
        let dims = self.data.shape().dims();
        let d = self.algebra.dim;
        let batch_numel: usize = dims[..dims.len() - 1].iter().product();
        let data = self.data.contiguous();
        let slice = data.as_f32_slice().unwrap();

        let mut out = vec![0.0f32; batch_numel * d];
        for b in 0..batch_numel {
            for i in 0..d {
                let k = self.algebra.grade(i);
                let sign = if (k * k.wrapping_sub(1) / 2) % 2 == 0 { 1.0 } else { -1.0 };
                out[b * d + i] = slice[b * d + i] * sign;
            }
        }

        Self {
            data: Tensor::from_f32(&out, dims),
            algebra: self.algebra.clone(),
        }
    }

    /// Batched norm squared: <M * M̃>₀ per element.
    ///
    /// Returns shape `[batch...]`.
    pub fn norm_squared(&self) -> Result<Tensor, KoreError> {
        let rev = self.reverse();
        let product = self.geometric_product(&rev)?;
        Ok(product.scalar_part())
    }

    /// Convert a single element (index into batch) to a scalar Multivector.
    pub fn get(&self, batch_idx: usize) -> Multivector {
        let d = self.algebra.dim;
        let data = self.data.contiguous();
        let slice = data.as_f32_slice().unwrap();
        let start = batch_idx * d;
        Multivector::from_coeffs(slice[start..start + d].to_vec())
    }

    /// Set a single element from a scalar Multivector.
    pub fn set(&mut self, batch_idx: usize, mv: &Multivector) {
        let d = self.algebra.dim;
        let data = self.data.contiguous();
        let mut new_data = data.as_f32_slice().unwrap().to_vec();
        let start = batch_idx * d;
        for i in 0..d {
            new_data[start + i] = mv.coeffs[i];
        }
        self.data = Tensor::from_f32(&new_data, self.data.shape().dims());
    }

    /// Batch size (product of all dims except last).
    pub fn batch_size(&self) -> usize {
        let dims = self.data.shape().dims();
        dims[..dims.len() - 1].iter().product()
    }

    /// Algebra dimension (number of basis blades).
    pub fn algebra_dim(&self) -> usize {
        self.algebra.dim
    }
}

impl std::fmt::Display for MultivectorTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dims = self.data.shape().dims();
        write!(f, "MultivectorTensor(Cl({},{}), shape={:?})",
            self.algebra.p, self.algebra.q, dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cl30() -> CliffordAlgebra { CliffordAlgebra::new(3, 0) }
    fn cl20() -> CliffordAlgebra { CliffordAlgebra::new(2, 0) }

    #[test]
    fn test_zeros() {
        let alg = cl30();
        let mv = MultivectorTensor::zeros(&alg, &[4]);
        assert_eq!(mv.data.shape().dims(), &[4, 8]);
        assert_eq!(mv.batch_size(), 4);
        assert_eq!(mv.algebra_dim(), 8);
    }

    #[test]
    fn test_from_scalars() {
        let alg = cl20();
        let scalars = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        let mv = MultivectorTensor::from_scalars(&alg, &scalars).unwrap();
        assert_eq!(mv.data.shape().dims(), &[3, 4]);

        let sp = mv.scalar_part();
        let sp_data = sp.as_f32_slice().unwrap();
        assert!((sp_data[0] - 1.0).abs() < 1e-6);
        assert!((sp_data[1] - 2.0).abs() < 1e-6);
        assert!((sp_data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_from_vectors() {
        let alg = cl30();
        // Batch of 2 vectors in R^3
        let vecs = Tensor::from_f32(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);
        let mv = MultivectorTensor::from_vectors(&alg, &vecs).unwrap();
        assert_eq!(mv.data.shape().dims(), &[2, 8]);

        // First vector: e1 component at blade index 0b001=1
        let mv0 = mv.get(0);
        assert!((mv0.coeffs[0b001] - 1.0).abs() < 1e-6);
        assert!(mv0.coeffs[0b010].abs() < 1e-6);

        // Second vector: e2 component at blade index 0b010=2
        let mv1 = mv.get(1);
        assert!(mv1.coeffs[0b001].abs() < 1e-6);
        assert!((mv1.coeffs[0b010] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_geometric_product_scalars() {
        let alg = cl20();
        let a = MultivectorTensor::from_scalars(&alg, &Tensor::from_f32(&[3.0, 5.0], &[2])).unwrap();
        let b = MultivectorTensor::from_scalars(&alg, &Tensor::from_f32(&[4.0, 2.0], &[2])).unwrap();
        let c = a.geometric_product(&b).unwrap();

        let sp = c.scalar_part();
        let sp_data = sp.as_f32_slice().unwrap();
        assert!((sp_data[0] - 12.0).abs() < 1e-5);
        assert!((sp_data[1] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_geometric_product_vectors() {
        let alg = cl20();
        // e1 * e2 = e12
        let a = MultivectorTensor::from_vectors(&alg, &Tensor::from_f32(&[1.0, 0.0], &[1, 2])).unwrap();
        let b = MultivectorTensor::from_vectors(&alg, &Tensor::from_f32(&[0.0, 1.0], &[1, 2])).unwrap();
        let c = a.geometric_product(&b).unwrap();

        let mv = c.get(0);
        assert!((mv.coeffs[0b11] - 1.0).abs() < 1e-6, "e12 coeff: {}", mv.coeffs[0b11]);
    }

    #[test]
    fn test_inner_product_batch() {
        let alg = cl30();
        // Batch of 2: dot products
        let a = MultivectorTensor::from_vectors(&alg, &Tensor::from_f32(
            &[1.0, 2.0, 3.0, 1.0, 0.0, 0.0], &[2, 3]
        )).unwrap();
        let b = MultivectorTensor::from_vectors(&alg, &Tensor::from_f32(
            &[4.0, 5.0, 6.0, 0.0, 0.0, 1.0], &[2, 3]
        )).unwrap();
        let c = a.inner_product(&b).unwrap();

        let sp = c.scalar_part();
        let sp_data = sp.as_f32_slice().unwrap();
        // 1*4 + 2*5 + 3*6 = 32
        assert!((sp_data[0] - 32.0).abs() < 1e-4, "got {}", sp_data[0]);
        // 1*0 + 0*0 + 0*1 = 0
        assert!(sp_data[1].abs() < 1e-4, "got {}", sp_data[1]);
    }

    #[test]
    fn test_outer_product_batch() {
        let alg = cl30();
        let a = MultivectorTensor::from_vectors(&alg, &Tensor::from_f32(&[1.0, 0.0, 0.0], &[1, 3])).unwrap();
        let b = MultivectorTensor::from_vectors(&alg, &Tensor::from_f32(&[0.0, 1.0, 0.0], &[1, 3])).unwrap();
        let c = a.outer_product(&b).unwrap();

        let mv = c.get(0);
        // e1 ∧ e2 = e12 (blade index 0b011 = 3)
        assert!((mv.coeffs[0b011] - 1.0).abs() < 1e-6);
        assert!(mv.scalar_part().abs() < 1e-6);
    }

    #[test]
    fn test_reverse_batch() {
        let alg = cl20();
        let mut mv = MultivectorTensor::zeros(&alg, &[1]);
        let mut m = Multivector::zero(&alg);
        m.coeffs[0] = 1.0;     // scalar: unchanged
        m.coeffs[0b01] = 2.0;  // e1: unchanged
        m.coeffs[0b11] = 3.0;  // e12: flipped
        mv.set(0, &m);

        let rev = mv.reverse();
        let r = rev.get(0);
        assert!((r.coeffs[0] - 1.0).abs() < 1e-6);
        assert!((r.coeffs[0b01] - 2.0).abs() < 1e-6);
        assert!((r.coeffs[0b11] - (-3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_norm_squared() {
        let alg = cl30();
        // Vector [1, 2, 3]: norm^2 = 1 + 4 + 9 = 14
        let v = MultivectorTensor::from_vectors(&alg, &Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3])).unwrap();
        let ns = v.norm_squared().unwrap();
        let ns_data = ns.as_f32_slice().unwrap();
        assert!((ns_data[0] - 14.0).abs() < 1e-4, "got {}", ns_data[0]);
    }

    #[test]
    fn test_grade_project() {
        let alg = cl30();
        let mut mv = MultivectorTensor::zeros(&alg, &[1]);
        let mut m = Multivector::zero(&alg);
        m.coeffs[0] = 5.0;      // grade 0
        m.coeffs[0b001] = 2.0;  // grade 1
        m.coeffs[0b011] = 3.0;  // grade 2
        mv.set(0, &m);

        let g1 = mv.grade_project(1);
        let g1_mv = g1.get(0);
        assert!(g1_mv.coeffs[0].abs() < 1e-6);         // no scalar
        assert!((g1_mv.coeffs[0b001] - 2.0).abs() < 1e-6); // e1 kept
        assert!(g1_mv.coeffs[0b011].abs() < 1e-6);      // no bivector
    }

    #[test]
    fn test_consistency_with_scalar_api() {
        // Verify batched geometric product matches scalar product
        let alg = cl30();
        let a_vec = [1.0, 2.0, 3.0];
        let b_vec = [4.0, 5.0, 6.0];

        // Scalar API
        let a_mv = Multivector::vector(&alg, &a_vec);
        let b_mv = Multivector::vector(&alg, &b_vec);
        let c_scalar = crate::products::geometric(&alg, &a_mv, &b_mv);

        // Tensor API
        let a_t = MultivectorTensor::from_vectors(&alg, &Tensor::from_f32(&a_vec, &[1, 3])).unwrap();
        let b_t = MultivectorTensor::from_vectors(&alg, &Tensor::from_f32(&b_vec, &[1, 3])).unwrap();
        let c_tensor = a_t.geometric_product(&b_t).unwrap();
        let c_t_mv = c_tensor.get(0);

        for i in 0..alg.dim {
            assert!(
                (c_scalar.coeffs[i] - c_t_mv.coeffs[i]).abs() < 1e-4,
                "blade {}: scalar={}, tensor={}",
                i, c_scalar.coeffs[i], c_t_mv.coeffs[i]
            );
        }
    }

    #[test]
    fn test_display() {
        let alg = cl30();
        let mv = MultivectorTensor::zeros(&alg, &[4]);
        let s = format!("{}", mv);
        assert!(s.contains("Cl(3,0)"));
        assert!(s.contains("[4, 8]"));
    }
}
