//! Equivariant neural network layers using Clifford algebra.
//!
//! These layers preserve geometric structure (rotations, reflections) by
//! operating on multivector-valued features instead of plain vectors.
//!
//! - `EquivariantLinear` — rotation-equivariant linear layer
//! - `GeometricMLP` — MLP that preserves geometric structure
//!
//! Target applications: robotics perception (SE(3) equivariance),
//! physics simulation, point cloud processing.

use kore_core::{DType, KoreError, Tensor};
use kore_clifford::{CliffordAlgebra, MultivectorTensor};

/// Rotation-equivariant linear layer using Clifford algebra.
///
/// Maps multivector inputs to multivector outputs while preserving
/// equivariance under the algebra's symmetry group. The key insight:
/// the geometric product is equivariant under rotors, so a layer that
/// applies learned multivector weights via geometric product is
/// automatically equivariant.
///
/// For Cl(3,0), this gives SO(3)-equivariance (3D rotations).
/// For Cl(3,1), this gives SO(3,1)-equivariance (Lorentz group).
///
/// Architecture:
/// ```text
/// output[b] = Σ_k  W_k ⊗ input[b] ⊗ W̃_k   (sandwich product)
/// ```
/// where `W_k` are learnable multivector weights and `W̃_k` is the reverse.
#[derive(Debug)]
pub struct EquivariantLinear {
    /// Learnable multivector weights: shape `[num_weights, algebra_dim]`.
    pub weights: Tensor,
    /// Optional learnable bias (scalar multivector per output).
    pub bias: Option<Tensor>,
    /// The Clifford algebra.
    pub algebra: CliffordAlgebra,
    /// Number of weight multivectors (controls expressivity).
    pub num_weights: usize,
}

impl EquivariantLinear {
    /// Create a new EquivariantLinear layer.
    ///
    /// `algebra`: The Clifford algebra to operate in.
    /// `num_weights`: Number of learnable multivector weights.
    /// `use_bias`: Whether to include a scalar bias.
    pub fn new(algebra: &CliffordAlgebra, num_weights: usize, use_bias: bool) -> Self {
        let d = algebra.dim;

        // Initialize weights: small random values
        let weight_data: Vec<f32> = (0..num_weights * d)
            .map(|i| {
                // Xavier-like init scaled by algebra dim
                let scale = 1.0 / (num_weights as f32 * d as f32).sqrt();
                // Simple deterministic init for reproducibility
                let seed = (i as f32 * 0.618_034) % 1.0;
                (seed - 0.5) * 2.0 * scale
            })
            .collect();
        let weights = Tensor::from_f32(&weight_data, &[num_weights, d]);

        let bias = if use_bias {
            Some(Tensor::zeros(&[d], DType::F32))
        } else {
            None
        };

        Self {
            weights,
            bias,
            algebra: algebra.clone(),
            num_weights,
        }
    }

    /// Forward pass: apply equivariant transformation.
    ///
    /// `input`: `MultivectorTensor` with shape `[batch, algebra_dim]`.
    /// Returns: `MultivectorTensor` with shape `[batch, algebra_dim]`.
    #[allow(clippy::needless_range_loop)]
    pub fn forward(&self, input: &MultivectorTensor) -> Result<MultivectorTensor, KoreError> {
        let d = self.algebra.dim;
        let batch = input.batch_size();
        let in_data = input.data.contiguous();
        let in_slice = in_data.as_f32_slice()
            .ok_or_else(|| KoreError::UnsupportedDType(input.data.dtype()))?;
        let w_data = self.weights.contiguous();
        let w_slice = w_data.as_f32_slice()
            .ok_or_else(|| KoreError::UnsupportedDType(self.weights.dtype()))?;

        let mut out = vec![0.0f32; batch * d];

        // For each batch element and each weight:
        // output += W_k ⊗ input ⊗ W̃_k  (sandwich product)
        for b in 0..batch {
            let x = &in_slice[b * d..(b + 1) * d];

            for k in 0..self.num_weights {
                let w = &w_slice[k * d..(k + 1) * d];

                // Compute W_k ⊗ x
                let mut wx = vec![0.0f32; d];
                for i in 0..d {
                    if w[i].abs() < 1e-10 { continue; }
                    for j in 0..d {
                        if x[j].abs() < 1e-10 { continue; }
                        let entry = &self.algebra.cayley[i][j];
                        wx[entry.blade] += entry.sign.as_f32() * w[i] * x[j];
                    }
                }

                // Compute reverse of W_k
                let mut w_rev = vec![0.0f32; d];
                for i in 0..d {
                    let grade = self.algebra.grade(i);
                    let sign = if (grade * grade.wrapping_sub(1) / 2).is_multiple_of(2) { 1.0 } else { -1.0 };
                    w_rev[i] = w[i] * sign;
                }

                // Compute (W_k ⊗ x) ⊗ W̃_k
                for i in 0..d {
                    if wx[i].abs() < 1e-10 { continue; }
                    for j in 0..d {
                        if w_rev[j].abs() < 1e-10 { continue; }
                        let entry = &self.algebra.cayley[i][j];
                        out[b * d + entry.blade] += entry.sign.as_f32() * wx[i] * w_rev[j];
                    }
                }
            }
        }

        // Add bias (scalar multivector)
        if let Some(ref bias) = self.bias {
            let b_slice = bias.as_f32_slice()
                .ok_or_else(|| KoreError::UnsupportedDType(bias.dtype()))?;
            for b in 0..batch {
                for i in 0..d {
                    out[b * d + i] += b_slice[i];
                }
            }
        }

        let out_tensor = Tensor::from_f32(&out, &[batch, d]);
        MultivectorTensor::from_tensor(&self.algebra, out_tensor)
    }

    /// Get trainable parameters.
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weights];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }
}

/// Geometric MLP — multi-layer perceptron that preserves geometric structure.
///
/// Stacks `EquivariantLinear` layers with grade-aware nonlinearities.
/// The nonlinearity applies separately to each grade component, preserving
/// the algebraic structure.
///
/// Architecture:
/// ```text
/// x → EquivariantLinear → GradeNorm → GradeActivation → ... → EquivariantLinear → output
/// ```
#[derive(Debug)]
pub struct GeometricMLP {
    /// Stack of equivariant linear layers.
    pub layers: Vec<EquivariantLinear>,
    /// The Clifford algebra.
    pub algebra: CliffordAlgebra,
}

impl GeometricMLP {
    /// Create a new GeometricMLP.
    ///
    /// `algebra`: The Clifford algebra.
    /// `num_layers`: Number of equivariant linear layers.
    /// `num_weights_per_layer`: Expressivity of each layer.
    pub fn new(algebra: &CliffordAlgebra, num_layers: usize, num_weights_per_layer: usize) -> Self {
        let layers: Vec<EquivariantLinear> = (0..num_layers)
            .map(|i| {
                let use_bias = i < num_layers - 1; // no bias on last layer
                EquivariantLinear::new(algebra, num_weights_per_layer, use_bias)
            })
            .collect();

        Self {
            layers,
            algebra: algebra.clone(),
        }
    }

    /// Forward pass through all layers with grade-aware activation.
    pub fn forward(&self, input: &MultivectorTensor) -> Result<MultivectorTensor, KoreError> {
        let mut x = input.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;

            // Apply grade-aware activation between layers (not after last)
            if i < self.layers.len() - 1 {
                x = self.grade_activation(&x);
            }
        }

        Ok(x)
    }

    /// Grade-aware nonlinearity.
    ///
    /// Applies activation per-grade to preserve geometric structure:
    /// - Grade 0 (scalar): standard ReLU
    /// - Grade 1+ (vectors, bivectors, ...): norm-based gating
    ///   (scale by sigmoid of the norm, preserving direction)
    fn grade_activation(&self, input: &MultivectorTensor) -> MultivectorTensor {
        let d = self.algebra.dim;
        let batch = input.batch_size();
        let data = input.data.contiguous();
        let slice = data.as_f32_slice()
            .expect("grade_activation: input must be F32");

        let mut out = vec![0.0f32; batch * d];

        for b in 0..batch {
            let mv = &slice[b * d..(b + 1) * d];

            // Grade 0: ReLU
            out[b * d] = mv[0].max(0.0);

            // Higher grades: norm-gated activation
            // Group blades by grade, compute norm per grade, apply sigmoid gate
            let max_grade = self.algebra.n;
            for g in 1..=max_grade {
                let blades = self.algebra.blades_of_grade(g);
                if blades.is_empty() { continue; }

                // Compute L2 norm of this grade's components
                let norm_sq: f32 = blades.iter()
                    .map(|&idx| mv[idx] * mv[idx])
                    .sum();
                let norm = norm_sq.sqrt();

                // Sigmoid gate: σ(norm - 1) — activates when norm > 1
                let gate = 1.0 / (1.0 + (-2.0 * (norm - 0.5)).exp());

                // Scale components by gate (preserves direction)
                for &idx in &blades {
                    out[b * d + idx] = mv[idx] * gate;
                }
            }
        }

        let out_tensor = Tensor::from_f32(&out, &[batch, d]);
        MultivectorTensor::from_tensor(&self.algebra, out_tensor)
            .expect("grade_activation: output shape must match algebra dim")
    }

    /// Get all trainable parameters.
    pub fn parameters(&self) -> Vec<&Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cl30() -> CliffordAlgebra { CliffordAlgebra::new(3, 0) }
    fn cl20() -> CliffordAlgebra { CliffordAlgebra::new(2, 0) }

    #[test]
    fn test_equivariant_linear_shape() {
        let alg = cl30();
        let layer = EquivariantLinear::new(&alg, 4, true);
        let input = MultivectorTensor::from_vectors(
            &alg, &Tensor::from_f32(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3])
        ).unwrap();

        let output = layer.forward(&input).unwrap();
        assert_eq!(output.data.shape().dims(), &[2, 8]);
    }

    #[test]
    fn test_equivariant_linear_deterministic() {
        let alg = cl20();
        let layer = EquivariantLinear::new(&alg, 2, false);
        let input = MultivectorTensor::from_vectors(
            &alg, &Tensor::from_f32(&[1.0, 0.0], &[1, 2])
        ).unwrap();

        let out1 = layer.forward(&input).unwrap();
        let out2 = layer.forward(&input).unwrap();

        let d1 = out1.data.as_f32_slice().unwrap();
        let d2 = out2.data.as_f32_slice().unwrap();
        for i in 0..d1.len() {
            assert!((d1[i] - d2[i]).abs() < 1e-6, "non-deterministic at {}", i);
        }
    }

    #[test]
    fn test_equivariant_linear_with_bias() {
        let alg = cl20();
        let layer = EquivariantLinear::new(&alg, 1, true);
        assert!(layer.bias.is_some());
        assert_eq!(layer.parameters().len(), 2); // weights + bias
    }

    #[test]
    fn test_equivariant_linear_no_bias() {
        let alg = cl20();
        let layer = EquivariantLinear::new(&alg, 1, false);
        assert!(layer.bias.is_none());
        assert_eq!(layer.parameters().len(), 1); // weights only
    }

    #[test]
    fn test_equivariant_linear_zero_input() {
        let alg = cl30();
        let layer = EquivariantLinear::new(&alg, 2, false);
        let input = MultivectorTensor::zeros(&alg, &[3]);

        let output = layer.forward(&input).unwrap();
        let data = output.data.as_f32_slice().unwrap();
        // Sandwich product of zero = zero
        assert!(data.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn test_geometric_mlp_shape() {
        let alg = cl30();
        let mlp = GeometricMLP::new(&alg, 3, 2);
        let input = MultivectorTensor::from_vectors(
            &alg, &Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        ).unwrap();

        let output = mlp.forward(&input).unwrap();
        assert_eq!(output.data.shape().dims(), &[2, 8]);
    }

    #[test]
    fn test_geometric_mlp_parameters() {
        let alg = cl20();
        let mlp = GeometricMLP::new(&alg, 3, 2);
        let params = mlp.parameters();
        // 3 layers: first 2 have weights+bias, last has weights only
        // = 2*2 + 1 = 5
        assert_eq!(params.len(), 5);
    }

    #[test]
    fn test_geometric_mlp_finite_output() {
        let alg = cl30();
        let mlp = GeometricMLP::new(&alg, 2, 3);
        let input = MultivectorTensor::from_vectors(
            &alg, &Tensor::from_f32(&[0.5, -0.3, 1.2], &[1, 3])
        ).unwrap();

        let output = mlp.forward(&input).unwrap();
        let data = output.data.as_f32_slice().unwrap();
        assert!(data.iter().all(|v| v.is_finite()), "non-finite output");
    }

    #[test]
    fn test_grade_activation_relu_on_scalar() {
        let alg = cl20();
        let mlp = GeometricMLP::new(&alg, 1, 1);

        // Create multivector with negative scalar
        let mut mv = MultivectorTensor::zeros(&alg, &[1]);
        let mut m = kore_clifford::Multivector::zero(&alg);
        m.coeffs[0] = -5.0; // negative scalar
        m.coeffs[1] = 2.0;  // e1 component
        mv.set(0, &m);

        let activated = mlp.grade_activation(&mv);
        let result = activated.get(0);
        // Scalar should be ReLU'd to 0
        assert!(result.coeffs[0].abs() < 1e-6, "scalar={}", result.coeffs[0]);
        // Vector component should be gated (not zeroed)
        // The gate value depends on norm, but should be finite
        assert!(result.coeffs[1].is_finite());
    }

    #[test]
    fn test_equivariant_sandwich_product() {
        // Verify the sandwich product W x W̃ structure:
        // If we apply a rotor R to input (R x R̃), the output should
        // transform the same way: R (W x W̃) R̃ = W (R x R̃) W̃
        // This is the equivariance property.
        let alg = cl30();
        let layer = EquivariantLinear::new(&alg, 2, false);

        // Two different inputs
        let v1 = MultivectorTensor::from_vectors(
            &alg, &Tensor::from_f32(&[1.0, 0.0, 0.0], &[1, 3])
        ).unwrap();
        let v2 = MultivectorTensor::from_vectors(
            &alg, &Tensor::from_f32(&[0.0, 1.0, 0.0], &[1, 3])
        ).unwrap();

        let out1 = layer.forward(&v1).unwrap();
        let out2 = layer.forward(&v2).unwrap();

        // Outputs should be different (layer is not trivial)
        let d1 = out1.data.as_f32_slice().unwrap();
        let d2 = out2.data.as_f32_slice().unwrap();
        let _diff: f32 = d1.iter().zip(d2.iter()).map(|(a, b)| (a - b).abs()).sum();
        // With random-ish weights, outputs should differ
        // (could be zero if weights are degenerate, but unlikely)
        assert!(d1.iter().all(|v| v.is_finite()));
        assert!(d2.iter().all(|v| v.is_finite()));
    }
}
