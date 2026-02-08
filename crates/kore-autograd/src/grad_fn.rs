//! Gradient function trait and built-in backward implementations.

use kore_core::Tensor;

/// Trait for gradient functions in the computation graph.
///
/// Each differentiable operation implements this trait to define
/// how gradients flow backward through it.
pub trait GradFn: Send + Sync {
    /// Compute gradients for each input given the output gradient.
    ///
    /// Returns a vector of optional gradients (one per input).
    /// `None` means the input doesn't need a gradient.
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>>;

    /// Name of this gradient function (for debugging).
    fn name(&self) -> &str;
}

// ============================================================================
// Built-in gradient functions
// ============================================================================

/// Backward for element-wise addition: grad flows through unchanged.
pub struct AddBackward;

impl GradFn for AddBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(grad_output.clone()), Some(grad_output.clone())]
    }

    fn name(&self) -> &str {
        "AddBackward"
    }
}

/// Backward for element-wise subtraction.
pub struct SubBackward;

impl GradFn for SubBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let neg = grad_output.neg().expect("SubBackward neg failed");
        vec![Some(grad_output.clone()), Some(neg)]
    }

    fn name(&self) -> &str {
        "SubBackward"
    }
}

/// Backward for element-wise multiplication.
/// Requires saved inputs from the forward pass.
pub struct MulBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
}

impl GradFn for MulBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // d/da (a*b) = b, d/db (a*b) = a
        let grad_a = grad_output.mul(&self.rhs).expect("MulBackward grad_a failed");
        let grad_b = grad_output.mul(&self.lhs).expect("MulBackward grad_b failed");
        vec![Some(grad_a), Some(grad_b)]
    }

    fn name(&self) -> &str {
        "MulBackward"
    }
}

/// Backward for matrix multiplication.
/// C = A @ B → dA = dC @ B^T, dB = A^T @ dC
pub struct MatmulBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
}

impl GradFn for MatmulBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let rhs_t = self.rhs.transpose().expect("MatmulBackward rhs transpose failed");
        let lhs_t = self.lhs.transpose().expect("MatmulBackward lhs transpose failed");

        let grad_a = grad_output.matmul(&rhs_t.contiguous()).expect("MatmulBackward grad_a failed");
        let grad_b = lhs_t.contiguous().matmul(grad_output).expect("MatmulBackward grad_b failed");

        vec![Some(grad_a), Some(grad_b)]
    }

    fn name(&self) -> &str {
        "MatmulBackward"
    }
}

/// Backward for ReLU activation.
pub struct ReluBackward {
    pub input: Tensor,
}

impl GradFn for ReluBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // grad * (input > 0)
        let mask = self.input.gt(&Tensor::zeros(self.input.shape().dims(), self.input.dtype()))
            .expect("ReluBackward mask failed");
        let grad = grad_output.mul(&mask).expect("ReluBackward mul failed");
        vec![Some(grad)]
    }

    fn name(&self) -> &str {
        "ReluBackward"
    }
}

/// Backward for sum reduction.
pub struct SumBackward {
    pub input_shape: Vec<usize>,
}

impl GradFn for SumBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // Gradient of sum is ones broadcast to input shape
        let numel: usize = self.input_shape.iter().product();
        let grad_val = grad_output.get_f32(0).unwrap_or(1.0);
        let data = vec![grad_val; numel];
        let grad = Tensor::from_f32(&data, &self.input_shape);
        vec![Some(grad)]
    }

    fn name(&self) -> &str {
        "SumBackward"
    }
}

/// Backward for mean reduction.
pub struct MeanBackward {
    pub input_shape: Vec<usize>,
}

impl GradFn for MeanBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let numel: usize = self.input_shape.iter().product();
        let grad_val = grad_output.get_f32(0).unwrap_or(1.0) / numel as f32;
        let data = vec![grad_val; numel];
        let grad = Tensor::from_f32(&data, &self.input_shape);
        vec![Some(grad)]
    }

    fn name(&self) -> &str {
        "MeanBackward"
    }
}

/// Backward for exp.
pub struct ExpBackward {
    pub output: Tensor,
}

impl GradFn for ExpBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // d/dx exp(x) = exp(x)
        let grad = grad_output.mul(&self.output).expect("ExpBackward mul failed");
        vec![Some(grad)]
    }

    fn name(&self) -> &str {
        "ExpBackward"
    }
}

/// Backward for log.
pub struct LogBackward {
    pub input: Tensor,
}

impl GradFn for LogBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // d/dx ln(x) = 1/x
        let recip = self.input.reciprocal().expect("LogBackward reciprocal failed");
        let grad = grad_output.mul(&recip).expect("LogBackward mul failed");
        vec![Some(grad)]
    }

    fn name(&self) -> &str {
        "LogBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_backward() {
        let grad = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        let grads = AddBackward.apply(&grad);
        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].as_ref().unwrap().as_f32_slice().unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(grads[1].as_ref().unwrap().as_f32_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_mul_backward() {
        let a = Tensor::from_f32(&[2.0, 3.0], &[2]);
        let b = Tensor::from_f32(&[4.0, 5.0], &[2]);
        let bw = MulBackward { lhs: a, rhs: b };

        let grad = Tensor::from_f32(&[1.0, 1.0], &[2]);
        let grads = bw.apply(&grad);
        // grad_a = grad * b = [4, 5], grad_b = grad * a = [2, 3]
        assert_eq!(grads[0].as_ref().unwrap().as_f32_slice().unwrap(), &[4.0, 5.0]);
        assert_eq!(grads[1].as_ref().unwrap().as_f32_slice().unwrap(), &[2.0, 3.0]);
    }

    #[test]
    fn test_matmul_backward() {
        // A=[2,3], B=[3,2], C=A@B=[2,2]
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let bw = MatmulBackward { lhs: a, rhs: b };

        let grad = Tensor::ones(&[2, 2]);
        let grads = bw.apply(&grad);

        // grad_a = dC @ B^T = [2,2] @ [2,3] = [2,3]
        assert_eq!(grads[0].as_ref().unwrap().shape().dims(), &[2, 3]);
        // grad_b = A^T @ dC = [3,2] @ [2,2] = [3,2]
        assert_eq!(grads[1].as_ref().unwrap().shape().dims(), &[3, 2]);
    }

    #[test]
    fn test_sum_backward() {
        let bw = SumBackward { input_shape: vec![2, 3] };
        let grad = Tensor::scalar(1.0);
        let grads = bw.apply(&grad);
        let g = grads[0].as_ref().unwrap();
        assert_eq!(g.shape().dims(), &[2, 3]);
        assert!(g.as_f32_slice().unwrap().iter().all(|&v| v == 1.0));
    }
}
