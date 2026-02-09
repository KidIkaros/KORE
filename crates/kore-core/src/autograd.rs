//! Core autograd types for automatic differentiation.
//!
//! Defines the `GradFn` trait and `GradNode` computation graph node.
//! These live in kore-core so that `Tensor` can carry gradient tracking
//! without circular dependencies.

use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::RwLock;

use crate::tensor::Tensor;

// ============================================================================
// GradFn trait
// ============================================================================

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
// GradNode
// ============================================================================

static NEXT_NODE_ID: AtomicUsize = AtomicUsize::new(0);

fn next_id() -> usize {
    NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed)
}

/// A node in the autograd computation graph.
///
/// Each node holds:
/// - An optional gradient function for backward
/// - Weak references to input nodes (prevents cycles)
/// - Thread-safe accumulated gradient
pub struct GradNode {
    pub id: usize,
    pub grad_fn: Option<Box<dyn GradFn>>,
    pub inputs: Vec<Weak<GradNode>>,
    pub grad: RwLock<Option<Tensor>>,
}

impl GradNode {
    /// Create a new leaf node (no grad_fn, e.g., a parameter).
    pub fn leaf() -> Arc<Self> {
        Arc::new(Self {
            id: next_id(),
            grad_fn: None,
            inputs: Vec::new(),
            grad: RwLock::new(None),
        })
    }

    /// Create a new interior node with a gradient function and inputs.
    pub fn with_grad_fn(
        grad_fn: Box<dyn GradFn>,
        inputs: Vec<Arc<GradNode>>,
    ) -> Arc<Self> {
        let weak_inputs = inputs.iter().map(Arc::downgrade).collect();
        Arc::new(Self {
            id: next_id(),
            grad_fn: Some(grad_fn),
            inputs: weak_inputs,
            grad: RwLock::new(None),
        })
    }

    /// Whether this is a leaf node (no grad_fn).
    pub fn is_leaf(&self) -> bool {
        self.grad_fn.is_none()
    }

    /// Accumulate gradient into this node (thread-safe).
    pub fn accumulate_grad(&self, grad: &Tensor) {
        let mut lock = self.grad.write();
        match lock.as_ref() {
            Some(existing) => {
                *lock = Some(existing.add(grad).expect("Gradient accumulation failed"));
            }
            None => {
                *lock = Some(grad.clone());
            }
        }
    }

    /// Get the current accumulated gradient.
    pub fn get_grad(&self) -> Option<Tensor> {
        self.grad.read().clone()
    }

    /// Clear the accumulated gradient.
    pub fn zero_grad(&self) {
        *self.grad.write() = None;
    }
}

// ============================================================================
// Backward pass
// ============================================================================

/// Execute the backward pass from a root node.
///
/// Performs a BFS traversal of the computation graph and
/// propagates gradients from the root to all leaf nodes.
pub fn backward(root: &Arc<GradNode>, grad_output: Tensor) {
    root.accumulate_grad(&grad_output);

    let sorted = topological_sort(root);

    for node in &sorted {
        if let Some(ref grad_fn) = node.grad_fn {
            let node_grad = match node.get_grad() {
                Some(g) => g,
                None => continue,
            };

            let input_grads = grad_fn.apply(&node_grad);

            for (weak_input, maybe_grad) in node.inputs.iter().zip(input_grads.into_iter()) {
                if let (Some(input_node), Some(grad)) = (weak_input.upgrade(), maybe_grad) {
                    input_node.accumulate_grad(&grad);
                }
            }
        }
    }
}

/// BFS topological sort from root → leaves (backward traversal order).
fn topological_sort(root: &Arc<GradNode>) -> Vec<Arc<GradNode>> {
    let mut sorted = Vec::new();
    let mut visited = HashSet::new();
    let mut queue: VecDeque<Arc<GradNode>> = VecDeque::new();

    queue.push_back(Arc::clone(root));
    visited.insert(root.id);

    while let Some(node) = queue.pop_front() {
        sorted.push(Arc::clone(&node));
        for weak_input in &node.inputs {
            if let Some(input) = weak_input.upgrade() {
                if visited.insert(input.id) {
                    queue.push_back(input);
                }
            }
        }
    }

    sorted
}

// ============================================================================
// No-grad scope
// ============================================================================

use std::cell::Cell;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Check if gradient computation is currently enabled.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|g| g.get())
}

/// Set whether gradient computation is enabled.
fn set_grad_enabled(enabled: bool) -> bool {
    GRAD_ENABLED.with(|g| {
        let prev = g.get();
        g.set(enabled);
        prev
    })
}

/// RAII guard that disables gradient computation in its scope.
///
/// # Example
/// ```ignore
/// let _guard = NoGradGuard::new();
/// // All operations here skip gradient tracking
/// // Gradients re-enabled when guard is dropped
/// ```
pub struct NoGradGuard {
    prev: bool,
}

impl NoGradGuard {
    pub fn new() -> Self {
        let prev = set_grad_enabled(false);
        Self { prev }
    }
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        set_grad_enabled(self.prev);
    }
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
    fn name(&self) -> &str { "AddBackward" }
}

/// Backward for element-wise subtraction.
pub struct SubBackward;

impl GradFn for SubBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let neg = grad_output.neg().expect("SubBackward neg failed");
        vec![Some(grad_output.clone()), Some(neg)]
    }
    fn name(&self) -> &str { "SubBackward" }
}

/// Backward for element-wise multiplication.
pub struct MulBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
}

impl GradFn for MulBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let grad_a = grad_output.mul(&self.rhs).expect("MulBackward grad_a failed");
        let grad_b = grad_output.mul(&self.lhs).expect("MulBackward grad_b failed");
        vec![Some(grad_a), Some(grad_b)]
    }
    fn name(&self) -> &str { "MulBackward" }
}

/// Backward for element-wise division.
pub struct DivBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
}

impl GradFn for DivBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // d/da (a/b) = 1/b, d/db (a/b) = -a/b^2
        let grad_a = grad_output.div(&self.rhs).expect("DivBackward grad_a failed");
        let b_sq = self.rhs.mul(&self.rhs).expect("DivBackward b^2 failed");
        let neg_a = self.lhs.neg().expect("DivBackward neg failed");
        let grad_b = grad_output.mul(&neg_a.div(&b_sq).expect("DivBackward -a/b^2 failed"))
            .expect("DivBackward grad_b failed");
        vec![Some(grad_a), Some(grad_b)]
    }
    fn name(&self) -> &str { "DivBackward" }
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
    fn name(&self) -> &str { "MatmulBackward" }
}

/// Backward for negation.
pub struct NegBackward;

impl GradFn for NegBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(grad_output.neg().expect("NegBackward failed"))]
    }
    fn name(&self) -> &str { "NegBackward" }
}

/// Backward for exp.
pub struct ExpBackward {
    pub output: Tensor,
}

impl GradFn for ExpBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let grad = grad_output.mul(&self.output).expect("ExpBackward mul failed");
        vec![Some(grad)]
    }
    fn name(&self) -> &str { "ExpBackward" }
}

/// Backward for log.
pub struct LogBackward {
    pub input: Tensor,
}

impl GradFn for LogBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let recip = self.input.reciprocal().expect("LogBackward reciprocal failed");
        let grad = grad_output.mul(&recip).expect("LogBackward mul failed");
        vec![Some(grad)]
    }
    fn name(&self) -> &str { "LogBackward" }
}

/// Backward for sqrt.
pub struct SqrtBackward {
    pub output: Tensor,
}

impl GradFn for SqrtBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // d/dx sqrt(x) = 1 / (2 * sqrt(x))
        let two_sqrt = self.output.mul_scalar(2.0).expect("SqrtBackward mul failed");
        let grad = grad_output.div(&two_sqrt).expect("SqrtBackward div failed");
        vec![Some(grad)]
    }
    fn name(&self) -> &str { "SqrtBackward" }
}

/// Backward for abs.
pub struct AbsBackward {
    pub input: Tensor,
}

impl GradFn for AbsBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // d/dx |x| = sign(x)
        let zeros = Tensor::zeros(self.input.shape().dims(), self.input.dtype());
        let pos = self.input.gt(&zeros).expect("AbsBackward gt failed");
        let neg = zeros.gt(&self.input).expect("AbsBackward lt failed");
        // sign = pos - neg (1 where positive, -1 where negative, 0 at zero)
        let sign = pos.sub(&neg).expect("AbsBackward sub failed");
        let grad = grad_output.mul(&sign).expect("AbsBackward mul failed");
        vec![Some(grad)]
    }
    fn name(&self) -> &str { "AbsBackward" }
}

/// Backward for sum reduction.
pub struct SumBackward {
    pub input_shape: Vec<usize>,
}

impl GradFn for SumBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let numel: usize = self.input_shape.iter().product();
        let grad_val = grad_output.get_f32(0).unwrap_or(1.0);
        let data = vec![grad_val; numel];
        let grad = Tensor::from_f32(&data, &self.input_shape);
        vec![Some(grad)]
    }
    fn name(&self) -> &str { "SumBackward" }
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
    fn name(&self) -> &str { "MeanBackward" }
}

/// Backward for scalar addition: grad flows through unchanged (1 input).
pub struct AddScalarBackward;

impl GradFn for AddScalarBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(grad_output.clone())]
    }
    fn name(&self) -> &str { "AddScalarBackward" }
}

/// Backward for scalar multiplication.
pub struct MulScalarBackward {
    pub scalar: f32,
}

impl GradFn for MulScalarBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let grad = grad_output.mul_scalar(self.scalar).expect("MulScalarBackward failed");
        vec![Some(grad)]
    }
    fn name(&self) -> &str { "MulScalarBackward" }
}

/// Backward for pow_scalar: d/dx x^n = n * x^(n-1)
pub struct PowScalarBackward {
    pub input: Tensor,
    pub exponent: f32,
}

impl GradFn for PowScalarBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let inner = self.input.pow_scalar(self.exponent - 1.0)
            .expect("PowScalarBackward pow failed")
            .mul_scalar(self.exponent)
            .expect("PowScalarBackward mul failed");
        let grad = grad_output.mul(&inner).expect("PowScalarBackward outer mul failed");
        vec![Some(grad)]
    }
    fn name(&self) -> &str { "PowScalarBackward" }
}

/// Backward for clamp.
pub struct ClampBackward {
    pub input: Tensor,
    pub min: f32,
    pub max: f32,
}

impl GradFn for ClampBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // Gradient is 1 where input is in [min, max], 0 otherwise
        let min_t = Tensor::from_f32(&[self.min], &[1]);
        let max_t = Tensor::from_f32(&[self.max], &[1]);
        let above_min = self.input.ge(&min_t).expect("ClampBackward ge failed");
        let below_max = max_t.ge(&self.input).expect("ClampBackward le failed");
        let mask = above_min.mul(&below_max).expect("ClampBackward mask failed");
        let grad = grad_output.mul(&mask).expect("ClampBackward mul failed");
        vec![Some(grad)]
    }
    fn name(&self) -> &str { "ClampBackward" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leaf_node() {
        let node = GradNode::leaf();
        assert!(node.is_leaf());
        assert!(node.get_grad().is_none());
    }

    #[test]
    fn test_grad_accumulation() {
        let node = GradNode::leaf();
        let g1 = Tensor::from_f32(&[1.0, 2.0], &[2]);
        let g2 = Tensor::from_f32(&[3.0, 4.0], &[2]);
        node.accumulate_grad(&g1);
        node.accumulate_grad(&g2);
        let grad = node.get_grad().unwrap();
        assert_eq!(grad.as_f32_slice().unwrap(), &[4.0, 6.0]);
    }

    #[test]
    fn test_no_grad_guard() {
        assert!(is_grad_enabled());
        {
            let _guard = NoGradGuard::new();
            assert!(!is_grad_enabled());
        }
        assert!(is_grad_enabled());
    }

    #[test]
    fn test_backward_add() {
        let a = GradNode::leaf();
        let b = GradNode::leaf();
        let c = GradNode::with_grad_fn(
            Box::new(AddBackward),
            vec![Arc::clone(&a), Arc::clone(&b)],
        );
        backward(&c, Tensor::scalar(1.0));
        assert_eq!(a.get_grad().unwrap().get_f32(0).unwrap(), 1.0);
        assert_eq!(b.get_grad().unwrap().get_f32(0).unwrap(), 1.0);
    }

    #[test]
    fn test_backward_mul() {
        let a = GradNode::leaf();
        let b = GradNode::leaf();
        let c = GradNode::with_grad_fn(
            Box::new(MulBackward {
                lhs: Tensor::scalar(3.0),
                rhs: Tensor::scalar(4.0),
            }),
            vec![Arc::clone(&a), Arc::clone(&b)],
        );
        backward(&c, Tensor::scalar(1.0));
        assert_eq!(a.get_grad().unwrap().get_f32(0).unwrap(), 4.0);
        assert_eq!(b.get_grad().unwrap().get_f32(0).unwrap(), 3.0);
    }

    #[test]
    fn test_backward_chain() {
        // d = (a + b) * b, a=2, b=3
        let a = GradNode::leaf();
        let b = GradNode::leaf();
        let c = GradNode::with_grad_fn(
            Box::new(AddBackward),
            vec![Arc::clone(&a), Arc::clone(&b)],
        );
        let d = GradNode::with_grad_fn(
            Box::new(MulBackward {
                lhs: Tensor::scalar(5.0),
                rhs: Tensor::scalar(3.0),
            }),
            vec![Arc::clone(&c), Arc::clone(&b)],
        );
        backward(&d, Tensor::scalar(1.0));
        assert_eq!(a.get_grad().unwrap().get_f32(0).unwrap(), 3.0);
        assert_eq!(b.get_grad().unwrap().get_f32(0).unwrap(), 8.0);
    }
}
