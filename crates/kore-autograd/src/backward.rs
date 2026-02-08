//! Backward pass execution.

use std::sync::Arc;

use kore_core::Tensor;

use crate::graph::{GradGraph, GradNode};

/// Execute the backward pass from a root node.
///
/// This performs a topological sort of the computation graph and
/// propagates gradients from the root to all leaf nodes.
///
/// # Arguments
/// * `root` - The output node to differentiate (typically a scalar loss)
/// * `grad_output` - The initial gradient (typically ones for a scalar loss)
pub fn backward(root: &Arc<GradNode>, grad_output: Tensor) {
    // Seed the root gradient
    root.accumulate_grad(&grad_output);

    // Get nodes in topological order (root first, leaves last)
    let sorted = GradGraph::topological_sort(root);

    // Propagate gradients
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grad_fn::{AddBackward, MulBackward};

    #[test]
    fn test_backward_simple_add() {
        // c = a + b
        let a = GradNode::leaf();
        let b = GradNode::leaf();
        let c = GradNode::with_grad_fn(
            Box::new(AddBackward),
            vec![Arc::clone(&a), Arc::clone(&b)],
        );

        // dc/dc = 1
        backward(&c, Tensor::scalar(1.0));

        // dc/da = 1, dc/db = 1
        let ga = a.get_grad().unwrap();
        let gb = b.get_grad().unwrap();
        assert_eq!(ga.get_f32(0).unwrap(), 1.0);
        assert_eq!(gb.get_f32(0).unwrap(), 1.0);
    }

    #[test]
    fn test_backward_mul() {
        // c = a * b where a=3, b=4
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

        // dc/da = b = 4, dc/db = a = 3
        let ga = a.get_grad().unwrap();
        let gb = b.get_grad().unwrap();
        assert_eq!(ga.get_f32(0).unwrap(), 4.0);
        assert_eq!(gb.get_f32(0).unwrap(), 3.0);
    }

    #[test]
    fn test_backward_chain() {
        // d = (a + b) * b
        // where a=2, b=3
        let a = GradNode::leaf();
        let b = GradNode::leaf();

        // c = a + b
        let c = GradNode::with_grad_fn(
            Box::new(AddBackward),
            vec![Arc::clone(&a), Arc::clone(&b)],
        );

        // d = c * b (c=5, b=3)
        let d = GradNode::with_grad_fn(
            Box::new(MulBackward {
                lhs: Tensor::scalar(5.0), // c = a+b = 2+3 = 5
                rhs: Tensor::scalar(3.0), // b = 3
            }),
            vec![Arc::clone(&c), Arc::clone(&b)],
        );

        backward(&d, Tensor::scalar(1.0));

        // d = (a+b)*b = ab + b^2
        // dd/da = b = 3
        // dd/db = a + 2b = 2 + 6 = 8
        // But through our graph: dd/dc = b = 3, dd/db_direct = c = 5
        // dc/da = 1, dc/db = 1
        // So dd/da = dd/dc * dc/da = 3 * 1 = 3 ✓
        // dd/db = dd/dc * dc/db + dd/db_direct = 3*1 + 5 = 8 ✓
        let ga = a.get_grad().unwrap();
        let gb = b.get_grad().unwrap();
        assert_eq!(ga.get_f32(0).unwrap(), 3.0);
        assert_eq!(gb.get_f32(0).unwrap(), 8.0);
    }
}
