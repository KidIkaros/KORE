//! Gradient checkpointing for memory-efficient training.
//!
//! Instead of storing all intermediate activations during the forward pass,
//! gradient checkpointing saves only the inputs to checkpointed segments.
//! During backward, it recomputes the forward pass to regenerate activations.
//!
//! This trades compute (2× forward) for memory (O(sqrt(N)) instead of O(N)
//! for a chain of N layers).
//!
//! # Usage
//! ```ignore
//! use kore_autograd::checkpoint::checkpoint;
//!
//! // Instead of: let output = expensive_forward(&input);
//! // Use:        let output = checkpoint(|x| expensive_forward(x), input);
//! ```

use std::sync::Arc;

use kore_core::Tensor;
use kore_core::autograd::{GradFn, GradNode};

/// A recomputable function: takes a slice of input tensors, returns output tensors.
///
/// This is the function signature that `checkpoint` wraps. During backward,
/// this function is called again to recompute activations.
pub trait CheckpointFn: Send + Sync {
    fn forward(&self, inputs: &[Tensor]) -> Vec<Tensor>;
}

/// Implement CheckpointFn for closures.
impl<F> CheckpointFn for F
where
    F: Fn(&[Tensor]) -> Vec<Tensor> + Send + Sync,
{
    fn forward(&self, inputs: &[Tensor]) -> Vec<Tensor> {
        (self)(inputs)
    }
}

/// Gradient function that recomputes the forward pass during backward.
///
/// Stores the original inputs and the checkpoint function. When `apply` is
/// called during backward, it:
/// 1. Re-runs the forward pass with grad tracking enabled
/// 2. Runs backward on the recomputed outputs
/// 3. Returns the gradients w.r.t. the original inputs
struct CheckpointBackward {
    /// Saved inputs (detached, no grad tracking).
    saved_inputs: Vec<Tensor>,
    /// The function to recompute.
    func: Arc<dyn CheckpointFn>,
    /// Number of outputs the function produces.
    num_outputs: usize,
}

impl GradFn for CheckpointBackward {
    fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // Re-enable grad tracking for the recomputation
        let mut inputs_with_grad: Vec<Tensor> = self.saved_inputs.iter().map(|t| {
            let mut t = t.clone();
            t.set_requires_grad(true);
            t
        }).collect();

        // Recompute forward
        let outputs = self.func.forward(&inputs_with_grad);

        // Run backward on the recomputed outputs
        // For simplicity, we handle single-output case (most common)
        if let Some(output) = outputs.first() {
            if let Some(node) = output.grad_node() {
                kore_core::autograd::backward(&node, grad_output.clone());
            }
        }

        // Collect gradients from the recomputed inputs
        inputs_with_grad.iter().map(|t| {
            t.grad_node().and_then(|n| n.get_grad())
        }).collect()
    }

    fn name(&self) -> &str {
        "CheckpointBackward"
    }
}

/// Run a function with gradient checkpointing.
///
/// During the forward pass, the function is executed normally but intermediate
/// activations are not stored in the autograd graph. Instead, only the inputs
/// are saved. During backward, the function is re-executed to recompute
/// activations and compute gradients.
///
/// # Arguments
/// * `func` - The function to checkpoint (must be deterministic)
/// * `inputs` - Input tensors
///
/// # Returns
/// Output tensors with a `CheckpointBackward` grad function attached.
///
/// # Memory savings
/// For a chain of N layers, normal backprop stores O(N) activations.
/// With checkpointing every sqrt(N) layers, memory drops to O(sqrt(N)).
pub fn checkpoint<F>(func: F, inputs: Vec<Tensor>) -> Vec<Tensor>
where
    F: Fn(&[Tensor]) -> Vec<Tensor> + Send + Sync + 'static,
{
    let func = Arc::new(func);

    // Save detached copies of inputs (no grad graph references)
    let saved_inputs: Vec<Tensor> = inputs.iter().map(|t| {
        let data = t.contiguous();
        let slice = data.as_f32_slice()
            .expect("checkpoint: input tensor must be F32");
        Tensor::from_f32(slice, data.shape().dims())
    }).collect();

    // Run forward without grad tracking to avoid building intermediate graph
    let outputs = {
        let _guard = kore_core::autograd::NoGradGuard::new();
        func.forward(&inputs)
    };

    // Attach CheckpointBackward to outputs so backward will recompute
    let num_outputs = outputs.len();
    let grad_fn = Box::new(CheckpointBackward {
        saved_inputs,
        func,
        num_outputs,
    });

    // Collect input grad nodes for the graph edge
    let input_nodes: Vec<Arc<GradNode>> = inputs.iter()
        .filter_map(|t| t.grad_node().cloned())
        .collect();

    if input_nodes.is_empty() {
        return outputs;
    }

    let checkpoint_node = GradNode::with_grad_fn(grad_fn, input_nodes);

    // Wrap outputs with the checkpoint node
    outputs.into_iter().map(|t| {
        t.with_grad_node(checkpoint_node.clone())
    }).collect()
}

/// Convenience: checkpoint a single-input, single-output function.
pub fn checkpoint_fn<F>(func: F, input: Tensor) -> Tensor
where
    F: Fn(&Tensor) -> Tensor + Send + Sync + 'static,
{
    let results = checkpoint(
        move |inputs: &[Tensor]| vec![func(&inputs[0])],
        vec![input],
    );
    results.into_iter().next()
        .expect("checkpoint_fn: function must produce at least one output")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_forward_matches() {
        // Verify that checkpoint produces the same forward output
        let input = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let direct = {
            let squared = input.mul(&input).unwrap();
            squared.sum().unwrap()
        };

        let checkpointed = checkpoint_fn(|x| {
            let squared = x.mul(x).unwrap();
            squared.sum().unwrap()
        }, input.clone());

        let d_data = direct.as_f32_slice().unwrap();
        let c_data = checkpointed.as_f32_slice().unwrap();
        assert!((d_data[0] - c_data[0]).abs() < 1e-6,
            "direct={}, checkpointed={}", d_data[0], c_data[0]);
    }

    #[test]
    fn test_checkpoint_shape_preserved() {
        let input = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

        let output = checkpoint_fn(|x| {
            x.mul(x).unwrap()
        }, input);

        assert_eq!(output.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_checkpoint_multi_input() {
        let a = Tensor::from_f32(&[1.0, 2.0], &[2]);
        let b = Tensor::from_f32(&[3.0, 4.0], &[2]);

        let outputs = checkpoint(
            |inputs: &[Tensor]| {
                let sum = inputs[0].add(&inputs[1]).unwrap();
                vec![sum]
            },
            vec![a, b],
        );

        assert_eq!(outputs.len(), 1);
        let data = outputs[0].as_f32_slice().unwrap();
        assert!((data[0] - 4.0).abs() < 1e-6);
        assert!((data[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_checkpoint_no_grad_inputs() {
        // When inputs don't require grad, checkpoint should still work
        let input = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);

        let output = checkpoint_fn(|x| {
            x.add(x).unwrap()
        }, input);

        let data = output.as_f32_slice().unwrap();
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_checkpoint_fn_convenience() {
        let input = Tensor::from_f32(&[2.0, 3.0], &[2]);
        let output = checkpoint_fn(|x| x.mul(x).unwrap(), input);
        let data = output.as_f32_slice().unwrap();
        assert!((data[0] - 4.0).abs() < 1e-6);
        assert!((data[1] - 9.0).abs() < 1e-6);
    }
}
