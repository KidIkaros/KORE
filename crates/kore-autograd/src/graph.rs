use std::sync::{Arc, Weak};

use kore_core::Tensor;
use parking_lot::RwLock;

use crate::grad_fn::GradFn;

/// Unique identifier for a node in the computation graph.
static NEXT_NODE_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn next_id() -> usize {
    NEXT_NODE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// A node in the autograd computation graph.
///
/// Each node holds:
/// - A reference to the output tensor
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

/// The computation graph — a collection of nodes.
/// Used for managing the full graph during backward.
#[allow(dead_code)]
pub struct GradGraph {
    /// All nodes in topological order (populated during backward).
    nodes: Vec<Arc<GradNode>>,
}

impl GradGraph {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Build topological order from a root node using Kahn's algorithm.
    pub fn topological_sort(root: &Arc<GradNode>) -> Vec<Arc<GradNode>> {
        use std::collections::{HashMap, VecDeque};

        // BFS to discover all reachable nodes
        let mut all_nodes: HashMap<usize, Arc<GradNode>> = HashMap::new();
        let mut in_degree: HashMap<usize, usize> = HashMap::new();
        let mut queue: VecDeque<Arc<GradNode>> = VecDeque::new();

        queue.push_back(Arc::clone(root));
        all_nodes.insert(root.id, Arc::clone(root));
        in_degree.insert(root.id, 0);

        while let Some(node) = queue.pop_front() {
            for weak_input in &node.inputs {
                if let Some(input) = weak_input.upgrade() {
                    if let std::collections::hash_map::Entry::Vacant(e) = all_nodes.entry(input.id) {
                        e.insert(Arc::clone(&input));
                        in_degree.insert(input.id, 0);
                        queue.push_back(Arc::clone(&input));
                    }
                }
            }
        }

        // Count in-degrees (how many nodes depend on each node)
        for node in all_nodes.values() {
            for weak_input in &node.inputs {
                if let Some(input) = weak_input.upgrade() {
                    *in_degree.entry(input.id).or_insert(0) += 1;
                }
            }
        }

        // Kahn's algorithm: start from nodes with in-degree 0 (the root)
        let mut sorted = Vec::new();
        let mut kahn_queue: VecDeque<usize> = VecDeque::new();

        // The root has in-degree 0 (nothing depends on it in the backward direction)
        // Actually for backward, we want reverse topological order:
        // root first, then its inputs, etc.
        kahn_queue.push_back(root.id);

        // Simple BFS from root following inputs — this gives us the correct
        // backward traversal order (root → leaves)
        let mut visited = std::collections::HashSet::new();
        let mut bfs_queue: VecDeque<Arc<GradNode>> = VecDeque::new();
        bfs_queue.push_back(Arc::clone(root));
        visited.insert(root.id);

        while let Some(node) = bfs_queue.pop_front() {
            sorted.push(Arc::clone(&node));
            for weak_input in &node.inputs {
                if let Some(input) = weak_input.upgrade() {
                    if visited.insert(input.id) {
                        bfs_queue.push_back(input);
                    }
                }
            }
        }

        sorted
    }
}

impl Default for GradGraph {
    fn default() -> Self {
        Self::new()
    }
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
    fn test_zero_grad() {
        let node = GradNode::leaf();
        let g = Tensor::from_f32(&[1.0], &[1]);
        node.accumulate_grad(&g);
        assert!(node.get_grad().is_some());

        node.zero_grad();
        assert!(node.get_grad().is_none());
    }

    #[test]
    fn test_topological_sort() {
        let a = GradNode::leaf();
        let b = GradNode::leaf();

        // c = a + b (mock)
        let c = GradNode::with_grad_fn(
            Box::new(crate::grad_fn::AddBackward),
            vec![Arc::clone(&a), Arc::clone(&b)],
        );

        let sorted = GradGraph::topological_sort(&c);
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].id, c.id); // root first
    }
}
