use std::collections::HashMap;

use kore_core::{Tensor, Result};

use crate::module::Module;

/// An ordered container that chains modules sequentially.
///
/// The output of each module is fed as input to the next.
///
/// # Example (Rust)
/// ```ignore
/// use kore_nn::{Sequential, Linear};
///
/// let model = Sequential::new(vec![
///     Box::new(Linear::new(784, 128, true)),
///     Box::new(Linear::new(128, 10, true)),
/// ]);
/// let output = model.forward(&input)?;
/// ```
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
    training: bool,
}

impl Sequential {
    /// Create a new Sequential container from an ordered list of modules.
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self { layers, training: true }
    }

    /// Create an empty Sequential container.
    pub fn empty() -> Self {
        Self { layers: Vec::new(), training: true }
    }

    /// Append a module to the end of the sequence.
    pub fn push(&mut self, module: Box<dyn Module>) {
        self.layers.push(module);
    }

    /// Number of sub-modules.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Whether the container is empty.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Get a reference to the module at the given index.
    pub fn get(&self, index: usize) -> Option<&dyn Module> {
        self.layers.get(index).map(|m| m.as_ref())
    }

    /// Get a mutable reference to the module at the given index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Box<dyn Module>> {
        self.layers.get_mut(index)
    }

    /// Deep-clone this Sequential, preserving all internal layer state
    /// (including packed quantized weights for BitLinear/QuatLinear).
    pub fn deep_clone(&self) -> Self {
        let cloned_layers: Vec<Box<dyn Module>> = self.layers.iter()
            .map(|l| l.clone_box())
            .collect();
        Self { layers: cloned_layers, training: self.training }
    }
}

impl Module for Sequential {
    fn clone_box(&self) -> Box<dyn Module> {
        let cloned_layers: Vec<Box<dyn Module>> = self.layers.iter()
            .map(|l| l.clone_box())
            .collect();
        Box::new(Sequential { layers: cloned_layers, training: self.training })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.layers.iter().flat_map(|m| m.parameters()).collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        let mut params = Vec::new();
        for (i, module) in self.layers.iter().enumerate() {
            for (name, tensor) in module.named_parameters() {
                params.push((format!("{}.{}", i, name), tensor));
            }
        }
        params
    }

    fn set_parameters(&mut self, params: &[Tensor]) -> usize {
        let mut offset = 0;
        for layer in &mut self.layers {
            offset += layer.set_parameters(&params[offset..]);
        }
        offset
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
        for layer in &mut self.layers {
            layer.train(mode);
        }
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut dict = HashMap::new();
        for (i, module) in self.layers.iter().enumerate() {
            for (name, tensor) in module.named_parameters() {
                dict.insert(format!("{}.{}", i, name), tensor.clone());
            }
        }
        dict
    }
}

impl std::fmt::Display for Sequential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Sequential(")?;
        for (i, _) in self.layers.iter().enumerate() {
            writeln!(f, "  ({i}): <module>")?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Linear;

    #[test]
    fn test_sequential_forward() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(4, 8, true)),
            Box::new(Linear::new(8, 2, true)),
        ]);
        let input = Tensor::ones(&[1, 4]);
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 2]);
    }

    #[test]
    fn test_sequential_parameters() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(4, 8, true)),
            Box::new(Linear::new(8, 2, false)),
        ]);
        // layer 0: weight + bias = 2, layer 1: weight = 1 → total 3
        assert_eq!(model.parameters().len(), 3);
    }

    #[test]
    fn test_sequential_empty() {
        let model = Sequential::empty();
        assert!(model.is_empty());
        assert_eq!(model.len(), 0);
        let input = Tensor::ones(&[1, 4]);
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 4]);
    }

    #[test]
    fn test_sequential_push() {
        let mut model = Sequential::empty();
        model.push(Box::new(Linear::new(4, 2, true)));
        assert_eq!(model.len(), 1);
        let input = Tensor::ones(&[1, 4]);
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 2]);
    }

    #[test]
    fn test_sequential_train_propagation() {
        let mut model = Sequential::new(vec![
            Box::new(Linear::new(4, 2, true)),
        ]);
        assert!(model.is_training());
        model.train(false);
        assert!(!model.is_training());
    }

    #[test]
    fn test_sequential_state_dict() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(4, 2, true)),
        ]);
        let sd = model.state_dict();
        assert!(sd.contains_key("0.weight"));
        assert!(sd.contains_key("0.bias"));
    }

    #[test]
    fn test_sequential_named_parameters() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(4, 2, true)),
            Box::new(Linear::new(2, 1, false)),
        ]);
        let np = model.named_parameters();
        let names: Vec<&str> = np.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"0.weight"));
        assert!(names.contains(&"0.bias"));
        assert!(names.contains(&"1.weight"));
    }
}
