use std::collections::HashMap;

use kore_core::{Tensor, Result};

use crate::module::Module;

/// A container that holds sub-modules in an indexed list.
///
/// Unlike `Sequential`, `ModuleList` does **not** chain forward calls.
/// It is a bookkeeping container — you iterate over it manually in your
/// own `forward()` implementation.
///
/// # Example (Rust)
/// ```ignore
/// use kore_nn::{ModuleList, Linear, Module};
///
/// struct MyModel {
///     layers: ModuleList,
/// }
///
/// impl Module for MyModel {
///     fn forward(&self, input: &Tensor) -> Result<Tensor> {
///         let mut x = input.clone();
///         for i in 0..self.layers.len() {
///             x = self.layers[i].forward(&x)?;
///             // custom logic between layers …
///         }
///         Ok(x)
///     }
///     // …
/// }
/// ```
pub struct ModuleList {
    modules: Vec<Box<dyn Module>>,
    training: bool,
}

impl ModuleList {
    /// Create a new ModuleList from a vector of modules.
    pub fn new(modules: Vec<Box<dyn Module>>) -> Self {
        Self { modules, training: true }
    }

    /// Create an empty ModuleList.
    pub fn empty() -> Self {
        Self { modules: Vec::new(), training: true }
    }

    /// Append a module to the list.
    pub fn push(&mut self, module: Box<dyn Module>) {
        self.modules.push(module);
    }

    /// Number of sub-modules.
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Whether the list is empty.
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Get a reference to the module at the given index.
    pub fn get(&self, index: usize) -> Option<&dyn Module> {
        self.modules.get(index).map(|m| m.as_ref())
    }

    /// Get a mutable reference to the module at the given index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Box<dyn Module>> {
        self.modules.get_mut(index)
    }

    /// Iterate over sub-modules by reference.
    pub fn iter(&self) -> impl Iterator<Item = &Box<dyn Module>> {
        self.modules.iter()
    }

    /// Iterate over sub-modules by mutable reference.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Box<dyn Module>> {
        self.modules.iter_mut()
    }
}

impl std::ops::Index<usize> for ModuleList {
    type Output = dyn Module;
    fn index(&self, index: usize) -> &Self::Output {
        self.modules[index].as_ref()
    }
}

impl Module for ModuleList {
    fn clone_box(&self) -> Box<dyn Module> {
        let cloned: Vec<Box<dyn Module>> = self.modules.iter()
            .map(|m| m.clone_box())
            .collect();
        Box::new(ModuleList { modules: cloned, training: self.training })
    }

    /// Forward is **not** automatically chained. This is a no-op that returns
    /// the input unchanged. Use `ModuleList` inside your own `Module::forward`.
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.clone())
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.modules.iter().flat_map(|m| m.parameters()).collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        let mut params = Vec::new();
        for (i, module) in self.modules.iter().enumerate() {
            for (name, tensor) in module.named_parameters() {
                params.push((format!("{}.{}", i, name), tensor));
            }
        }
        params
    }

    fn set_parameters(&mut self, params: &[Tensor]) -> usize {
        let mut offset = 0;
        for module in &mut self.modules {
            offset += module.set_parameters(&params[offset..]);
        }
        offset
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
        for module in &mut self.modules {
            module.train(mode);
        }
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut dict = HashMap::new();
        for (i, module) in self.modules.iter().enumerate() {
            for (name, tensor) in module.named_parameters() {
                dict.insert(format!("{}.{}", i, name), tensor.clone());
            }
        }
        dict
    }
}

impl std::fmt::Display for ModuleList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ModuleList(")?;
        for (i, _) in self.modules.iter().enumerate() {
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
    fn test_module_list_parameters() {
        let list = ModuleList::new(vec![
            Box::new(Linear::new(4, 8, true)),
            Box::new(Linear::new(8, 2, false)),
        ]);
        // layer 0: weight + bias = 2, layer 1: weight = 1 → total 3
        assert_eq!(list.parameters().len(), 3);
    }

    #[test]
    fn test_module_list_index() {
        let list = ModuleList::new(vec![
            Box::new(Linear::new(4, 2, true)),
        ]);
        let input = Tensor::ones(&[1, 4]);
        let output = list[0].forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 2]);
    }

    #[test]
    fn test_module_list_push() {
        let mut list = ModuleList::empty();
        assert!(list.is_empty());
        list.push(Box::new(Linear::new(4, 2, true)));
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn test_module_list_train_propagation() {
        let mut list = ModuleList::new(vec![
            Box::new(Linear::new(4, 2, true)),
        ]);
        assert!(list.is_training());
        list.train(false);
        assert!(!list.is_training());
    }

    #[test]
    fn test_module_list_state_dict() {
        let list = ModuleList::new(vec![
            Box::new(Linear::new(4, 2, true)),
        ]);
        let sd = list.state_dict();
        assert!(sd.contains_key("0.weight"));
        assert!(sd.contains_key("0.bias"));
    }

    #[test]
    fn test_module_list_iter() {
        let list = ModuleList::new(vec![
            Box::new(Linear::new(4, 2, true)),
            Box::new(Linear::new(2, 1, false)),
        ]);
        let count = list.iter().count();
        assert_eq!(count, 2);
    }
}
