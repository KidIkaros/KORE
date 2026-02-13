use rand::Rng;

use kore_core::{DType, Tensor};

use crate::module::Module;

/// Fully connected linear layer: y = x @ W^T + b
#[derive(Clone)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
    training: bool,
}

impl Linear {
    /// Create a new Linear layer with Xavier uniform initialization.
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        // Xavier uniform: U(-sqrt(6/(in+out)), sqrt(6/(in+out)))
        let limit = (6.0 / (in_features + out_features) as f32).sqrt();
        let mut rng = rand::thread_rng();
        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| rng.gen_range(-limit..limit))
            .collect();

        let mut weight = Tensor::from_f32(&weight_data, &[out_features, in_features]);
        weight.set_requires_grad(true);

        let bias_tensor = if bias {
            let mut b = Tensor::zeros(&[out_features], DType::F32);
            b.set_requires_grad(true);
            Some(b)
        } else {
            None
        };

        Self {
            weight,
            bias: bias_tensor,
            training: true,
        }
    }

    /// Create a Linear layer from existing weight and optional bias tensors.
    pub fn from_weight(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias, training: false }
    }

    /// Get the weight tensor.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Get the bias tensor (if present).
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// Input feature dimension.
    pub fn in_features(&self) -> usize {
        self.weight.shape().dims()[1]
    }

    /// Output feature dimension.
    pub fn out_features(&self) -> usize {
        self.weight.shape().dims()[0]
    }

    /// Whether this layer has a bias term.
    pub fn has_bias(&self) -> bool {
        self.bias.is_some()
    }
}

impl Module for Linear {
    fn clone_box(&self) -> Box<dyn Module> { Box::new(self.clone()) }

    fn forward(&self, input: &Tensor) -> kore_core::Result<Tensor> {
        // y = x @ W^T
        let wt = self.weight.transpose()?;
        let mut output = input.matmul(&wt.contiguous())?;

        // + bias
        if let Some(ref bias) = self.bias {
            output = output.add(bias)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        let mut params = vec![("weight".into(), &self.weight)];
        if let Some(ref b) = self.bias {
            params.push(("bias".into(), b));
        }
        params
    }

    fn set_parameters(&mut self, params: &[Tensor]) -> usize {
        let mut n = 0;
        self.weight = params[n].clone(); n += 1;
        if self.bias.is_some() {
            self.bias = Some(params[n].clone()); n += 1;
        }
        n
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_creation() {
        let layer = Linear::new(4, 3, true);
        assert_eq!(layer.weight().shape().dims(), &[3, 4]);
        assert_eq!(layer.bias().unwrap().shape().dims(), &[3]);
        assert!(layer.weight().requires_grad());
    }

    #[test]
    fn test_linear_forward() {
        let layer = Linear::new(3, 2, false);
        let input = Tensor::ones(&[1, 3]);
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 2]);
    }

    #[test]
    fn test_linear_parameters() {
        let layer = Linear::new(4, 3, true);
        assert_eq!(layer.parameters().len(), 2); // weight + bias

        let layer_no_bias = Linear::new(4, 3, false);
        assert_eq!(layer_no_bias.parameters().len(), 1);
    }

    #[test]
    fn test_linear_state_dict() {
        let layer = Linear::new(4, 3, true);
        let sd = layer.state_dict();
        assert!(sd.contains_key("weight"));
        assert!(sd.contains_key("bias"));
    }
}
