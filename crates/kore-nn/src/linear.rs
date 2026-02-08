use kore_core::{DType, Tensor};

use crate::module::Module;

/// Fully connected linear layer: y = x @ W^T + b
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
    training: bool,
}

impl Linear {
    /// Create a new Linear layer with Xavier initialization.
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        // Xavier uniform initialization: U(-sqrt(6/(in+out)), sqrt(6/(in+out)))
        let limit = (6.0 / (in_features + out_features) as f32).sqrt();
        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|i| {
                // Simple deterministic pseudo-random for reproducibility
                let x = ((i as f32 * 0.618034) % 1.0) * 2.0 - 1.0;
                x * limit
            })
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

    /// Get the weight tensor.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Get the bias tensor (if present).
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        // y = x @ W^T
        let wt = self.weight.transpose().expect("Linear weight transpose failed");
        let mut output = input.matmul(&wt.contiguous()).expect("Linear matmul failed");

        // + bias
        if let Some(ref bias) = self.bias {
            output = output.add(bias).expect("Linear bias add failed");
        }

        output
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(&str, &Tensor)> {
        let mut params = vec![("weight", &self.weight)];
        if let Some(ref b) = self.bias {
            params.push(("bias", b));
        }
        params
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
        let output = layer.forward(&input);
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
