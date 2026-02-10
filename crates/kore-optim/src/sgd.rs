use kore_core::Tensor;

/// Stochastic Gradient Descent optimizer with momentum and weight decay.
pub struct SGD {
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    velocities: Vec<Tensor>,
    initialized: bool,
}

impl SGD {
    pub fn new(lr: f32, momentum: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            momentum,
            weight_decay,
            velocities: Vec::new(),
            initialized: false,
        }
    }

    /// Learning rate.
    pub fn lr(&self) -> f32 { self.lr }

    /// Momentum coefficient.
    pub fn momentum(&self) -> f32 { self.momentum }

    /// Weight decay coefficient.
    pub fn weight_decay(&self) -> f32 { self.weight_decay }

    /// Perform one optimization step.
    ///
    /// `params` and `grads` must be the same length and correspond 1:1.
    pub fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        if !self.initialized {
            self.velocities = grads
                .iter()
                .map(|g| Tensor::zeros(g.shape().dims(), g.dtype()))
                .collect();
            self.initialized = true;
        }

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let mut g = grad.clone();

            // Weight decay: g = g + wd * param
            if self.weight_decay > 0.0 {
                let decay = param.mul_scalar(self.weight_decay).expect("SGD weight decay failed");
                g = g.add(&decay).expect("SGD weight decay add failed");
            }

            // Momentum: v = momentum * v + g
            if self.momentum > 0.0 {
                let v = self.velocities[i]
                    .mul_scalar(self.momentum)
                    .expect("SGD momentum mul failed")
                    .add(&g)
                    .expect("SGD momentum add failed");
                self.velocities[i] = v.clone();
                g = v;
            }

            // param = param - lr * g
            let update = g.mul_scalar(self.lr).expect("SGD lr mul failed");
            let new_param = param.sub(&update).expect("SGD param update failed");

            // Copy new values into param
            if let (Some(dst), Some(src)) = (param.as_f32_slice_mut(), new_param.as_f32_slice()) {
                dst.copy_from_slice(src);
            }
        }
    }

    /// Zero all accumulated gradients.
    pub fn zero_grad(&mut self) {
        // Gradients are managed externally; this is a no-op placeholder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_step() {
        let mut params = vec![Tensor::from_f32(&[1.0, 2.0, 3.0], &[3])];
        let grads = vec![Tensor::from_f32(&[0.1, 0.2, 0.3], &[3])];

        let mut opt = SGD::new(0.1, 0.0, 0.0);
        opt.step(&mut params, &grads);

        let data = params[0].as_f32_slice().unwrap();
        assert!((data[0] - 0.99).abs() < 1e-6);
        assert!((data[1] - 1.98).abs() < 1e-6);
        assert!((data[2] - 2.97).abs() < 1e-6);
    }
}
