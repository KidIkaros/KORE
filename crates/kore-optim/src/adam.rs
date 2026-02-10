use kore_core::Tensor;

/// Adam optimizer with decoupled weight decay (AdamW).
pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    m: Vec<Tensor>,  // First moment
    v: Vec<Tensor>,  // Second moment
    t: usize,        // Step count
    initialized: bool,
}

impl Adam {
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
            initialized: false,
        }
    }

    /// Create with default hyperparameters (lr=1e-3, betas=(0.9, 0.999), eps=1e-8).
    pub fn default_with_lr(lr: f32) -> Self {
        Self::new(lr, 0.9, 0.999, 1e-8, 0.0)
    }

    /// Perform one optimization step.
    pub fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        if !self.initialized {
            self.m = grads
                .iter()
                .map(|g| Tensor::zeros(g.shape().dims(), g.dtype()))
                .collect();
            self.v = grads
                .iter()
                .map(|g| Tensor::zeros(g.shape().dims(), g.dtype()))
                .collect();
            self.initialized = true;
        }

        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            // Decoupled weight decay
            if self.weight_decay > 0.0 {
                let decay = param.mul_scalar(self.weight_decay * self.lr).expect("Adam wd failed");
                let new_p = param.sub(&decay).expect("Adam wd sub failed");
                if let (Some(dst), Some(src)) = (param.as_f32_slice_mut(), new_p.as_f32_slice()) {
                    dst.copy_from_slice(src);
                }
            }

            // m = beta1 * m + (1 - beta1) * grad
            let m_new = self.m[i]
                .mul_scalar(self.beta1).expect("Adam: m * beta1")
                .add(&grad.mul_scalar(1.0 - self.beta1).expect("Adam: grad scale")).expect("Adam: m update");
            self.m[i] = m_new;

            // v = beta2 * v + (1 - beta2) * grad^2
            let grad_sq = grad.mul(grad).expect("Adam: grad^2");
            let v_new = self.v[i]
                .mul_scalar(self.beta2).expect("Adam: v * beta2")
                .add(&grad_sq.mul_scalar(1.0 - self.beta2).expect("Adam: grad_sq scale")).expect("Adam: v update");
            self.v[i] = v_new;

            // Bias-corrected estimates
            let m_hat = self.m[i].mul_scalar(1.0 / bc1).expect("Adam: m_hat");
            let v_hat = self.v[i].mul_scalar(1.0 / bc2).expect("Adam: v_hat");

            // param -= lr * m_hat / (sqrt(v_hat) + eps)
            let v_sqrt = v_hat.sqrt().expect("Adam: sqrt").add_scalar(self.eps).expect("Adam: eps");
            let update = m_hat.div(&v_sqrt).expect("Adam: div").mul_scalar(self.lr).expect("Adam: lr scale");
            let new_param = param.sub(&update).expect("Adam: param update");

            if let (Some(dst), Some(src)) = (param.as_f32_slice_mut(), new_param.as_f32_slice()) {
                dst.copy_from_slice(src);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_step() {
        let mut params = vec![Tensor::from_f32(&[1.0, 2.0, 3.0], &[3])];
        let grads = vec![Tensor::from_f32(&[0.1, 0.2, 0.3], &[3])];

        let mut opt = Adam::default_with_lr(0.001);
        opt.step(&mut params, &grads);

        // After one step, params should have decreased
        let data = params[0].as_f32_slice().unwrap();
        assert!(data[0] < 1.0);
        assert!(data[1] < 2.0);
        assert!(data[2] < 3.0);
    }

    #[test]
    fn test_adam_multiple_steps() {
        let mut params = vec![Tensor::from_f32(&[5.0], &[1])];
        let grads = vec![Tensor::from_f32(&[1.0], &[1])];

        let mut opt = Adam::default_with_lr(0.1);
        for _ in 0..10 {
            opt.step(&mut params, &grads);
        }

        // Should have moved significantly toward 0
        let val = params[0].get_f32(0).unwrap();
        assert!(val < 5.0);
    }
}
