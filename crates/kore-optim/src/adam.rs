use kore_core::Tensor;

/// Per-group hyperparameters for Adam.
///
/// Allows different learning rates, weight decay, and freeze flags
/// for different subsets of parameters (e.g. ViT encoder vs Mamba predictor).
#[derive(Clone, Debug)]
pub struct ParamGroup {
    /// Learning rate override for this group (None = use optimizer default).
    pub lr: Option<f32>,
    /// Weight decay override (None = use optimizer default).
    pub weight_decay: Option<f32>,
    /// If true, parameters in this group are frozen (no updates).
    pub frozen: bool,
    /// Indices into the flat parameter list that belong to this group.
    pub param_indices: Vec<usize>,
}

impl ParamGroup {
    /// Create a group with default overrides (inherits optimizer settings).
    pub fn new(param_indices: Vec<usize>) -> Self {
        Self { lr: None, weight_decay: None, frozen: false, param_indices }
    }

    /// Builder: set learning rate.
    pub fn with_lr(mut self, lr: f32) -> Self { self.lr = Some(lr); self }

    /// Builder: set weight decay.
    pub fn with_weight_decay(mut self, wd: f32) -> Self { self.weight_decay = Some(wd); self }

    /// Builder: freeze this group.
    pub fn frozen(mut self) -> Self { self.frozen = true; self }
}

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

    /// Learning rate.
    pub fn lr(&self) -> f32 { self.lr }

    /// First moment decay rate.
    pub fn beta1(&self) -> f32 { self.beta1 }

    /// Second moment decay rate.
    pub fn beta2(&self) -> f32 { self.beta2 }

    /// Epsilon for numerical stability.
    pub fn eps(&self) -> f32 { self.eps }

    /// Weight decay coefficient.
    pub fn weight_decay(&self) -> f32 { self.weight_decay }

    /// Create with default hyperparameters (lr=1e-3, betas=(0.9, 0.999), eps=1e-8).
    pub fn default_with_lr(lr: f32) -> Self {
        Self::new(lr, 0.9, 0.999, 1e-8, 0.0)
    }

    /// Set the global learning rate.
    pub fn set_lr(&mut self, lr: f32) { self.lr = lr; }

    /// Current step count.
    pub fn step_count(&self) -> usize { self.t }

    /// Perform one optimization step (all params use global hyperparameters).
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

    /// Perform one step with parameter groups.
    ///
    /// Each `ParamGroup` specifies which parameter indices it covers and
    /// optional per-group overrides for `lr` and `weight_decay`.
    /// Parameters not in any group are skipped.
    pub fn step_groups(
        &mut self,
        params: &mut [Tensor],
        grads: &[Tensor],
        groups: &[ParamGroup],
    ) {
        assert_eq!(params.len(), grads.len(), "params and grads length mismatch");

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

        for group in groups {
            if group.frozen { continue; }
            let lr = group.lr.unwrap_or(self.lr);
            let wd = group.weight_decay.unwrap_or(self.weight_decay);

            for &i in &group.param_indices {
                if i >= params.len() { continue; }
                let param = &mut params[i];
                let grad = &grads[i];

                // Decoupled weight decay
                if wd > 0.0 {
                    let decay = param.mul_scalar(wd * lr).expect("Adam wd failed");
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
                let update = m_hat.div(&v_sqrt).expect("Adam: div").mul_scalar(lr).expect("Adam: lr scale");
                let new_param = param.sub(&update).expect("Adam: param update");

                if let (Some(dst), Some(src)) = (param.as_f32_slice_mut(), new_param.as_f32_slice()) {
                    dst.copy_from_slice(src);
                }
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

    #[test]
    fn test_param_groups_different_lr() {
        // Two params: group0 with lr=0.1, group1 with lr=0.001
        let mut params = vec![
            Tensor::from_f32(&[5.0], &[1]),
            Tensor::from_f32(&[5.0], &[1]),
        ];
        let grads = vec![
            Tensor::from_f32(&[1.0], &[1]),
            Tensor::from_f32(&[1.0], &[1]),
        ];

        let groups = vec![
            ParamGroup::new(vec![0]).with_lr(0.1),
            ParamGroup::new(vec![1]).with_lr(0.001),
        ];

        let mut opt = Adam::default_with_lr(0.01);
        for _ in 0..10 {
            opt.step_groups(&mut params, &grads, &groups);
        }

        let v0 = params[0].get_f32(0).unwrap();
        let v1 = params[1].get_f32(0).unwrap();
        // Group 0 (lr=0.1) should have moved more than group 1 (lr=0.001)
        assert!(v0 < v1, "param0 ({v0}) should be smaller than param1 ({v1}) due to higher lr");
    }

    #[test]
    fn test_param_groups_frozen() {
        let mut params = vec![
            Tensor::from_f32(&[5.0], &[1]),
            Tensor::from_f32(&[5.0], &[1]),
        ];
        let grads = vec![
            Tensor::from_f32(&[1.0], &[1]),
            Tensor::from_f32(&[1.0], &[1]),
        ];

        let groups = vec![
            ParamGroup::new(vec![0]).with_lr(0.1),
            ParamGroup::new(vec![1]).frozen(),
        ];

        let mut opt = Adam::default_with_lr(0.01);
        opt.step_groups(&mut params, &grads, &groups);

        let v0 = params[0].get_f32(0).unwrap();
        let v1 = params[1].get_f32(0).unwrap();
        assert!(v0 < 5.0, "param0 should have been updated");
        assert!((v1 - 5.0).abs() < 1e-7, "param1 should be frozen at 5.0, got {v1}");
    }

    #[test]
    fn test_param_groups_weight_decay() {
        let mut params = vec![Tensor::from_f32(&[5.0], &[1])];
        let grads = vec![Tensor::from_f32(&[0.0], &[1])]; // zero grad

        let groups = vec![
            ParamGroup::new(vec![0]).with_lr(0.1).with_weight_decay(0.1),
        ];

        let mut opt = Adam::default_with_lr(0.01);
        opt.step_groups(&mut params, &grads, &groups);

        let v = params[0].get_f32(0).unwrap();
        // With zero grad but weight decay, param should shrink
        assert!(v < 5.0, "weight decay should shrink param, got {v}");
    }
}
