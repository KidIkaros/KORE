//! LoRA (Low-Rank Adaptation) and QLoRA layers.
//!
//! LoRA freezes the base weight and adds a low-rank decomposition:
//!   y = (W + (alpha/rank) * B @ A) @ x
//! where A: [rank, in_features], B: [out_features, rank].
//!
//! Only A and B are trainable — massive parameter savings for fine-tuning.
//!
//! QLoRA combines LoRA with BitLinear: the base weight is ternary-quantized,
//! and the low-rank adapters remain in f32.

use std::collections::HashMap;

use kore_core::{DType, KoreError, Tensor};

use crate::module::Module;
use crate::linear::Linear;
use crate::bit_linear::BitLinear;

/// LoRA adapter on top of a standard f32 Linear layer.
///
/// Forward: y = W @ x + (alpha/rank) * B @ A @ x + bias
#[derive(Clone)]
pub struct LoraLinear {
    /// Frozen base weight: [out_features, in_features]
    base_weight: Tensor,
    /// Low-rank down projection: [rank, in_features]
    lora_a: Tensor,
    /// Low-rank up projection: [out_features, rank]
    lora_b: Tensor,
    /// Optional bias
    bias: Option<Tensor>,
    /// LoRA scaling factor
    alpha: f32,
    rank: usize,
    in_features: usize,
    out_features: usize,
    training: bool,
}

impl LoraLinear {
    /// Wrap an existing Linear layer with LoRA adapters.
    ///
    /// `rank`: rank of the low-rank decomposition (typically 4-64)
    /// `alpha`: scaling factor (typically equal to rank)
    pub fn from_linear(linear: &Linear, rank: usize, alpha: f32) -> Self {
        let in_features = linear.in_features();
        let out_features = linear.out_features();

        // A is initialized with small random values (Kaiming-like)
        let a_scale = (1.0 / in_features as f32).sqrt();
        let a_data: Vec<f32> = (0..rank * in_features)
            .map(|i| {
                let x = ((i * 2654435761 + 1013904223) & 0xFFFFFF) as f32 / 0xFFFFFF as f32;
                (x * 2.0 - 1.0) * a_scale
            })
            .collect();
        let lora_a = Tensor::from_f32(&a_data, &[rank, in_features]);

        // B is initialized to zero so LoRA starts as identity
        let lora_b = Tensor::zeros(&[out_features, rank], DType::F32);

        Self {
            base_weight: linear.weight().clone(),
            lora_a,
            lora_b,
            bias: linear.bias().cloned(),
            alpha,
            rank,
            in_features,
            out_features,
            training: true,
        }
    }

    /// Create a standalone LoRA linear (no pre-existing base weight).
    pub fn new(in_features: usize, out_features: usize, rank: usize, alpha: f32, bias: bool) -> Self {
        let linear = Linear::new(in_features, out_features, bias);
        Self::from_linear(&linear, rank, alpha)
    }

    pub fn rank(&self) -> usize { self.rank }
    pub fn alpha(&self) -> f32 { self.alpha }
    pub fn in_features(&self) -> usize { self.in_features }
    pub fn out_features(&self) -> usize { self.out_features }

    /// Number of trainable parameters (only A and B).
    pub fn trainable_params(&self) -> usize {
        self.rank * self.in_features + self.out_features * self.rank
            + self.bias.as_ref().map_or(0, |b| b.numel())
    }

    /// Total parameters (base + LoRA).
    pub fn total_params(&self) -> usize {
        self.in_features * self.out_features + self.trainable_params()
    }

    /// Merge LoRA weights into the base weight (for inference).
    /// Returns a new Linear with the merged weight.
    pub fn merge(&self) -> Result<Linear, KoreError> {
        let scaling = self.alpha / self.rank as f32;

        // delta_W = scaling * B @ A → [out_features, in_features]
        let delta = self.lora_b.matmul(&self.lora_a)?
            .mul_scalar(scaling)?;

        let merged_weight = self.base_weight.add(&delta)?;

        Ok(Linear::from_weight(merged_weight, self.bias.clone()))
    }
}

impl Module for LoraLinear {
    fn clone_box(&self) -> Box<dyn Module> { Box::new(self.clone()) }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.lora_a, &mut self.lora_b];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    fn forward(&self, input: &Tensor) -> kore_core::Result<Tensor> {
        let scaling = self.alpha / self.rank as f32;

        // Base: input @ W^T
        let wt = self.base_weight.transpose()?;
        let mut output = input.matmul(&wt.contiguous())?;

        // LoRA: input @ A^T @ B^T * scaling
        let at = self.lora_a.transpose()?;
        let bt = self.lora_b.transpose()?;
        let lora_out = input.matmul(&at.contiguous())?
            .matmul(&bt.contiguous())?
            .mul_scalar(scaling)?;

        output = output.add(&lora_out)?;

        if let Some(ref bias) = self.bias {
            output = output.add(bias)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.lora_a, &self.lora_b];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        let mut params = vec![
            ("lora_a".into(), &self.lora_a),
            ("lora_b".into(), &self.lora_b),
        ];
        if let Some(ref b) = self.bias {
            params.push(("bias".into(), b));
        }
        params
    }

    fn set_parameters(&mut self, params: &[Tensor]) -> usize {
        let mut n = 0;
        self.lora_a = params[n].clone(); n += 1;
        self.lora_b = params[n].clone(); n += 1;
        if self.bias.is_some() { self.bias = Some(params[n].clone()); n += 1; }
        n
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();
        sd.insert("base_weight".to_string(), self.base_weight.clone());
        sd.insert("lora_a".to_string(), self.lora_a.clone());
        sd.insert("lora_b".to_string(), self.lora_b.clone());
        if let Some(ref b) = self.bias {
            sd.insert("bias".to_string(), b.clone());
        }
        sd
    }
}

/// QLoRA: LoRA on top of a ternary-quantized (BitLinear) base.
///
/// Base weight is stored in 1.58-bit ternary format (frozen).
/// LoRA adapters A and B remain in f32 (trainable).
///
/// Forward: y = BitLinear(x) + (alpha/rank) * B @ A @ x
#[derive(Clone)]
pub struct QLoraLinear {
    /// Frozen ternary base
    base: BitLinear,
    /// Low-rank down projection: [rank, in_features]
    lora_a: Tensor,
    /// Low-rank up projection: [out_features, rank]
    lora_b: Tensor,
    /// Optional bias (f32, trainable)
    bias: Option<Tensor>,
    alpha: f32,
    rank: usize,
    in_features: usize,
    out_features: usize,
    training: bool,
}

impl QLoraLinear {
    /// Create QLoRA from an existing Linear layer.
    ///
    /// Quantizes the base weight to ternary, then adds LoRA adapters.
    pub fn from_linear(linear: &Linear, rank: usize, alpha: f32, threshold: f32) -> Self {
        let in_features = linear.in_features();
        let out_features = linear.out_features();

        let base = BitLinear::from_linear(linear, threshold);

        let a_scale = (1.0 / in_features as f32).sqrt();
        let a_data: Vec<f32> = (0..rank * in_features)
            .map(|i| {
                let x = ((i * 2654435761 + 1013904223) & 0xFFFFFF) as f32 / 0xFFFFFF as f32;
                (x * 2.0 - 1.0) * a_scale
            })
            .collect();
        let lora_a = Tensor::from_f32(&a_data, &[rank, in_features]);
        let lora_b = Tensor::zeros(&[out_features, rank], DType::F32);

        Self {
            base,
            lora_a,
            lora_b,
            bias: linear.bias().cloned(),
            alpha,
            rank,
            in_features,
            out_features,
            training: true,
        }
    }

    /// Create QLoRA from an existing BitLinear layer.
    pub fn from_bit_linear(base: BitLinear, rank: usize, alpha: f32) -> Self {
        let in_features = base.in_features();
        let out_features = base.out_features();

        let a_scale = (1.0 / in_features as f32).sqrt();
        let a_data: Vec<f32> = (0..rank * in_features)
            .map(|i| {
                let x = ((i * 2654435761 + 1013904223) & 0xFFFFFF) as f32 / 0xFFFFFF as f32;
                (x * 2.0 - 1.0) * a_scale
            })
            .collect();
        let lora_a = Tensor::from_f32(&a_data, &[rank, in_features]);
        let lora_b = Tensor::zeros(&[out_features, rank], DType::F32);

        Self {
            base,
            lora_a,
            lora_b,
            bias: None,
            alpha,
            rank,
            in_features,
            out_features,
            training: true,
        }
    }

    pub fn rank(&self) -> usize { self.rank }
    pub fn alpha(&self) -> f32 { self.alpha }
    pub fn in_features(&self) -> usize { self.in_features }
    pub fn out_features(&self) -> usize { self.out_features }

    /// Trainable parameter count (only LoRA A, B, and bias).
    pub fn trainable_params(&self) -> usize {
        self.rank * self.in_features + self.out_features * self.rank
            + self.bias.as_ref().map_or(0, |b| b.numel())
    }

    /// Compression ratio of the base weight.
    pub fn base_compression_ratio(&self) -> f32 {
        self.base.compression_ratio()
    }
}

impl Module for QLoraLinear {
    fn clone_box(&self) -> Box<dyn Module> { Box::new(self.clone()) }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.lora_a, &mut self.lora_b];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    fn forward(&self, input: &Tensor) -> kore_core::Result<Tensor> {
        let scaling = self.alpha / self.rank as f32;

        // Base: ternary matmul
        let mut output = self.base.forward(input)?;

        // LoRA: input @ A^T @ B^T * scaling
        let at = self.lora_a.transpose()?;
        let bt = self.lora_b.transpose()?;
        let lora_out = input.matmul(&at.contiguous())?
            .matmul(&bt.contiguous())?
            .mul_scalar(scaling)?;

        output = output.add(&lora_out)?;

        if let Some(ref bias) = self.bias {
            output = output.add(bias)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.lora_a, &self.lora_b];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        let mut params = vec![
            ("lora_a".into(), &self.lora_a),
            ("lora_b".into(), &self.lora_b),
        ];
        if let Some(ref b) = self.bias {
            params.push(("bias".into(), b));
        }
        params
    }

    fn set_parameters(&mut self, params: &[Tensor]) -> usize {
        let mut n = 0;
        self.lora_a = params[n].clone(); n += 1;
        self.lora_b = params[n].clone(); n += 1;
        if self.bias.is_some() { self.bias = Some(params[n].clone()); n += 1; }
        n
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

impl std::fmt::Display for LoraLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LoraLinear(in={}, out={}, rank={}, alpha={}, trainable={}/{})",
            self.in_features, self.out_features, self.rank, self.alpha,
            self.trainable_params(), self.total_params(),
        )
    }
}

impl std::fmt::Display for QLoraLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "QLoraLinear(in={}, out={}, rank={}, alpha={}, base_compression={:.1}x, trainable={})",
            self.in_features, self.out_features, self.rank, self.alpha,
            self.base_compression_ratio(), self.trainable_params(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_creation() {
        let linear = Linear::new(64, 32, true);
        let lora = LoraLinear::from_linear(&linear, 8, 8.0);
        assert_eq!(lora.rank(), 8);
        assert_eq!(lora.in_features(), 64);
        assert_eq!(lora.out_features(), 32);
        // Trainable: A(8*64) + B(32*8) + bias(32) = 512 + 256 + 32 = 800
        assert_eq!(lora.trainable_params(), 800);
    }

    #[test]
    fn test_lora_forward_shape() {
        let linear = Linear::new(16, 8, false);
        let lora = LoraLinear::from_linear(&linear, 4, 4.0);

        let x = Tensor::ones(&[3, 16]);
        let y = lora.forward(&x).unwrap();
        assert_eq!(y.shape().dims(), &[3, 8]);
    }

    #[test]
    fn test_lora_starts_as_base() {
        // With B initialized to zero, LoRA output should match base Linear
        let linear = Linear::new(8, 4, false);
        let lora = LoraLinear::from_linear(&linear, 4, 4.0);

        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 8]);
        let y_linear = crate::Module::forward(&linear, &x).unwrap();
        let y_lora = lora.forward(&x).unwrap();

        let l_data = y_linear.as_f32_slice().unwrap();
        let r_data = y_lora.as_f32_slice().unwrap();
        for i in 0..4 {
            assert!(
                (l_data[i] - r_data[i]).abs() < 1e-4,
                "mismatch at {}: linear={}, lora={}", i, l_data[i], r_data[i]
            );
        }
    }

    #[test]
    fn test_lora_merge() {
        let linear = Linear::new(8, 4, true);
        let lora = LoraLinear::from_linear(&linear, 4, 4.0);
        let merged = lora.merge().unwrap();

        let x = Tensor::from_f32(&[1.0; 8], &[1, 8]);
        let y_lora = lora.forward(&x).unwrap();
        let y_merged = crate::Module::forward(&merged, &x).unwrap();

        let l_data = y_lora.as_f32_slice().unwrap();
        let m_data = y_merged.as_f32_slice().unwrap();
        for i in 0..4 {
            assert!(
                (l_data[i] - m_data[i]).abs() < 1e-3,
                "merge mismatch at {}: lora={}, merged={}", i, l_data[i], m_data[i]
            );
        }
    }

    #[test]
    fn test_lora_parameters_only_adapters() {
        let linear = Linear::new(16, 8, true);
        let lora = LoraLinear::from_linear(&linear, 4, 4.0);
        let params = lora.parameters();
        // Should only return lora_a, lora_b, bias (not base_weight)
        assert_eq!(params.len(), 3);
    }

    #[test]
    fn test_qlora_creation() {
        let linear = Linear::new(64, 32, false);
        let qlora = QLoraLinear::from_linear(&linear, 8, 8.0, 0.3);
        assert_eq!(qlora.rank(), 8);
        assert!(qlora.base_compression_ratio() > 5.0);
    }

    #[test]
    fn test_qlora_forward_shape() {
        let linear = Linear::new(16, 8, false);
        let qlora = QLoraLinear::from_linear(&linear, 4, 4.0, 0.3);

        let x = Tensor::ones(&[3, 16]);
        let y = qlora.forward(&x).unwrap();
        assert_eq!(y.shape().dims(), &[3, 8]);
        let data = y.as_f32_slice().unwrap();
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_qlora_display() {
        let linear = Linear::new(1024, 512, false);
        let qlora = QLoraLinear::from_linear(&linear, 16, 16.0, 0.3);
        let s = format!("{}", qlora);
        assert!(s.contains("QLoraLinear"));
        assert!(s.contains("rank=16"));
    }

    #[test]
    fn test_lora_display() {
        let lora = LoraLinear::new(256, 128, 8, 8.0, false);
        let s = format!("{}", lora);
        assert!(s.contains("LoraLinear"));
        assert!(s.contains("rank=8"));
    }
}
