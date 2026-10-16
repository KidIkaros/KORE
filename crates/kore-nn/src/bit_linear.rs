//! BitLinear: 1.58-bit linear layer using ternary weights {-1, 0, +1}.
//!
//! Uses kore-btes for ternary encoding and kore-kernels for VT-ALU-accelerated
//! ternary matrix multiplication. Weights are stored in base-243 packed format
//! (5 trits per byte) with per-row f32 scales.
//!
//! Memory: ~1.6 bits/param vs 32 bits/param for f32 Linear → **20× compression**.
//!
//! Forward: `y = ternary_matmul(W_packed, scales, x) + bias`
//!
//! For QAT (Quantization-Aware Training), use `BitLinear::from_linear()` which
//! quantizes an existing f32 Linear layer's weights to ternary.

use std::collections::HashMap;

use kore_core::{DType, Tensor};
use kore_kernels::cpu_ternary_matmul::{pack_weights_ternary, ternary_matmul};

use crate::module::Module;

/// A linear layer with ternary-quantized weights (1.58-bit).
///
/// Stores weights in base-243 packed format with per-row scales.
/// Forward pass uses VT-ALU-accelerated ternary matmul from kore-kernels.
pub struct BitLinear {
    /// Packed ternary weights: [out_features, ceil(in_features/5)] bytes
    packed_weights: Vec<u8>,
    /// Per-row scale factors: [out_features]
    scales: Vec<f32>,
    /// Optional device-resident packed weights tensor: [out_features, in_features] (DType::Ternary)
    gpu_weights: Option<Tensor>,
    /// Optional device-resident scale factors tensor: [out_features] (DType::F32)
    gpu_scales: Option<Tensor>,
    /// Optional bias: [out_features]
    bias: Option<Tensor>,
    /// Logical dimensions
    in_features: usize,
    out_features: usize,
    /// Quantization threshold (trits with |normalized_value| < threshold become 0)
    threshold: f32,
    /// Whether weights were quantized in the Fast Walsh-Hadamard rotated basis
    hadamard_rotated: bool,
    training: bool,
}

impl BitLinear {
    /// Create a BitLinear layer by quantizing f32 weights to ternary.
    ///
    /// # Arguments
    /// * `weight` - f32 weight tensor of shape [out_features, in_features]
    /// * `bias` - optional f32 bias tensor of shape [out_features]
    /// * `threshold` - quantization threshold (default: 0.3). Values with
    ///   |normalized_weight| < threshold are mapped to 0.
    pub fn new(weight: &Tensor, bias: Option<&Tensor>, threshold: f32) -> Self {
        Self::build(weight, bias, threshold, false)
    }

    /// Create a BitLinear layer in the Fast Walsh-Hadamard rotated basis (`QuIP#` / `SpinQuant` style).
    ///
    /// Rotates each row of `weight` via normalized FWHT before ternary quantization to spread
    /// weight and activation outliers across the feature dimension.
    pub fn new_with_hadamard(weight: &Tensor, bias: Option<&Tensor>, threshold: f32) -> Self {
        Self::build(weight, bias, threshold, true)
    }

    fn build(
        weight: &Tensor,
        bias: Option<&Tensor>,
        threshold: f32,
        hadamard_rotated: bool,
    ) -> Self {
        let dims = weight.shape().dims();
        assert!(
            dims.len() == 2,
            "BitLinear weight must be 2D, got {:?}",
            dims
        );
        let out_features = dims[0];
        let in_features = dims[1];

        let w_prepared = if hadamard_rotated {
            weight
                .cpu()
                .unwrap_or_else(|_| weight.clone())
                .hadamard()
                .expect("BitLinear Hadamard rotation failed")
        } else {
            weight.cpu().unwrap_or_else(|_| weight.clone()).contiguous()
        };
        let w_slice = w_prepared
            .as_f32_slice()
            .expect("BitLinear requires f32 weights");

        let (packed_weights, scales) =
            pack_weights_ternary(w_slice, out_features, in_features, threshold);

        let bias_tensor = bias.cloned();

        Self {
            packed_weights,
            scales,
            gpu_weights: None,
            gpu_scales: None,
            bias: bias_tensor,
            in_features,
            out_features,
            threshold,
            hadamard_rotated,
            training: false,
        }
    }

    /// Create a BitLinear from an existing f32 Linear layer.
    ///
    /// Quantizes the Linear's weights to ternary format.
    pub fn from_linear(linear: &crate::Linear, threshold: f32) -> Self {
        Self::new(linear.weight(), linear.bias(), threshold)
    }

    /// Create a Hadamard-rotated BitLinear from an existing f32 Linear layer.
    pub fn from_linear_with_hadamard(linear: &crate::Linear, threshold: f32) -> Self {
        Self::new_with_hadamard(linear.weight(), linear.bias(), threshold)
    }

    /// Move packed ternary weights, per-row scales, and optional bias to the target device.
    pub fn to(&self, device: kore_core::Device) -> kore_core::Result<Self> {
        if device.is_cpu() {
            return Ok(Self {
                packed_weights: self.packed_weights.clone(),
                scales: self.scales.clone(),
                gpu_weights: None,
                gpu_scales: None,
                bias: self.bias.as_ref().map(|b| b.to(device)).transpose()?,
                in_features: self.in_features,
                out_features: self.out_features,
                threshold: self.threshold,
                hadamard_rotated: self.hadamard_rotated,
                training: self.training,
            });
        }

        let w_storage = kore_core::Storage::from_packed_bytes(
            DType::Ternary,
            self.out_features * self.in_features,
            self.packed_weights.clone(),
        );
        let gpu_weights =
            Tensor::from_storage(w_storage, &[self.out_features, self.in_features]).to(device)?;
        let gpu_scales = Tensor::from_f32(&self.scales, &[self.out_features]).to(device)?;
        let bias = self.bias.as_ref().map(|b| b.to(device)).transpose()?;

        Ok(Self {
            packed_weights: self.packed_weights.clone(),
            scales: self.scales.clone(),
            gpu_weights: Some(gpu_weights),
            gpu_scales: Some(gpu_scales),
            bias,
            in_features: self.in_features,
            out_features: self.out_features,
            threshold: self.threshold,
            hadamard_rotated: self.hadamard_rotated,
            training: self.training,
        })
    }

    /// Whether weights are stored in the Hadamard-rotated basis.
    pub fn is_hadamard_rotated(&self) -> bool {
        self.hadamard_rotated
    }

    /// Create a BitLinear with random ternary weights (for testing).
    pub fn random(in_features: usize, out_features: usize, bias: bool) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let weight = Tensor::from_f32(&weight_data, &[out_features, in_features]);

        let bias_tensor = if bias {
            Some(Tensor::zeros(&[out_features], DType::F32))
        } else {
            None
        };

        Self::new(&weight, bias_tensor.as_ref(), 0.3)
    }

    /// Input feature dimension.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Output feature dimension.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Quantization threshold.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Packed weight bytes (base-243 encoded).
    pub fn packed_weights(&self) -> &[u8] {
        &self.packed_weights
    }

    /// Per-row scale factors.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Memory usage in bytes for weights (packed ternary + scales).
    pub fn weight_memory_bytes(&self) -> usize {
        self.packed_weights.len() + self.scales.len() * 4
    }

    /// Equivalent f32 memory for comparison.
    pub fn f32_weight_memory_bytes(&self) -> usize {
        self.out_features * self.in_features * 4
    }

    /// Compression ratio vs f32.
    pub fn compression_ratio(&self) -> f32 {
        self.f32_weight_memory_bytes() as f32 / self.weight_memory_bytes() as f32
    }

    /// Dequantize packed weights back to f32 (for inspection/debugging).
    pub fn dequantize_weights(&self) -> Tensor {
        let k_packed = self.in_features.div_ceil(5);
        let mut result = vec![0.0f32; self.out_features * self.in_features];

        for row in 0..self.out_features {
            let scale = self.scales[row];
            let row_packed = &self.packed_weights[row * k_packed..(row + 1) * k_packed];

            let mut col = 0;
            for &byte in row_packed {
                let trits = kore_btes::encoder::decode_trits(byte);
                for &t in &trits {
                    if col >= self.in_features {
                        break;
                    }
                    result[row * self.in_features + col] = (t as i8 as f32) * scale;
                    col += 1;
                }
            }
        }

        let t = Tensor::from_f32(&result, &[self.out_features, self.in_features]);
        if self.hadamard_rotated {
            t.hadamard().unwrap_or(t)
        } else {
            t
        }
    }
}

impl Module for BitLinear {
    /// Forward pass: y = ternary_matmul(W_packed, scales, x) + bias
    ///
    /// Input shape: [batch, in_features] or [in_features]
    /// Output shape: [batch, out_features] or [out_features]
    fn forward(&self, input: &Tensor) -> kore_core::Result<Tensor> {
        let input_dims = input.shape().dims();
        let ndim = input_dims.len();

        if ndim == 0 || input_dims[ndim - 1] != self.in_features {
            return Err(kore_core::KoreError::ShapeMismatch {
                expected: vec![self.in_features],
                got: input_dims.to_vec(),
            });
        }

        // Handle batched input: reshape to [M, K] where M = product of batch dims
        let batch_size: usize = if ndim > 1 {
            input_dims[..ndim - 1].iter().product()
        } else {
            1
        };

        let mut input_2d = if ndim == 1 {
            input.reshape(&[1, self.in_features as isize])?
        } else if ndim > 2 {
            input.reshape(&[batch_size as isize, self.in_features as isize])?
        } else {
            input.clone()
        };

        if self.hadamard_rotated {
            input_2d = input_2d.hadamard()?;
        }

        // GPU fast path via DeviceBackend::quantized_matmul
        let target_dev = if !input_2d.device().is_cpu() {
            Some(input_2d.device())
        } else {
            self.gpu_weights.as_ref().map(|w| w.device())
        };

        let mut output = if let Some(dev) = target_dev {
            if let Some(backend) = kore_core::device::get_backend(dev) {
                let x_dev = if input_2d.device() != dev {
                    input_2d.to(dev)?
                } else {
                    input_2d.clone()
                };
                let w_dev = match &self.gpu_weights {
                    Some(w) if w.device() == dev => w.clone(),
                    _ => {
                        let w_storage = kore_core::Storage::from_packed_bytes(
                            DType::Ternary,
                            self.out_features * self.in_features,
                            self.packed_weights.clone(),
                        );
                        Tensor::from_storage(w_storage, &[self.out_features, self.in_features])
                            .to(dev)?
                    }
                };
                let s_dev = match &self.gpu_scales {
                    Some(s) if s.device() == dev => s.clone(),
                    _ => Tensor::from_f32(&self.scales, &[self.out_features]).to(dev)?,
                };
                backend.quantized_matmul(&x_dev, &w_dev, &s_dev, 2)?
            } else {
                let input_t = input_2d.cpu()?.transpose()?.contiguous();
                ternary_matmul(
                    &self.packed_weights,
                    &self.scales,
                    &input_t,
                    self.out_features,
                    batch_size,
                    self.in_features,
                )?
                .transpose()?
                .contiguous()
                .to(dev)?
            }
        } else {
            let input_t = input_2d.transpose()?.contiguous();
            let out_t = ternary_matmul(
                &self.packed_weights,
                &self.scales,
                &input_t,
                self.out_features,
                batch_size,
                self.in_features,
            )?;
            out_t.transpose()?.contiguous()
        };

        // Add bias
        if let Some(ref bias) = self.bias {
            let b_dev = if bias.device() != output.device() {
                bias.to(output.device())?
            } else {
                bias.clone()
            };
            output = output.add(&b_dev)?;
        }

        // Reshape back to original batch dims
        if ndim == 1 {
            output = output.reshape(&[self.out_features as isize])?;
        } else if ndim > 2 {
            let mut out_shape: Vec<isize> =
                input_dims[..ndim - 1].iter().map(|&d| d as isize).collect();
            out_shape.push(self.out_features as isize);
            output = output.reshape(&out_shape)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        // BitLinear weights are packed bytes, not Tensors.
        // Only bias is a trainable parameter (for fine-tuning scenarios).
        let mut params = Vec::new();
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(&str, &Tensor)> {
        let mut params = Vec::new();
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

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut sd = HashMap::new();
        // Store dequantized weights for compatibility
        sd.insert("weight".to_string(), self.dequantize_weights());
        if let Some(ref b) = self.bias {
            sd.insert("bias".to_string(), b.clone());
        }
        sd
    }
}

impl std::fmt::Display for BitLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BitLinear(in={}, out={}, threshold={}, compression={:.1}x, mem={:.1}KB)",
            self.in_features,
            self.out_features,
            self.threshold,
            self.compression_ratio(),
            self.weight_memory_bytes() as f32 / 1024.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Linear;

    #[test]
    fn test_bit_linear_creation() {
        let bl = BitLinear::random(64, 32, true);
        assert_eq!(bl.in_features(), 64);
        assert_eq!(bl.out_features(), 32);
        assert!(bl.compression_ratio() > 5.0); // should be ~12-20x
    }

    #[test]
    fn test_bit_linear_forward_shape() {
        let bl = BitLinear::random(16, 8, false);

        // 1D input
        let x1 = Tensor::ones(&[16]);
        let y1 = bl.forward(&x1).unwrap();
        assert_eq!(y1.shape().dims(), &[8]);

        // 2D input
        let x2 = Tensor::ones(&[4, 16]);
        let y2 = bl.forward(&x2).unwrap();
        assert_eq!(y2.shape().dims(), &[4, 8]);
    }

    #[test]
    fn test_bit_linear_from_linear() {
        let linear = Linear::new(8, 4, true);
        let bl = BitLinear::from_linear(&linear, 0.3);

        assert_eq!(bl.in_features(), 8);
        assert_eq!(bl.out_features(), 4);

        // Forward should produce finite output
        let x = Tensor::ones(&[2, 8]);
        let y = bl.forward(&x).unwrap();
        assert_eq!(y.shape().dims(), &[2, 4]);
        let data = y.as_f32_slice().unwrap();
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_bit_linear_approximates_linear() {
        // Create a linear with known weights that are ternary-friendly
        let w_data = vec![
            1.0, -1.0, 0.0, 1.0, // row 0
            -1.0, 1.0, 1.0, 0.0, // row 1
        ];
        let weight = Tensor::from_f32(&w_data, &[2, 4]);
        let bl = BitLinear::new(&weight, None, 0.3);

        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let y = bl.forward(&x).unwrap();
        let y_data = y.as_f32_slice().unwrap();

        // Expected: row0 = 1*1 + (-1)*2 + 0*3 + 1*4 = 3.0
        //           row1 = (-1)*1 + 1*2 + 1*3 + 0*4 = 4.0
        // With ternary quantization and scaling, should be close
        assert!((y_data[0] - 3.0).abs() < 0.5, "got {}", y_data[0]);
        assert!((y_data[1] - 4.0).abs() < 0.5, "got {}", y_data[1]);
    }

    #[test]
    fn test_bit_linear_compression() {
        let bl = BitLinear::random(4096, 4096, false);
        let ratio = bl.compression_ratio();
        // base-243: 5 trits/byte = 1.6 bits/trit, plus 4 bytes scale per row
        // f32: 32 bits/param. Ratio should be ~12-20x
        assert!(ratio > 10.0, "compression ratio {} too low", ratio);
    }

    #[test]
    fn test_bit_linear_dequantize() {
        let w_data = vec![1.0, -1.0, 0.0, 1.0, -1.0, 1.0, 1.0, 0.0];
        let weight = Tensor::from_f32(&w_data, &[2, 4]);
        let bl = BitLinear::new(&weight, None, 0.3);

        let deq = bl.dequantize_weights();
        assert_eq!(deq.shape().dims(), &[2, 4]);
        let deq_data = deq.as_f32_slice().unwrap();
        // Signs should match original
        assert!(deq_data[0] > 0.0); // was 1.0
        assert!(deq_data[1] < 0.0); // was -1.0
    }

    #[test]
    fn test_bit_linear_state_dict() {
        let bl = BitLinear::random(8, 4, true);
        let sd = bl.state_dict();
        assert!(sd.contains_key("weight"));
        assert!(sd.contains_key("bias"));
        assert_eq!(sd["weight"].shape().dims(), &[4, 8]);
    }

    #[test]
    fn test_bit_linear_display() {
        let bl = BitLinear::random(1024, 512, false);
        let s = format!("{}", bl);
        assert!(s.contains("BitLinear"));
        assert!(s.contains("1024"));
        assert!(s.contains("512"));
    }

    #[test]
    fn test_bit_linear_bias() {
        let w = Tensor::from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let b = Tensor::from_f32(&[10.0, 20.0], &[2]);
        let bl = BitLinear::new(&w, Some(&b), 0.3);

        let x = Tensor::from_f32(&[1.0, 1.0], &[1, 2]);
        let y = bl.forward(&x).unwrap();
        let y_data = y.as_f32_slice().unwrap();
        // Output should include bias offset
        assert!(y_data[0] > 5.0, "bias not applied: got {}", y_data[0]);
        assert!(y_data[1] > 15.0, "bias not applied: got {}", y_data[1]);
    }

    #[test]
    fn test_bit_linear_hadamard_rotation() {
        let k = 64;
        let m = 8;
        let w_data: Vec<f32> = (0..(m * k))
            .map(|i| {
                ((i as f32 * 1.37 + (i * i) as f32 * 0.19).sin() * 2.5)
                    + if i % k == 3 { 12.0 } else { 0.0 }
            })
            .collect();
        let x_data: Vec<f32> = (0..(2 * k)).map(|i| (i as f32 * 0.31).cos()).collect();
        let w = Tensor::from_f32(&w_data, &[m, k]);
        let x = Tensor::from_f32(&x_data, &[2, k]);

        // Reference FP32: x @ w^T
        let wt = w.transpose().unwrap().contiguous();
        let ref_out = x.matmul(&wt).unwrap();
        let ref_slice = ref_out.as_f32_slice().unwrap();

        let bl_unrot = BitLinear::new(&w, None, 0.3);
        let bl_rot = BitLinear::new_with_hadamard(&w, None, 0.3);
        assert!(bl_rot.is_hadamard_rotated());

        let y_unrot = bl_unrot.forward(&x).unwrap();
        let y_rot = bl_rot.forward(&x).unwrap();

        let mse_unrot: f32 = ref_slice
            .iter()
            .zip(y_unrot.as_f32_slice().unwrap())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let mse_rot: f32 = ref_slice
            .iter()
            .zip(y_rot.as_f32_slice().unwrap())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();

        // With a 12.0 channel outlier, unrotated ternary quantization zeroes out all non-outlier weights!
        assert!(
            mse_rot < mse_unrot * 0.5,
            "Hadamard rotation should improve BitLinear accuracy on outlier weights: rot={mse_rot} vs unrot={mse_unrot}"
        );
    }
}
