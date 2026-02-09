//! QuatLinear: 2-bit linear layer using quaternary weights {-3, -1, +1, +3}.
//!
//! Uses kore-btes for quaternary encoding and kore-kernels for AVX2-accelerated
//! quaternary matrix multiplication. Weights are stored in 2-bit packed format
//! (4 values per byte) with per-row f32 scales.
//!
//! Memory: 2 bits/param vs 32 bits/param for f32 Linear → **16× compression**.
//!
//! Forward: `y = quat_matmul(W_packed, scales, x) + bias`
//!
//! Compared to `BitLinear` (1.58-bit ternary):
//! - Higher fidelity: 4 quantization levels vs 3
//! - Slightly more memory: 2 bits vs 1.58 bits per param
//! - Better accuracy for models sensitive to quantization error

use std::collections::HashMap;

use kore_core::{DType, Tensor};
use kore_kernels::cpu_quat_matmul::{quat_matmul, pack_weights_quaternary};

use crate::module::Module;

/// A linear layer with quaternary-quantized weights (2-bit).
///
/// Stores weights in 2-bit packed format with per-row scales.
/// Forward pass uses AVX2-accelerated quaternary matmul from kore-kernels.
pub struct QuatLinear {
    /// Packed quaternary weights: [out_features, ceil(in_features/4)] bytes
    packed_weights: Vec<u8>,
    /// Per-row scale factors: [out_features]
    scales: Vec<f32>,
    /// Optional bias: [out_features]
    bias: Option<Tensor>,
    /// Logical dimensions
    in_features: usize,
    out_features: usize,
    training: bool,
}

impl QuatLinear {
    /// Create a QuatLinear layer by quantizing f32 weights to quaternary.
    ///
    /// # Arguments
    /// * `weight` - f32 weight tensor of shape [out_features, in_features]
    /// * `bias` - optional f32 bias tensor of shape [out_features]
    pub fn new(weight: &Tensor, bias: Option<&Tensor>) -> Self {
        let dims = weight.shape().dims();
        assert!(dims.len() == 2, "QuatLinear weight must be 2D, got {:?}", dims);
        let out_features = dims[0];
        let in_features = dims[1];

        let w_data = weight.contiguous();
        let w_slice = w_data.as_f32_slice().expect("QuatLinear requires f32 weights");

        let (packed_weights, scales) = pack_weights_quaternary(w_slice, out_features, in_features);

        let bias_tensor = bias.cloned();

        Self {
            packed_weights,
            scales,
            bias: bias_tensor,
            in_features,
            out_features,
            training: false,
        }
    }

    /// Create a QuatLinear from an existing f32 Linear layer.
    pub fn from_linear(linear: &crate::Linear) -> Self {
        Self::new(linear.weight(), linear.bias())
    }

    /// Create a QuatLinear with random weights (for testing).
    pub fn random(in_features: usize, out_features: usize, bias: bool) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| rng.gen_range(-3.0..3.0))
            .collect();
        let weight = Tensor::from_f32(&weight_data, &[out_features, in_features]);

        let bias_tensor = if bias {
            Some(Tensor::zeros(&[out_features], DType::F32))
        } else {
            None
        };

        Self::new(&weight, bias_tensor.as_ref())
    }

    /// Input feature dimension.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Output feature dimension.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Packed weight bytes (2-bit encoded).
    pub fn packed_weights(&self) -> &[u8] {
        &self.packed_weights
    }

    /// Per-row scale factors.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Memory usage in bytes for weights (packed quaternary + scales).
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
        let k_packed = (self.in_features + 3) / 4;
        let mut result = vec![0.0f32; self.out_features * self.in_features];
        const QUAT_VALUES: [f32; 4] = [-3.0, -1.0, 1.0, 3.0];

        for row in 0..self.out_features {
            let scale = self.scales[row];
            let row_packed = &self.packed_weights[row * k_packed..(row + 1) * k_packed];

            let mut col = 0;
            for &byte in row_packed {
                for i in 0..4 {
                    if col >= self.in_features {
                        break;
                    }
                    let idx = ((byte >> (2 * i)) & 0x3) as usize;
                    result[row * self.in_features + col] = QUAT_VALUES[idx] * scale;
                    col += 1;
                }
            }
        }

        Tensor::from_f32(&result, &[self.out_features, self.in_features])
    }
}

impl std::fmt::Display for QuatLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "QuatLinear(in={}, out={}, bias={}, compression={:.1}x)",
            self.in_features,
            self.out_features,
            self.bias.is_some(),
            self.compression_ratio(),
        )
    }
}

impl Module for QuatLinear {
    /// Forward pass: y = quat_matmul(W_packed, scales, x) + bias
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

        let batch_size: usize = if ndim > 1 {
            input_dims[..ndim - 1].iter().product()
        } else {
            1
        };

        let input_2d = if ndim == 1 {
            input.reshape(&[1, self.in_features as isize])?
        } else if ndim > 2 {
            input.reshape(&[batch_size as isize, self.in_features as isize])?
        } else {
            input.clone()
        };

        // quat_matmul expects: a_packed[M,K] @ b[K,N] → [M,N]
        // We need: output[batch, out] = input[batch, in] @ W^T[in, out]
        // Which is: output^T[out, batch] = W[out, in] @ input^T[in, batch]
        let input_t = input_2d.transpose()?.contiguous();

        let mut output = quat_matmul(
            &self.packed_weights,
            &self.scales,
            &input_t,
            self.out_features,
            batch_size,
            self.in_features,
        )?;

        // output is [out_features, batch_size], transpose to [batch_size, out_features]
        output = output.transpose()?.contiguous();

        // Add bias
        if let Some(ref bias) = self.bias {
            output = output.add(bias)?;
        }

        // Reshape back to original batch dims
        if ndim == 1 {
            output = output.reshape(&[self.out_features as isize])?;
        } else if ndim > 2 {
            let mut out_shape: Vec<isize> = input_dims[..ndim - 1].iter().map(|&d| d as isize).collect();
            out_shape.push(self.out_features as isize);
            output = output.reshape(&out_shape)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
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
        sd.insert("weight".to_string(), self.dequantize_weights());
        if let Some(ref b) = self.bias {
            sd.insert("bias".to_string(), b.clone());
        }
        sd
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quat_linear_shape() {
        let ql = QuatLinear::random(64, 32, false);
        let input = Tensor::from_f32(&vec![0.1; 2 * 64], &[2, 64]);
        let output = ql.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 32]);
    }

    #[test]
    fn test_quat_linear_1d_input() {
        let ql = QuatLinear::random(16, 8, false);
        let input = Tensor::from_f32(&vec![1.0; 16], &[16]);
        let output = ql.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[8]);
    }

    #[test]
    fn test_quat_linear_with_bias() {
        let ql = QuatLinear::random(16, 8, true);
        assert_eq!(ql.parameters().len(), 1); // bias only
        assert_eq!(ql.named_parameters().len(), 1);
        assert_eq!(ql.named_parameters()[0].0, "bias");
    }

    #[test]
    fn test_quat_linear_no_bias() {
        let ql = QuatLinear::random(16, 8, false);
        assert_eq!(ql.parameters().len(), 0);
    }

    #[test]
    fn test_quat_linear_from_linear() {
        let linear = crate::Linear::new(32, 16, true);
        let ql = QuatLinear::from_linear(&linear);
        assert_eq!(ql.in_features(), 32);
        assert_eq!(ql.out_features(), 16);

        let input = Tensor::from_f32(&vec![0.5; 32], &[1, 32]);
        let output = ql.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 16]);
        let data = output.as_f32_slice().unwrap();
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_quat_linear_compression() {
        let ql = QuatLinear::random(1024, 512, false);
        let ratio = ql.compression_ratio();
        // 2-bit packing: 4 values per byte → ~16× compression
        // With per-row scales overhead it's slightly less
        assert!(ratio > 10.0, "compression ratio {} too low", ratio);
    }

    #[test]
    fn test_quat_linear_dequantize() {
        let weight = Tensor::from_f32(&[3.0, 1.0, -1.0, -3.0, 0.5, -0.5, 1.5, -1.5], &[2, 4]);
        let ql = QuatLinear::new(&weight, None);
        let deq = ql.dequantize_weights();
        assert_eq!(deq.shape().dims(), &[2, 4]);
        let data = deq.as_f32_slice().unwrap();
        assert!(data.iter().all(|v| v.is_finite()));
        // Signs should be preserved for large values
        assert!(data[0] > 0.0); // was 3.0
        assert!(data[3] < 0.0); // was -3.0
    }

    #[test]
    fn test_quat_linear_state_dict() {
        let ql = QuatLinear::random(16, 8, true);
        let sd = ql.state_dict();
        assert!(sd.contains_key("weight"));
        assert!(sd.contains_key("bias"));
        assert_eq!(sd["weight"].shape().dims(), &[8, 16]);
    }
}
