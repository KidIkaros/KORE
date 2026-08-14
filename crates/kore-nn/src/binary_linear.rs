//! BinaryLinear: 1-bit linear layer using binary weights {-1, +1}.
//!
//! Uses kore-btes for binary encoding and kore-kernels for CPU binary matrix
//! multiplication. Weights are stored packed (1 bit per weight) with a single
//! f32 scale per row, then dequantized IN-MEMORY (not on-disk) before the matmul.
//!
//! Memory: 1 bit/param vs 32 bits/param for f32 Linear → **32× compression**.
//!
//! Forward: `y = binary_matmul(W_packed, scale, x) + bias`

use std::collections::HashMap;

use kore_core::{DType, Tensor};
use kore_kernels::cpu_binary_matmul::{binary_matmul, pack_weights_binary};

use crate::module::Module;

/// A linear layer with binary-quantized weights (1-bit).
///
/// Stores weights in packed format (1 bit per weight) with a single f32 scale per row.
/// Forward pass dequantizes in-memory and uses CPU binary matmul from kore-kernels.
pub struct BinaryLinear {
    /// Packed binary weights: [out_features, ceil(in_features/8)] bytes
    packed_weights: Vec<u8>,
    /// Per-row scale factor: [out_features]
    scales: Vec<f32>,
    /// Optional bias: [out_features]
    bias: Option<Tensor>,
    /// Logical dimensions
    in_features: usize,
    out_features: usize,
    training: bool,
}

impl BinaryLinear {
    /// Create a BinaryLinear layer by quantizing f32 weights to binary.
    ///
    /// # Arguments
    /// * `weight` - f32 weight tensor of shape [out_features, in_features]
    /// * `bias` - optional f32 bias tensor of shape [out_features]
    pub fn new(weight: &Tensor, bias: Option<&Tensor>) -> Self {
        let dims = weight.shape().dims();
        assert!(
            dims.len() == 2,
            "BinaryLinear weight must be 2D, got {:?}",
            dims
        );
        let out_features = dims[0];
        let in_features = dims[1];

        let w_data = weight.contiguous();
        let w_slice = w_data
            .as_f32_slice()
            .expect("BinaryLinear requires f32 weights");

        let (packed_weights, scales) = pack_weights_binary(w_slice, out_features, in_features);

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

    /// Create a BinaryLinear from an existing f32 Linear layer.
    pub fn from_linear(linear: &crate::Linear) -> Self {
        Self::new(linear.weight(), linear.bias())
    }

    /// Create a BinaryLinear with random binary weights (for testing).
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

    /// Packed weight bytes (1 bit per weight).
    pub fn packed_weights(&self) -> &[u8] {
        &self.packed_weights
    }

    /// Per-row scale factors.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Memory usage in bytes for weights (packed binary + scales).
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
        let k_packed = self.in_features.div_ceil(8);
        let mut result = vec![0.0f32; self.out_features * self.in_features];

        for row in 0..self.out_features {
            let scale = self.scales[row];
            let row_packed = &self.packed_weights[row * k_packed..(row + 1) * k_packed];

            let mut col = 0;
            for &byte in row_packed {
                for bit in 0..8 {
                    if col >= self.in_features {
                        break;
                    }
                    let val = if (byte >> bit) & 1 == 1 { 1.0 } else { -1.0 };
                    result[row * self.in_features + col] = val * scale;
                    col += 1;
                }
                if col >= self.in_features {
                    break;
                }
            }
        }

        Tensor::from_f32(&result, &[self.out_features, self.in_features])
    }
}

impl Module for BinaryLinear {
    /// Forward pass: y = binary_matmul(W_packed, scale, x) + bias
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

        // binary_matmul expects: a_packed[M,K] @ b[K,N] → [M,N]
        // We need: output[batch, out] = input[batch, in] @ W^T[in, out]
        // Which is: output^T[out, batch] = W[out, in] @ input^T[in, batch]
        let input_t = input_2d.transpose()?.contiguous();

        let mut output = binary_matmul(
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
            let mut out_shape: Vec<isize> =
                input_dims[..ndim - 1].iter().map(|&d| d as isize).collect();
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

impl std::fmt::Display for BinaryLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BinaryLinear(in={}, out={}, compression={:.1}x, mem={:.1}KB)",
            self.in_features,
            self.out_features,
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
    fn test_binary_linear_creation() {
        let bl = BinaryLinear::random(64, 32, true);
        assert_eq!(bl.in_features(), 64);
        assert_eq!(bl.out_features(), 32);
        assert!(bl.compression_ratio() > 10.0);
    }

    #[test]
    fn test_binary_linear_forward_shape() {
        let bl = BinaryLinear::random(16, 8, false);

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
    fn test_binary_linear_from_linear() {
        let linear = Linear::new(8, 4, true);
        let bl = BinaryLinear::from_linear(&linear);

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
    fn test_binary_linear_approximates_linear() {
        // Create weights that are binary-friendly (all ±1)
        let w_data = vec![
            1.0, -1.0, 1.0, -1.0, // row 0
            -1.0, 1.0, -1.0, 1.0, // row 1
        ];
        let weight = Tensor::from_f32(&w_data, &[2, 4]);
        let bl = BinaryLinear::new(&weight, None);

        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let y = bl.forward(&x).unwrap();
        let y_data = y.as_f32_slice().unwrap();

        // Expected (binary, no quantization error since weights are exactly ±1):
        // row0 = 1*1 + (-1)*2 + 1*3 + (-1)*4 = 1 - 2 + 3 - 4 = -2
        // row1 = (-1)*1 + 1*2 + (-1)*3 + 1*4 = -1 + 2 - 3 + 4 = 2
        // With scale = abs_max = 1.0, exact match expected
        assert!((y_data[0] - (-2.0)).abs() < 0.5, "got {}", y_data[0]);
        assert!((y_data[1] - 2.0).abs() < 0.5, "got {}", y_data[1]);
    }

    #[test]
    fn test_binary_linear_compression() {
        let bl = BinaryLinear::random(4096, 4096, false);
        let ratio = bl.compression_ratio();
        // 1 bit/param + 4 bytes scale per row vs 32 bits/param
        // Should be ~20-30x
        assert!(ratio > 15.0, "compression ratio {} too low", ratio);
    }

    #[test]
    fn test_binary_linear_dequantize() {
        let w_data = vec![1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0];
        let weight = Tensor::from_f32(&w_data, &[2, 4]);
        let bl = BinaryLinear::new(&weight, None);

        let deq = bl.dequantize_weights();
        assert_eq!(deq.shape().dims(), &[2, 4]);
        let deq_data = deq.as_f32_slice().unwrap();
        // Signs should match (scale=1.0 for these weights)
        assert!(deq_data[0] > 0.0);
        assert!(deq_data[1] < 0.0);
        assert!(deq_data[2] > 0.0);
        assert!(deq_data[3] < 0.0);
    }

    #[test]
    fn test_binary_linear_state_dict() {
        let bl = BinaryLinear::random(8, 4, true);
        let sd = bl.state_dict();
        assert!(sd.contains_key("weight"));
        assert!(sd.contains_key("bias"));
        assert_eq!(sd["weight"].shape().dims(), &[4, 8]);
    }

    #[test]
    fn test_binary_linear_bias() {
        let w = Tensor::from_f32(&[1.0, -1.0, 1.0, -1.0], &[2, 2]);
        let b = Tensor::from_f32(&[10.0, 20.0], &[2]);
        let bl = BinaryLinear::new(&w, Some(&b));

        let x = Tensor::from_f32(&[1.0, 1.0], &[1, 2]);
        let y = bl.forward(&x).unwrap();
        let y_data = y.as_f32_slice().unwrap();
        // Output should include bias offset
        assert!(y_data[0] > 5.0, "bias not applied: got {}", y_data[0]);
        assert!(y_data[1] > 15.0, "bias not applied: got {}", y_data[1]);
    }
}
