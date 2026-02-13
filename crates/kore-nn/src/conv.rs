//! Convolution layers — Conv1d and Conv2d.

use rand::Rng;

use kore_core::{DType, Tensor};
use crate::module::Module;

/// 1D convolution layer: y = conv1d(x, weight) + bias
///
/// Input shape: [batch, in_channels, length]
/// Output shape: [batch, out_channels, out_length]
/// where out_length = (length - kernel_size + 2*padding) / stride + 1
pub struct Conv1d {
    weight: Tensor,  // [out_channels, in_channels, kernel_size]
    bias: Option<Tensor>,  // [out_channels]
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    training: bool,
}

impl Conv1d {
    /// Create a new Conv1d layer.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Self {
        // Kaiming uniform initialization
        let fan_in = in_channels * kernel_size;
        let limit = (6.0 / fan_in as f32).sqrt();
        let total = out_channels * in_channels * kernel_size;
        let mut rng = rand::thread_rng();
        let weight_data: Vec<f32> = (0..total)
            .map(|_| rng.gen_range(-limit..limit))
            .collect();

        let mut weight = Tensor::from_f32(&weight_data, &[out_channels, in_channels, kernel_size]);
        weight.set_requires_grad(true);

        let bias_tensor = if bias {
            let mut b = Tensor::zeros(&[out_channels], DType::F32);
            b.set_requires_grad(true);
            Some(b)
        } else {
            None
        };

        Self {
            weight,
            bias: bias_tensor,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            training: true,
        }
    }

    /// Compute output length.
    pub fn output_length(&self, input_length: usize) -> kore_core::Result<usize> {
        let padded = input_length + 2 * self.padding;
        if padded < self.kernel_size {
            return Err(kore_core::KoreError::ShapeMismatch {
                expected: vec![self.kernel_size],
                got: vec![padded],
            });
        }
        Ok((padded - self.kernel_size) / self.stride + 1)
    }
}

impl Module for Conv1d {
    fn forward(&self, input: &Tensor) -> kore_core::Result<Tensor> {
        let data = input.contiguous();
        let dims = data.shape().dims().to_vec();
        if dims.len() != 3 {
            return Err(kore_core::KoreError::ShapeMismatch {
                expected: vec![0, 0, 0],
                got: dims,
            });
        }

        let batch = dims[0];
        let _in_ch = dims[1];
        let in_len = dims[2];
        let out_len = self.output_length(in_len)?;
        let x = data.as_f32_slice()
            .ok_or_else(|| kore_core::KoreError::UnsupportedDType(data.dtype()))?;
        let w = self.weight.contiguous();
        let w_data = w.as_f32_slice()
            .ok_or_else(|| kore_core::KoreError::UnsupportedDType(self.weight.dtype()))?;

        let mut output = vec![0.0f32; batch * self.out_channels * out_len];

        for b in 0..batch {
            for oc in 0..self.out_channels {
                for ol in 0..out_len {
                    let mut acc = 0.0f32;
                    let in_start = ol * self.stride;

                    for ic in 0..self.in_channels {
                        for k in 0..self.kernel_size {
                            let in_pos = in_start + k;
                            let in_pos = if in_pos >= self.padding {
                                in_pos - self.padding
                            } else {
                                continue; // padding region
                            };
                            if in_pos >= in_len {
                                continue; // padding region
                            }

                            let x_idx = b * self.in_channels * in_len + ic * in_len + in_pos;
                            let w_idx = oc * self.in_channels * self.kernel_size
                                + ic * self.kernel_size
                                + k;
                            acc += x[x_idx] * w_data[w_idx];
                        }
                    }

                    if let Some(ref bias) = self.bias {
                        let b_data = bias.as_f32_slice()
                            .ok_or_else(|| kore_core::KoreError::UnsupportedDType(bias.dtype()))?;
                        acc += b_data[oc];
                    }

                    output[b * self.out_channels * out_len + oc * out_len + ol] = acc;
                }
            }
        }

        Ok(Tensor::from_f32(&output, &[batch, self.out_channels, out_len]))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
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

/// 2D convolution layer: y = conv2d(x, weight) + bias
///
/// Input shape: [batch, in_channels, height, width]
/// Output shape: [batch, out_channels, out_h, out_w]
pub struct Conv2d {
    weight: Tensor,  // [out_channels, in_channels, kh, kw]
    bias: Option<Tensor>,
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
    padding: usize,
    training: bool,
}

impl Conv2d {
    /// Create a new Conv2d layer with square kernel.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Self {
        Self::new_rect(in_channels, out_channels, kernel_size, kernel_size, stride, padding, bias)
    }

    /// Create with rectangular kernel.
    pub fn new_rect(
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Self {
        let fan_in = in_channels * kernel_h * kernel_w;
        let limit = (6.0 / fan_in as f32).sqrt();
        let total = out_channels * in_channels * kernel_h * kernel_w;
        let mut rng = rand::thread_rng();
        let weight_data: Vec<f32> = (0..total)
            .map(|_| rng.gen_range(-limit..limit))
            .collect();

        let mut weight = Tensor::from_f32(&weight_data, &[out_channels, in_channels, kernel_h, kernel_w]);
        weight.set_requires_grad(true);

        let bias_tensor = if bias {
            let mut b = Tensor::zeros(&[out_channels], DType::F32);
            b.set_requires_grad(true);
            Some(b)
        } else {
            None
        };

        Self {
            weight,
            bias: bias_tensor,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
            training: true,
        }
    }

    /// Create a Conv2d from pre-existing weight and optional bias tensors.
    ///
    /// Weight shape: [out_channels, in_channels, kernel_h, kernel_w]
    /// Bias shape: [out_channels]
    pub fn from_weight(weight: Tensor, bias: Option<Tensor>, stride: usize, padding: usize) -> Self {
        let dims = weight.shape().dims();
        assert_eq!(dims.len(), 4, "Conv2d weight must be 4D");
        Self {
            out_channels: dims[0],
            in_channels: dims[1],
            kernel_h: dims[2],
            kernel_w: dims[3],
            weight,
            bias,
            stride,
            padding,
            training: false,
        }
    }

    /// Weight tensor.
    pub fn weight(&self) -> &Tensor { &self.weight }

    /// Input channels.
    pub fn in_channels(&self) -> usize { self.in_channels }

    /// Output channels.
    pub fn out_channels(&self) -> usize { self.out_channels }

    /// Kernel size (square kernels return kernel_h).
    pub fn kernel_size(&self) -> usize { self.kernel_h }

    /// Stride.
    pub fn stride(&self) -> usize { self.stride }

    /// Padding.
    pub fn padding(&self) -> usize { self.padding }

    /// Bias tensor (if present).
    pub fn bias(&self) -> Option<&Tensor> { self.bias.as_ref() }

    /// Compute output dimensions.
    pub fn output_size(&self, in_h: usize, in_w: usize) -> kore_core::Result<(usize, usize)> {
        let padded_h = in_h + 2 * self.padding;
        let padded_w = in_w + 2 * self.padding;
        if padded_h < self.kernel_h || padded_w < self.kernel_w {
            return Err(kore_core::KoreError::ShapeMismatch {
                expected: vec![self.kernel_h, self.kernel_w],
                got: vec![padded_h, padded_w],
            });
        }
        let out_h = (padded_h - self.kernel_h) / self.stride + 1;
        let out_w = (padded_w - self.kernel_w) / self.stride + 1;
        Ok((out_h, out_w))
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> kore_core::Result<Tensor> {
        let data = input.contiguous();
        let dims = data.shape().dims().to_vec();
        if dims.len() != 4 {
            return Err(kore_core::KoreError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: dims,
            });
        }

        let batch = dims[0];
        let in_h = dims[2];
        let in_w = dims[3];
        let (out_h, out_w) = self.output_size(in_h, in_w)?;
        let x = data.as_f32_slice()
            .ok_or_else(|| kore_core::KoreError::UnsupportedDType(data.dtype()))?;
        let w = self.weight.contiguous();
        let w_data = w.as_f32_slice()
            .ok_or_else(|| kore_core::KoreError::UnsupportedDType(self.weight.dtype()))?;

        let mut output = vec![0.0f32; batch * self.out_channels * out_h * out_w];

        for b in 0..batch {
            for oc in 0..self.out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut acc = 0.0f32;

                        for ic in 0..self.in_channels {
                            for kh in 0..self.kernel_h {
                                for kw in 0..self.kernel_w {
                                    let ih = oh * self.stride + kh;
                                    let iw = ow * self.stride + kw;

                                    let ih = if ih >= self.padding { ih - self.padding } else { continue };
                                    let iw = if iw >= self.padding { iw - self.padding } else { continue };
                                    if ih >= in_h || iw >= in_w { continue; }

                                    let x_idx = b * self.in_channels * in_h * in_w
                                        + ic * in_h * in_w
                                        + ih * in_w
                                        + iw;
                                    let w_idx = oc * self.in_channels * self.kernel_h * self.kernel_w
                                        + ic * self.kernel_h * self.kernel_w
                                        + kh * self.kernel_w
                                        + kw;
                                    acc += x[x_idx] * w_data[w_idx];
                                }
                            }
                        }

                        if let Some(ref bias) = self.bias {
                            let b_data = bias.as_f32_slice()
                                .ok_or_else(|| kore_core::KoreError::UnsupportedDType(bias.dtype()))?;
                            acc += b_data[oc];
                        }

                        let o_idx = b * self.out_channels * out_h * out_w
                            + oc * out_h * out_w
                            + oh * out_w
                            + ow;
                        output[o_idx] = acc;
                    }
                }
            }
        }

        Ok(Tensor::from_f32(&output, &[batch, self.out_channels, out_h, out_w]))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv1d_shape() {
        let conv = Conv1d::new(3, 8, 3, 1, 0, true);
        let input = Tensor::from_f32(&vec![0.1; 2 * 3 * 10], &[2, 3, 10]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 8, 8]); // (10 - 3)/1 + 1 = 8
    }

    #[test]
    fn test_conv1d_with_padding() {
        let conv = Conv1d::new(1, 1, 3, 1, 1, false);
        let input = Tensor::from_f32(&vec![1.0; 1 * 1 * 5], &[1, 1, 5]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 1, 5]); // same padding
    }

    #[test]
    fn test_conv1d_stride() {
        let conv = Conv1d::new(1, 1, 3, 2, 0, false);
        let input = Tensor::from_f32(&vec![1.0; 1 * 1 * 10], &[1, 1, 10]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 1, 4]); // (10 - 3)/2 + 1 = 4
    }

    #[test]
    fn test_conv1d_parameters() {
        let conv = Conv1d::new(3, 8, 5, 1, 0, true);
        assert_eq!(conv.parameters().len(), 2);
    }

    #[test]
    fn test_conv2d_shape() {
        let conv = Conv2d::new(3, 16, 3, 1, 0, true);
        let input = Tensor::from_f32(&vec![0.1; 1 * 3 * 8 * 8], &[1, 3, 8, 8]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 16, 6, 6]); // (8-3)/1+1 = 6
    }

    #[test]
    fn test_conv2d_with_padding() {
        let conv = Conv2d::new(1, 1, 3, 1, 1, false);
        let input = Tensor::from_f32(&vec![1.0; 1 * 1 * 5 * 5], &[1, 1, 5, 5]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 1, 5, 5]); // same padding
    }

    #[test]
    fn test_conv2d_stride() {
        let conv = Conv2d::new(1, 1, 3, 2, 0, false);
        let input = Tensor::from_f32(&vec![1.0; 1 * 1 * 7 * 7], &[1, 1, 7, 7]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 1, 3, 3]); // (7-3)/2+1 = 3
    }

    #[test]
    fn test_conv2d_parameters() {
        let conv = Conv2d::new(3, 16, 3, 1, 0, true);
        assert_eq!(conv.parameters().len(), 2);
        assert_eq!(conv.weight.shape().dims(), &[16, 3, 3, 3]);
    }

    #[test]
    fn test_conv2d_batched() {
        let conv = Conv2d::new(1, 2, 3, 1, 0, false);
        let input = Tensor::from_f32(&vec![0.5; 4 * 1 * 6 * 6], &[4, 1, 6, 6]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[4, 2, 4, 4]);
    }
}
