//! Pooling layers — MaxPool2d, AvgPool2d, AdaptiveAvgPool2d.
//!
//! Essential building blocks for CNN architectures like SqueezeNet, ResNet, etc.
//! All operate on 4D tensors with shape `[batch, channels, height, width]`.

use kore_core::{KoreError, Tensor};
use crate::module::Module;

/// 2D max pooling layer.
///
/// Input shape: `[batch, channels, height, width]`
/// Output shape: `[batch, channels, out_h, out_w]`
/// where `out_h = (height + 2*padding - kernel_size) / stride + 1`
pub struct MaxPool2d {
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl MaxPool2d {
    /// Create a new MaxPool2d layer.
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self { kernel_size, stride, padding }
    }

    /// Convenience: kernel_size = stride, no padding.
    pub fn simple(kernel_size: usize) -> Self {
        Self::new(kernel_size, kernel_size, 0)
    }

    /// Compute output dimensions.
    pub fn output_size(&self, in_h: usize, in_w: usize) -> kore_core::Result<(usize, usize)> {
        let padded_h = in_h + 2 * self.padding;
        let padded_w = in_w + 2 * self.padding;
        if padded_h < self.kernel_size || padded_w < self.kernel_size {
            return Err(KoreError::ShapeMismatch {
                expected: vec![self.kernel_size, self.kernel_size],
                got: vec![padded_h, padded_w],
            });
        }
        let out_h = (padded_h - self.kernel_size) / self.stride + 1;
        let out_w = (padded_w - self.kernel_size) / self.stride + 1;
        Ok((out_h, out_w))
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Tensor) -> kore_core::Result<Tensor> {
        let data = input.contiguous();
        let dims = data.shape().dims().to_vec();
        if dims.len() != 4 {
            return Err(KoreError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: dims,
            });
        }

        let batch = dims[0];
        let channels = dims[1];
        let in_h = dims[2];
        let in_w = dims[3];
        let (out_h, out_w) = self.output_size(in_h, in_w)?;
        let x = data.as_f32_slice()
            .ok_or_else(|| KoreError::UnsupportedDType(data.dtype()))?;

        let mut output = vec![f32::NEG_INFINITY; batch * channels * out_h * out_w];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f32::NEG_INFINITY;

                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                let ih = oh * self.stride + kh;
                                let iw = ow * self.stride + kw;

                                let ih = if ih >= self.padding { ih - self.padding } else { continue };
                                let iw = if iw >= self.padding { iw - self.padding } else { continue };
                                if ih >= in_h || iw >= in_w { continue; }

                                let idx = b * channels * in_h * in_w
                                    + c * in_h * in_w
                                    + ih * in_w
                                    + iw;
                                max_val = max_val.max(x[idx]);
                            }
                        }

                        let o_idx = b * channels * out_h * out_w
                            + c * out_h * out_w
                            + oh * out_w
                            + ow;
                        output[o_idx] = max_val;
                    }
                }
            }
        }

        Ok(Tensor::from_f32(&output, &[batch, channels, out_h, out_w]))
    }

    fn parameters(&self) -> Vec<&Tensor> { vec![] }
    fn named_parameters(&self) -> Vec<(&str, &Tensor)> { vec![] }
}

/// 2D average pooling layer.
///
/// Input shape: `[batch, channels, height, width]`
/// Output shape: `[batch, channels, out_h, out_w]`
pub struct AvgPool2d {
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl AvgPool2d {
    /// Create a new AvgPool2d layer.
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self { kernel_size, stride, padding }
    }

    /// Convenience: kernel_size = stride, no padding.
    pub fn simple(kernel_size: usize) -> Self {
        Self::new(kernel_size, kernel_size, 0)
    }

    /// Compute output dimensions.
    pub fn output_size(&self, in_h: usize, in_w: usize) -> kore_core::Result<(usize, usize)> {
        let padded_h = in_h + 2 * self.padding;
        let padded_w = in_w + 2 * self.padding;
        if padded_h < self.kernel_size || padded_w < self.kernel_size {
            return Err(KoreError::ShapeMismatch {
                expected: vec![self.kernel_size, self.kernel_size],
                got: vec![padded_h, padded_w],
            });
        }
        let out_h = (padded_h - self.kernel_size) / self.stride + 1;
        let out_w = (padded_w - self.kernel_size) / self.stride + 1;
        Ok((out_h, out_w))
    }
}

impl Module for AvgPool2d {
    fn forward(&self, input: &Tensor) -> kore_core::Result<Tensor> {
        let data = input.contiguous();
        let dims = data.shape().dims().to_vec();
        if dims.len() != 4 {
            return Err(KoreError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: dims,
            });
        }

        let batch = dims[0];
        let channels = dims[1];
        let in_h = dims[2];
        let in_w = dims[3];
        let (out_h, out_w) = self.output_size(in_h, in_w)?;
        let x = data.as_f32_slice()
            .ok_or_else(|| KoreError::UnsupportedDType(data.dtype()))?;

        let mut output = vec![0.0f32; batch * channels * out_h * out_w];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;
                        let mut count = 0usize;

                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                let ih = oh * self.stride + kh;
                                let iw = ow * self.stride + kw;

                                let ih = if ih >= self.padding { ih - self.padding } else { continue };
                                let iw = if iw >= self.padding { iw - self.padding } else { continue };
                                if ih >= in_h || iw >= in_w { continue; }

                                let idx = b * channels * in_h * in_w
                                    + c * in_h * in_w
                                    + ih * in_w
                                    + iw;
                                sum += x[idx];
                                count += 1;
                            }
                        }

                        let o_idx = b * channels * out_h * out_w
                            + c * out_h * out_w
                            + oh * out_w
                            + ow;
                        output[o_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                    }
                }
            }
        }

        Ok(Tensor::from_f32(&output, &[batch, channels, out_h, out_w]))
    }

    fn parameters(&self) -> Vec<&Tensor> { vec![] }
    fn named_parameters(&self) -> Vec<(&str, &Tensor)> { vec![] }
}

/// Adaptive 2D average pooling — outputs a fixed spatial size.
///
/// Input shape: `[batch, channels, height, width]`
/// Output shape: `[batch, channels, output_h, output_w]`
///
/// Commonly used with `output_size = (1, 1)` for global average pooling
/// before a classifier head.
pub struct AdaptiveAvgPool2d {
    output_h: usize,
    output_w: usize,
}

impl AdaptiveAvgPool2d {
    /// Create with target output spatial dimensions.
    pub fn new(output_h: usize, output_w: usize) -> Self {
        Self { output_h, output_w }
    }

    /// Global average pooling: output 1×1 per channel.
    pub fn global() -> Self {
        Self::new(1, 1)
    }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, input: &Tensor) -> kore_core::Result<Tensor> {
        let data = input.contiguous();
        let dims = data.shape().dims().to_vec();
        if dims.len() != 4 {
            return Err(KoreError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: dims,
            });
        }

        let batch = dims[0];
        let channels = dims[1];
        let in_h = dims[2];
        let in_w = dims[3];
        let x = data.as_f32_slice()
            .ok_or_else(|| KoreError::UnsupportedDType(data.dtype()))?;

        let mut output = vec![0.0f32; batch * channels * self.output_h * self.output_w];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..self.output_h {
                    for ow in 0..self.output_w {
                        // Compute input region for this output pixel
                        let ih_start = (oh * in_h) / self.output_h;
                        let ih_end = ((oh + 1) * in_h) / self.output_h;
                        let iw_start = (ow * in_w) / self.output_w;
                        let iw_end = ((ow + 1) * in_w) / self.output_w;

                        let mut sum = 0.0f32;
                        let mut count = 0usize;

                        for ih in ih_start..ih_end {
                            for iw in iw_start..iw_end {
                                let idx = b * channels * in_h * in_w
                                    + c * in_h * in_w
                                    + ih * in_w
                                    + iw;
                                sum += x[idx];
                                count += 1;
                            }
                        }

                        let o_idx = b * channels * self.output_h * self.output_w
                            + c * self.output_h * self.output_w
                            + oh * self.output_w
                            + ow;
                        output[o_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                    }
                }
            }
        }

        Ok(Tensor::from_f32(&output, &[batch, channels, self.output_h, self.output_w]))
    }

    fn parameters(&self) -> Vec<&Tensor> { vec![] }
    fn named_parameters(&self) -> Vec<(&str, &Tensor)> { vec![] }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_pool2d_basic() {
        let pool = MaxPool2d::simple(2);
        // [1, 1, 4, 4] → [1, 1, 2, 2]
        let input = Tensor::from_f32(&[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ], &[1, 1, 4, 4]);

        let output = pool.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 1, 2, 2]);
        let data = output.as_f32_slice().unwrap();
        assert_eq!(data, &[6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_max_pool2d_stride() {
        let pool = MaxPool2d::new(3, 1, 0);
        // [1, 1, 4, 4] with kernel=3, stride=1 → [1, 1, 2, 2]
        let input = Tensor::from_f32(&[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ], &[1, 1, 4, 4]);

        let output = pool.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 1, 2, 2]);
        let data = output.as_f32_slice().unwrap();
        assert_eq!(data, &[11.0, 12.0, 15.0, 16.0]);
    }

    #[test]
    fn test_avg_pool2d_basic() {
        let pool = AvgPool2d::simple(2);
        let input = Tensor::from_f32(&[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ], &[1, 1, 4, 4]);

        let output = pool.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 1, 2, 2]);
        let data = output.as_f32_slice().unwrap();
        // avg of [1,2,5,6]=3.5, [3,4,7,8]=5.5, [9,10,13,14]=11.5, [11,12,15,16]=13.5
        assert!((data[0] - 3.5).abs() < 1e-6);
        assert!((data[1] - 5.5).abs() < 1e-6);
        assert!((data[2] - 11.5).abs() < 1e-6);
        assert!((data[3] - 13.5).abs() < 1e-6);
    }

    #[test]
    fn test_adaptive_avg_pool2d_global() {
        let pool = AdaptiveAvgPool2d::global();
        let input = Tensor::from_f32(&[
            1.0, 2.0, 3.0, 4.0,
        ], &[1, 1, 2, 2]);

        let output = pool.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 1, 1, 1]);
        let data = output.as_f32_slice().unwrap();
        assert!((data[0] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_adaptive_avg_pool2d_downsample() {
        let pool = AdaptiveAvgPool2d::new(2, 2);
        let input = Tensor::from_f32(&[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ], &[1, 1, 4, 4]);

        let output = pool.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 1, 2, 2]);
        let data = output.as_f32_slice().unwrap();
        // Top-left quadrant: avg(1,2,5,6) = 3.5
        assert!((data[0] - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_max_pool2d_multichannel() {
        let pool = MaxPool2d::simple(2);
        // [1, 2, 2, 2] → [1, 2, 1, 1]
        let input = Tensor::from_f32(&[
            1.0, 2.0, 3.0, 4.0,  // channel 0
            5.0, 6.0, 7.0, 8.0,  // channel 1
        ], &[1, 2, 2, 2]);

        let output = pool.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 2, 1, 1]);
        let data = output.as_f32_slice().unwrap();
        assert_eq!(data, &[4.0, 8.0]);
    }

    #[test]
    fn test_pool_no_parameters() {
        let pool = MaxPool2d::simple(2);
        assert!(pool.parameters().is_empty());
        assert!(pool.named_parameters().is_empty());
    }

    #[test]
    fn test_adaptive_avg_pool2d_batch() {
        let pool = AdaptiveAvgPool2d::global();
        // Batch of 2, 3 channels, 4×4
        let data: Vec<f32> = (0..2 * 3 * 4 * 4).map(|i| i as f32).collect();
        let input = Tensor::from_f32(&data, &[2, 3, 4, 4]);

        let output = pool.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 3, 1, 1]);
        let out_data = output.as_f32_slice().unwrap();
        // Each channel's global average should be finite
        assert!(out_data.iter().all(|v| v.is_finite()));
        // First channel of first batch: avg(0..16) = 7.5
        assert!((out_data[0] - 7.5).abs() < 1e-4);
    }
}
