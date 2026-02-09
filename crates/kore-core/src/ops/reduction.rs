//! Reduction operations: sum, mean, max, min.

use rayon::prelude::*;

use crate::autograd;
use crate::dtype::DType;
use crate::error::KoreError;
use crate::tensor::Tensor;
use crate::Result;

const PAR_THRESHOLD: usize = 8192;

impl Tensor {
    /// Sum all elements, returning a scalar tensor.
    pub fn sum(&self) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(KoreError::UnsupportedDType(self.dtype()));
        }
        let data = self.contiguous();
        let slice = data.as_f32_slice().unwrap();
        let total: f32 = if slice.len() >= PAR_THRESHOLD {
            slice.par_iter().sum()
        } else {
            slice.iter().sum()
        };
        let result = Tensor::scalar(total);
        if self.tracks_grad() && autograd::is_grad_enabled() {
            let node = autograd::GradNode::with_grad_fn(
                Box::new(autograd::SumBackward { input_shape: self.shape().dims().to_vec() }),
                self.grad_node().into_iter().cloned().collect(),
            );
            Ok(result.with_grad_node(node))
        } else {
            Ok(result)
        }
    }

    /// Sum along a specific axis, reducing that dimension.
    pub fn sum_axis(&self, axis: usize) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(KoreError::UnsupportedDType(self.dtype()));
        }
        if axis >= self.ndim() {
            return Err(KoreError::InvalidAxis {
                axis,
                ndim: self.ndim(),
            });
        }

        let data = self.contiguous();
        let dims = data.shape().dims();
        let slice = data.as_f32_slice().unwrap();

        // Build output shape (remove the axis dimension)
        let mut out_dims: Vec<usize> = dims.to_vec();
        out_dims.remove(axis);
        if out_dims.is_empty() {
            return self.sum();
        }

        let out_numel: usize = out_dims.iter().product();
        let mut result = vec![0.0f32; out_numel];

        let axis_size = dims[axis];
        let outer_size: usize = dims[..axis].iter().product();
        let inner_size: usize = dims[axis + 1..].iter().product();

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut acc = 0.0f32;
                for k in 0..axis_size {
                    let idx = outer * axis_size * inner_size + k * inner_size + inner;
                    acc += slice[idx];
                }
                result[outer * inner_size + inner] = acc;
            }
        }

        Ok(Tensor::from_f32(&result, &out_dims))
    }

    /// Mean of all elements, returning a scalar tensor.
    pub fn mean(&self) -> Result<Tensor> {
        let s = self.sum()?;
        let n = self.numel() as f32;
        s.mul_scalar(1.0 / n)
    }

    /// Mean along a specific axis.
    pub fn mean_axis(&self, axis: usize) -> Result<Tensor> {
        let s = self.sum_axis(axis)?;
        let n = self.shape().dim(axis).ok_or(KoreError::InvalidAxis {
            axis,
            ndim: self.ndim(),
        })? as f32;
        s.mul_scalar(1.0 / n)
    }

    /// Maximum element, returning a scalar tensor.
    pub fn max(&self) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(KoreError::UnsupportedDType(self.dtype()));
        }
        let data = self.contiguous();
        let slice = data.as_f32_slice().unwrap();
        let val = if slice.len() >= PAR_THRESHOLD {
            slice.par_iter().cloned().reduce(|| f32::NEG_INFINITY, f32::max)
        } else {
            slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        };
        Ok(Tensor::scalar(val))
    }

    /// Minimum element, returning a scalar tensor.
    pub fn min(&self) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(KoreError::UnsupportedDType(self.dtype()));
        }
        let data = self.contiguous();
        let slice = data.as_f32_slice().unwrap();
        let val = if slice.len() >= PAR_THRESHOLD {
            slice.par_iter().cloned().reduce(|| f32::INFINITY, f32::min)
        } else {
            slice.iter().cloned().fold(f32::INFINITY, f32::min)
        };
        Ok(Tensor::scalar(val))
    }

    /// Index of the maximum element along the last axis.
    pub fn argmax(&self, axis: isize) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(KoreError::UnsupportedDType(self.dtype()));
        }

        let ndim = self.ndim();
        let axis = if axis < 0 {
            (ndim as isize + axis) as usize
        } else {
            axis as usize
        };

        if axis >= ndim {
            return Err(KoreError::InvalidAxis { axis, ndim });
        }

        let data = self.contiguous();
        let dims = data.shape().dims();
        let slice = data.as_f32_slice().unwrap();

        let axis_size = dims[axis];
        let outer_size: usize = dims[..axis].iter().product();
        let inner_size: usize = dims[axis + 1..].iter().product();

        let out_numel = outer_size * inner_size;
        let mut result = vec![0.0f32; out_numel]; // store as f32 for simplicity

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut best_val = f32::NEG_INFINITY;
                let mut best_idx = 0usize;
                for k in 0..axis_size {
                    let idx = outer * axis_size * inner_size + k * inner_size + inner;
                    if slice[idx] > best_val {
                        best_val = slice[idx];
                        best_idx = k;
                    }
                }
                result[outer * inner_size + inner] = best_idx as f32;
            }
        }

        let mut out_dims: Vec<usize> = dims.to_vec();
        out_dims.remove(axis);
        if out_dims.is_empty() {
            out_dims.push(1);
        }

        Ok(Tensor::from_f32(&result, &out_dims))
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn test_sum() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let s = t.sum().unwrap();
        assert_eq!(s.get_f32(0).unwrap(), 10.0);
    }

    #[test]
    fn test_sum_axis() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

        let s0 = t.sum_axis(0).unwrap();
        assert_eq!(s0.shape().dims(), &[3]);
        assert_eq!(s0.as_f32_slice().unwrap(), &[5.0, 7.0, 9.0]);

        let s1 = t.sum_axis(1).unwrap();
        assert_eq!(s1.shape().dims(), &[2]);
        assert_eq!(s1.as_f32_slice().unwrap(), &[6.0, 15.0]);
    }

    #[test]
    fn test_mean() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let m = t.mean().unwrap();
        assert!((m.get_f32(0).unwrap() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_mean_axis() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let m = t.mean_axis(0).unwrap();
        assert_eq!(m.shape().dims(), &[3]);
        let data = m.as_f32_slice().unwrap();
        assert!((data[0] - 2.5).abs() < 1e-6);
        assert!((data[1] - 3.5).abs() < 1e-6);
        assert!((data[2] - 4.5).abs() < 1e-6);
    }

    #[test]
    fn test_max_min() {
        let t = Tensor::from_f32(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5]);
        assert_eq!(t.max().unwrap().get_f32(0).unwrap(), 5.0);
        assert_eq!(t.min().unwrap().get_f32(0).unwrap(), 1.0);
    }

    #[test]
    fn test_argmax() {
        let t = Tensor::from_f32(&[1.0, 3.0, 2.0, 5.0, 4.0, 6.0], &[2, 3]);
        let am = t.argmax(-1).unwrap();
        assert_eq!(am.shape().dims(), &[2]);
        assert_eq!(am.as_f32_slice().unwrap(), &[1.0, 2.0]);
    }
}
