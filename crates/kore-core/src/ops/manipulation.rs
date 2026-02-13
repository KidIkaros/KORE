//! Tensor manipulation operations: cat, stack, split, chunk, where, masked_fill,
//! triu, tril, pad, softmax.

use crate::dtype::DType;
use crate::error::KoreError;
use crate::tensor::Tensor;
use crate::Result;

impl Tensor {
    /// Concatenate tensors along a given axis.
    ///
    /// All tensors must have the same shape except along `axis`.
    pub fn cat(tensors: &[&Tensor], axis: isize) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(KoreError::StorageError("cat: empty tensor list".into()));
        }
        let first = tensors[0];
        let ndim = first.ndim();
        if ndim == 0 {
            return Err(KoreError::StorageError("cat: cannot concatenate scalars".into()));
        }

        let axis = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };
        if axis >= ndim {
            return Err(KoreError::StorageError(format!("cat: axis {} out of range for {}D tensor", axis, ndim)));
        }

        // Validate shapes match on all non-cat axes
        for t in &tensors[1..] {
            if t.ndim() != ndim {
                return Err(KoreError::ShapeMismatch {
                    expected: first.shape().dims().to_vec(),
                    got: t.shape().dims().to_vec(),
                });
            }
            for d in 0..ndim {
                if d != axis && t.shape().dims()[d] != first.shape().dims()[d] {
                    return Err(KoreError::ShapeMismatch {
                        expected: first.shape().dims().to_vec(),
                        got: t.shape().dims().to_vec(),
                    });
                }
            }
        }

        // Compute output shape
        let mut out_shape: Vec<usize> = first.shape().dims().to_vec();
        let cat_dim: usize = tensors.iter().map(|t| t.shape().dims()[axis]).sum();
        out_shape[axis] = cat_dim;

        let numel: usize = out_shape.iter().product();
        let mut result = vec![0.0f32; numel];

        // Compute strides for output
        let outer: usize = out_shape[..axis].iter().product();
        let inner: usize = out_shape[axis + 1..].iter().product();

        let mut cat_offset = 0;
        for t in tensors {
            let t_cont = t.contiguous();
            let t_data = t_cont.as_f32_slice().ok_or(KoreError::UnsupportedDType(t.dtype()))?;
            let t_axis_size = t.shape().dims()[axis];

            for o in 0..outer {
                for a in 0..t_axis_size {
                    let src_start = (o * t_axis_size + a) * inner;
                    let dst_start = (o * cat_dim + (cat_offset + a)) * inner;
                    result[dst_start..dst_start + inner]
                        .copy_from_slice(&t_data[src_start..src_start + inner]);
                }
            }
            cat_offset += t_axis_size;
        }

        Ok(Tensor::from_f32(&result, &out_shape))
    }

    /// Stack tensors along a new axis.
    ///
    /// All tensors must have the same shape. A new dimension is inserted at `axis`.
    pub fn stack(tensors: &[&Tensor], axis: isize) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(KoreError::StorageError("stack: empty tensor list".into()));
        }
        let first = tensors[0];
        let ndim = first.ndim();
        let axis = if axis < 0 { (ndim as isize + 1 + axis) as usize } else { axis as usize };
        if axis > ndim {
            return Err(KoreError::StorageError(format!("stack: axis {} out of range", axis)));
        }

        // Validate all shapes match
        for t in &tensors[1..] {
            if t.shape().dims() != first.shape().dims() {
                return Err(KoreError::ShapeMismatch {
                    expected: first.shape().dims().to_vec(),
                    got: t.shape().dims().to_vec(),
                });
            }
        }

        // Unsqueeze each tensor at axis, then cat
        let mut unsqueezed: Vec<Tensor> = Vec::with_capacity(tensors.len());
        for t in tensors {
            let mut new_shape: Vec<isize> = t.shape().dims().iter().map(|&d| d as isize).collect();
            new_shape.insert(axis, 1);
            unsqueezed.push(t.reshape(&new_shape)?);
        }

        let refs: Vec<&Tensor> = unsqueezed.iter().collect();
        Tensor::cat(&refs, axis as isize)
    }

    /// Split tensor into chunks along an axis.
    ///
    /// Returns `ceil(dim[axis] / chunk_size)` tensors.
    pub fn chunk(&self, chunks: usize, axis: isize) -> Result<Vec<Tensor>> {
        if chunks == 0 {
            return Err(KoreError::StorageError("chunk: chunks must be > 0".into()));
        }
        let ndim = self.ndim();
        let axis = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };
        if axis >= ndim {
            return Err(KoreError::StorageError(format!("chunk: axis {} out of range", axis)));
        }

        let dim_size = self.shape().dims()[axis];
        let chunk_size = dim_size.div_ceil(chunks);
        self.split(chunk_size, axis as isize)
    }

    /// Split tensor into pieces of `split_size` along an axis.
    ///
    /// Last piece may be smaller.
    pub fn split(&self, split_size: usize, axis: isize) -> Result<Vec<Tensor>> {
        if split_size == 0 {
            return Err(KoreError::StorageError("split: split_size must be > 0".into()));
        }
        let ndim = self.ndim();
        let axis = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };
        if axis >= ndim {
            return Err(KoreError::StorageError(format!("split: axis {} out of range", axis)));
        }

        let dim_size = self.shape().dims()[axis];
        let data = self.contiguous();
        let src = data.as_f32_slice().ok_or(KoreError::UnsupportedDType(self.dtype()))?;

        let outer: usize = self.shape().dims()[..axis].iter().product();
        let inner: usize = self.shape().dims()[axis + 1..].iter().product();

        let mut results = Vec::new();
        let mut offset = 0;
        while offset < dim_size {
            let this_size = split_size.min(dim_size - offset);
            let mut chunk_shape = self.shape().dims().to_vec();
            chunk_shape[axis] = this_size;
            let chunk_numel: usize = chunk_shape.iter().product();
            let mut chunk_data = vec![0.0f32; chunk_numel];

            for o in 0..outer {
                for a in 0..this_size {
                    let src_start = (o * dim_size + (offset + a)) * inner;
                    let dst_start = (o * this_size + a) * inner;
                    chunk_data[dst_start..dst_start + inner]
                        .copy_from_slice(&src[src_start..src_start + inner]);
                }
            }

            results.push(Tensor::from_f32(&chunk_data, &chunk_shape));
            offset += this_size;
        }

        Ok(results)
    }

    /// Element-wise conditional: `where(condition, self, other)`.
    ///
    /// Returns `self` where `condition > 0`, else `other`.
    /// All three tensors must have the same shape.
    pub fn where_cond(&self, condition: &Tensor, other: &Tensor) -> Result<Tensor> {
        if self.dtype() != DType::F32 || other.dtype() != DType::F32 || condition.dtype() != DType::F32 {
            return Err(KoreError::UnsupportedDType(self.dtype()));
        }
        if self.shape() != other.shape() || self.shape() != condition.shape() {
            return Err(KoreError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: other.shape().dims().to_vec(),
            });
        }

        let a = self.contiguous();
        let b = other.contiguous();
        let c = condition.contiguous();
        let a_data = a.as_f32_slice().unwrap();
        let b_data = b.as_f32_slice().unwrap();
        let c_data = c.as_f32_slice().unwrap();

        let result: Vec<f32> = a_data.iter().zip(b_data.iter()).zip(c_data.iter())
            .map(|((&a, &b), &c)| if c > 0.0 { a } else { b })
            .collect();

        Ok(Tensor::from_f32(&result, self.shape().dims()))
    }

    /// Replace elements where `mask > 0` with `value`.
    pub fn masked_fill(&self, mask: &Tensor, value: f32) -> Result<Tensor> {
        if self.dtype() != DType::F32 || mask.dtype() != DType::F32 {
            return Err(KoreError::UnsupportedDType(self.dtype()));
        }
        if self.shape() != mask.shape() {
            return Err(KoreError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: mask.shape().dims().to_vec(),
            });
        }

        let data = self.contiguous();
        let m = mask.contiguous();
        let d = data.as_f32_slice().unwrap();
        let m_data = m.as_f32_slice().unwrap();

        let result: Vec<f32> = d.iter().zip(m_data.iter())
            .map(|(&v, &m)| if m > 0.0 { value } else { v })
            .collect();

        Ok(Tensor::from_f32(&result, self.shape().dims()))
    }

    /// Upper triangular matrix. Elements below the k-th diagonal are set to 0.
    pub fn triu(&self, k: isize) -> Result<Tensor> {
        tri_op(self, k, true)
    }

    /// Lower triangular matrix. Elements above the k-th diagonal are set to 0.
    pub fn tril(&self, k: isize) -> Result<Tensor> {
        tri_op(self, k, false)
    }

    /// Softmax over the last dimension.
    pub fn softmax(&self, axis: isize) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(KoreError::UnsupportedDType(self.dtype()));
        }
        let ndim = self.ndim();
        let axis = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };
        if axis >= ndim {
            return Err(KoreError::StorageError(format!("softmax: axis {} out of range", axis)));
        }

        let data = self.contiguous();
        let src = data.as_f32_slice().unwrap();
        let mut result = src.to_vec();

        let outer: usize = self.shape().dims()[..axis].iter().product();
        let axis_size = self.shape().dims()[axis];
        let inner: usize = self.shape().dims()[axis + 1..].iter().product();

        for o in 0..outer {
            for i in 0..inner {
                // Find max for numerical stability
                let mut max_val = f32::NEG_INFINITY;
                for a in 0..axis_size {
                    let idx = (o * axis_size + a) * inner + i;
                    if result[idx] > max_val {
                        max_val = result[idx];
                    }
                }

                // Exp and sum
                let mut sum = 0.0f32;
                for a in 0..axis_size {
                    let idx = (o * axis_size + a) * inner + i;
                    result[idx] = (result[idx] - max_val).exp();
                    sum += result[idx];
                }

                // Normalize
                if sum > 0.0 {
                    for a in 0..axis_size {
                        let idx = (o * axis_size + a) * inner + i;
                        result[idx] /= sum;
                    }
                }
            }
        }

        Ok(Tensor::from_f32(&result, self.shape().dims()))
    }

    /// Log-softmax over the given axis.
    pub fn log_softmax(&self, axis: isize) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(KoreError::UnsupportedDType(self.dtype()));
        }
        let ndim = self.ndim();
        let axis = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };
        if axis >= ndim {
            return Err(KoreError::StorageError(format!("log_softmax: axis {} out of range", axis)));
        }

        let data = self.contiguous();
        let src = data.as_f32_slice().unwrap();
        let mut result = src.to_vec();

        let outer: usize = self.shape().dims()[..axis].iter().product();
        let axis_size = self.shape().dims()[axis];
        let inner: usize = self.shape().dims()[axis + 1..].iter().product();

        for o in 0..outer {
            for i in 0..inner {
                let mut max_val = f32::NEG_INFINITY;
                for a in 0..axis_size {
                    let idx = (o * axis_size + a) * inner + i;
                    if result[idx] > max_val {
                        max_val = result[idx];
                    }
                }

                let mut log_sum_exp = 0.0f32;
                for a in 0..axis_size {
                    let idx = (o * axis_size + a) * inner + i;
                    log_sum_exp += (result[idx] - max_val).exp();
                }
                let log_sum_exp = max_val + log_sum_exp.ln();

                for a in 0..axis_size {
                    let idx = (o * axis_size + a) * inner + i;
                    result[idx] -= log_sum_exp;
                }
            }
        }

        Ok(Tensor::from_f32(&result, self.shape().dims()))
    }

    /// Gather elements along an axis using index tensor.
    ///
    /// `index` must have the same number of dimensions as `self`.
    /// Output has the same shape as `index`.
    pub fn gather(&self, axis: isize, index: &Tensor) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(KoreError::UnsupportedDType(self.dtype()));
        }
        let ndim = self.ndim();
        let axis = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };
        if axis >= ndim {
            return Err(KoreError::StorageError(format!("gather: axis {} out of range", axis)));
        }

        let data = self.contiguous();
        let src = data.as_f32_slice().unwrap();
        let idx_data = index.contiguous();
        let indices = idx_data.as_f32_slice().ok_or(KoreError::UnsupportedDType(index.dtype()))?;

        let out_shape = index.shape().dims();
        let numel: usize = out_shape.iter().product();
        let mut result = vec![0.0f32; numel];

        let src_shape = self.shape().dims();
        let src_strides = compute_strides(src_shape);
        let out_strides = compute_strides(out_shape);

        for flat_idx in 0..numel {
            // Convert flat index to multi-dimensional index in output
            let mut multi_idx = vec![0usize; ndim];
            let mut remaining = flat_idx;
            for d in 0..ndim {
                multi_idx[d] = remaining / out_strides[d];
                remaining %= out_strides[d];
            }

            // Replace the axis dimension with the gathered index
            let gathered_idx = indices[flat_idx] as usize;
            if gathered_idx >= src_shape[axis] {
                return Err(KoreError::StorageError(format!(
                    "gather: index {} out of range for axis {} with size {}",
                    gathered_idx, axis, src_shape[axis]
                )));
            }
            multi_idx[axis] = gathered_idx;

            // Convert back to flat index in source
            let src_flat: usize = multi_idx.iter().zip(src_strides.iter()).map(|(&i, &s)| i * s).sum();
            result[flat_idx] = src[src_flat];
        }

        Ok(Tensor::from_f32(&result, out_shape))
    }
}

fn tri_op(tensor: &Tensor, k: isize, upper: bool) -> Result<Tensor> {
    if tensor.dtype() != DType::F32 {
        return Err(KoreError::UnsupportedDType(tensor.dtype()));
    }
    let dims = tensor.shape().dims();
    if dims.len() < 2 {
        return Err(KoreError::StorageError("triu/tril requires at least 2D tensor".into()));
    }

    let data = tensor.contiguous();
    let src = data.as_f32_slice().unwrap();
    let mut result = src.to_vec();

    let rows = dims[dims.len() - 2];
    let cols = dims[dims.len() - 1];
    let batch: usize = dims[..dims.len() - 2].iter().product();

    for b in 0..batch.max(1) {
        for r in 0..rows {
            for c in 0..cols {
                let idx = b * rows * cols + r * cols + c;
                let diag = c as isize - r as isize;
                if upper {
                    if diag < k { result[idx] = 0.0; }
                } else if diag > k {
                    result[idx] = 0.0;
                }
            }
        }
    }

    Ok(Tensor::from_f32(&result, dims))
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn test_cat_axis0() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_f32(&[5.0, 6.0], &[1, 2]);
        let c = Tensor::cat(&[&a, &b], 0).unwrap();
        assert_eq!(c.shape().dims(), &[3, 2]);
        assert_eq!(c.as_f32_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_cat_axis1() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[2, 3]);
        let c = Tensor::cat(&[&a, &b], 1).unwrap();
        assert_eq!(c.shape().dims(), &[2, 5]);
        assert_eq!(c.as_f32_slice().unwrap(), &[1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn test_stack() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_f32(&[4.0, 5.0, 6.0], &[3]);
        let c = Tensor::stack(&[&a, &b], 0).unwrap();
        assert_eq!(c.shape().dims(), &[2, 3]);
        assert_eq!(c.as_f32_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_stack_axis1() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_f32(&[4.0, 5.0, 6.0], &[3]);
        let c = Tensor::stack(&[&a, &b], 1).unwrap();
        assert_eq!(c.shape().dims(), &[3, 2]);
        assert_eq!(c.as_f32_slice().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_split() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);
        let parts = a.split(2, 0).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].as_f32_slice().unwrap(), &[1.0, 2.0]);
        assert_eq!(parts[1].as_f32_slice().unwrap(), &[3.0, 4.0]);
        assert_eq!(parts[2].as_f32_slice().unwrap(), &[5.0, 6.0]);
    }

    #[test]
    fn test_chunk() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
        let parts = a.chunk(2, 0).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].as_f32_slice().unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(parts[1].as_f32_slice().unwrap(), &[4.0, 5.0]);
    }

    #[test]
    fn test_where_cond() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_f32(&[10.0, 20.0, 30.0], &[3]);
        let cond = Tensor::from_f32(&[1.0, 0.0, 1.0], &[3]);
        let c = a.where_cond(&cond, &b).unwrap();
        assert_eq!(c.as_f32_slice().unwrap(), &[1.0, 20.0, 3.0]);
    }

    #[test]
    fn test_masked_fill() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let mask = Tensor::from_f32(&[0.0, 1.0, 1.0, 0.0], &[2, 2]);
        let b = a.masked_fill(&mask, f32::NEG_INFINITY).unwrap();
        let data = b.as_f32_slice().unwrap();
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], f32::NEG_INFINITY);
        assert_eq!(data[2], f32::NEG_INFINITY);
        assert_eq!(data[3], 4.0);
    }

    #[test]
    fn test_triu() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
        let u = a.triu(0).unwrap();
        assert_eq!(u.as_f32_slice().unwrap(), &[1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]);
    }

    #[test]
    fn test_tril() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
        let l = a.tril(0).unwrap();
        assert_eq!(l.as_f32_slice().unwrap(), &[1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_softmax() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        let s = a.softmax(-1).unwrap();
        let data = s.as_f32_slice().unwrap();
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax should sum to 1, got {}", sum);
        assert!(data[2] > data[1] && data[1] > data[0]);
    }

    #[test]
    fn test_softmax_2d() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
        let s = a.softmax(-1).unwrap();
        let data = s.as_f32_slice().unwrap();
        // Each row should sum to 1
        let row0_sum: f32 = data[0..3].iter().sum();
        let row1_sum: f32 = data[3..6].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-5);
        assert!((row1_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        let ls = a.log_softmax(-1).unwrap();
        let data = ls.as_f32_slice().unwrap();
        // exp(log_softmax) should sum to 1
        let sum: f32 = data.iter().map(|v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // All values should be <= 0
        assert!(data.iter().all(|&v| v <= 0.0));
    }

    #[test]
    fn test_gather() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let idx = Tensor::from_f32(&[0.0, 2.0, 1.0, 0.0], &[2, 2]);
        let g = a.gather(1, &idx).unwrap();
        assert_eq!(g.shape().dims(), &[2, 2]);
        let data = g.as_f32_slice().unwrap();
        assert_eq!(data[0], 1.0); // row 0, col 0
        assert_eq!(data[1], 3.0); // row 0, col 2
        assert_eq!(data[2], 5.0); // row 1, col 1
        assert_eq!(data[3], 4.0); // row 1, col 0
    }

    #[test]
    fn test_cat_negative_axis() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[2, 3]);
        let c = Tensor::cat(&[&a, &b], -1).unwrap();
        assert_eq!(c.shape().dims(), &[2, 5]);
    }

    #[test]
    fn test_triu_with_offset() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
        let u = a.triu(1).unwrap();
        assert_eq!(u.as_f32_slice().unwrap(), &[0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]);
    }
}
