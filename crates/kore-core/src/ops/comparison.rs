//! Comparison operations on tensors.

use crate::dtype::DType;
use crate::error::KoreError;
use crate::tensor::Tensor;
use crate::Result;

impl Tensor {
    /// Element-wise equality check. Returns a tensor of 1.0 (true) or 0.0 (false).
    pub fn eq_tensor(&self, other: &Tensor) -> Result<Tensor> {
        cmp_op(self, other, |a, b| if (a - b).abs() < 1e-7 { 1.0 } else { 0.0 })
    }

    /// Element-wise greater-than. Returns 1.0 where self > other.
    pub fn gt(&self, other: &Tensor) -> Result<Tensor> {
        cmp_op(self, other, |a, b| if a > b { 1.0 } else { 0.0 })
    }

    /// Element-wise less-than. Returns 1.0 where self < other.
    pub fn lt(&self, other: &Tensor) -> Result<Tensor> {
        cmp_op(self, other, |a, b| if a < b { 1.0 } else { 0.0 })
    }

    /// Element-wise greater-than-or-equal. Returns 1.0 where self >= other.
    pub fn ge(&self, other: &Tensor) -> Result<Tensor> {
        cmp_op(self, other, |a, b| if a >= b { 1.0 } else { 0.0 })
    }

    /// Element-wise less-than-or-equal. Returns 1.0 where self <= other.
    pub fn le(&self, other: &Tensor) -> Result<Tensor> {
        cmp_op(self, other, |a, b| if a <= b { 1.0 } else { 0.0 })
    }
}

fn cmp_op(a: &Tensor, b: &Tensor, op: impl Fn(f32, f32) -> f32) -> Result<Tensor> {
    if a.dtype() != DType::F32 || b.dtype() != DType::F32 {
        return Err(KoreError::DTypeMismatch {
            expected: a.dtype(),
            got: b.dtype(),
        });
    }
    if a.shape() != b.shape() {
        return Err(KoreError::ShapeMismatch {
            expected: a.shape().dims().to_vec(),
            got: b.shape().dims().to_vec(),
        });
    }

    let a = a.contiguous();
    let b = b.contiguous();
    let a_data = a.as_f32_slice().unwrap();
    let b_data = b.as_f32_slice().unwrap();
    let result: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(&x, &y)| op(x, y))
        .collect();

    Ok(Tensor::from_f32(&result, a.shape().dims()))
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn test_eq() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_f32(&[1.0, 0.0, 3.0], &[3]);
        let c = a.eq_tensor(&b).unwrap();
        assert_eq!(c.as_f32_slice().unwrap(), &[1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_gt_lt() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_f32(&[2.0, 2.0, 1.0], &[3]);
        assert_eq!(a.gt(&b).unwrap().as_f32_slice().unwrap(), &[0.0, 0.0, 1.0]);
        assert_eq!(a.lt(&b).unwrap().as_f32_slice().unwrap(), &[1.0, 0.0, 0.0]);
    }
}
