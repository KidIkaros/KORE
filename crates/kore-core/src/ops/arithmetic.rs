//! Element-wise arithmetic operations on tensors.

use crate::error::KoreError;
use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::Result;

impl Tensor {
    /// Element-wise addition: self + other.
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        binary_op(self, other, |a, b| a + b)
    }

    /// Element-wise subtraction: self - other.
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        binary_op(self, other, |a, b| a - b)
    }

    /// Element-wise multiplication: self * other.
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        binary_op(self, other, |a, b| a * b)
    }

    /// Element-wise division: self / other.
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        binary_op(self, other, |a, b| a / b)
    }

    /// Element-wise negation: -self.
    pub fn neg(&self) -> Result<Tensor> {
        unary_op(self, |a| -a)
    }

    /// Element-wise absolute value.
    pub fn abs(&self) -> Result<Tensor> {
        unary_op(self, |a| a.abs())
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> Result<Tensor> {
        unary_op(self, |a| a.sqrt())
    }

    /// Element-wise reciprocal: 1/self.
    pub fn reciprocal(&self) -> Result<Tensor> {
        unary_op(self, |a| 1.0 / a)
    }

    /// Element-wise exponential: e^self.
    pub fn exp(&self) -> Result<Tensor> {
        unary_op(self, |a| a.exp())
    }

    /// Element-wise natural logarithm.
    pub fn log(&self) -> Result<Tensor> {
        unary_op(self, |a| a.ln())
    }

    /// Element-wise power: self^exponent.
    pub fn pow_scalar(&self, exponent: f32) -> Result<Tensor> {
        unary_op(self, |a| a.powf(exponent))
    }

    /// Scalar addition: self + scalar.
    pub fn add_scalar(&self, scalar: f32) -> Result<Tensor> {
        unary_op(self, |a| a + scalar)
    }

    /// Scalar multiplication: self * scalar.
    pub fn mul_scalar(&self, scalar: f32) -> Result<Tensor> {
        unary_op(self, |a| a * scalar)
    }

    /// Matrix multiplication: self @ other.
    ///
    /// Supports:
    /// - [M, K] @ [K, N] → [M, N]
    /// - [B, M, K] @ [B, K, N] → [B, M, N] (batched)
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let a = self.contiguous();
        let b = other.contiguous();

        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();

        if a.dtype() != DType::F32 || b.dtype() != DType::F32 {
            return Err(KoreError::UnsupportedDType(a.dtype()));
        }

        match (a_dims.len(), b_dims.len()) {
            (2, 2) => matmul_2d(&a, &b),
            (3, 3) => matmul_batched(&a, &b),
            (2, 1) => matvec(&a, &b),
            _ => Err(KoreError::ShapeMismatch {
                expected: a_dims.to_vec(),
                got: b_dims.to_vec(),
            }),
        }
    }

    /// Clamp all elements to [min, max].
    pub fn clamp(&self, min: f32, max: f32) -> Result<Tensor> {
        unary_op(self, |a| a.clamp(min, max))
    }
}

/// Apply a unary operation element-wise (f32 only for now).
fn unary_op(a: &Tensor, op: impl Fn(f32) -> f32) -> Result<Tensor> {
    if a.dtype() != DType::F32 {
        return Err(KoreError::UnsupportedDType(a.dtype()));
    }
    let a = a.contiguous();
    let a_data = a.as_f32_slice().unwrap();
    let result: Vec<f32> = a_data.iter().map(|&v| op(v)).collect();
    Ok(Tensor::from_f32(&result, a.shape().dims()))
}

/// Apply a binary operation element-wise with broadcasting (f32 only for now).
fn binary_op(a: &Tensor, b: &Tensor, op: impl Fn(f32, f32) -> f32) -> Result<Tensor> {
    if a.dtype() != DType::F32 || b.dtype() != DType::F32 {
        return Err(KoreError::DTypeMismatch {
            expected: a.dtype(),
            got: b.dtype(),
        });
    }

    let out_shape = a.shape().broadcast_with(b.shape()).ok_or_else(|| {
        KoreError::BroadcastError {
            a: a.shape().dims().to_vec(),
            b: b.shape().dims().to_vec(),
        }
    })?;

    let numel = out_shape.numel();
    let mut result = vec![0.0f32; numel];

    // Fast path: same shape, both contiguous
    if a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous() {
        let a_data = a.as_f32_slice().unwrap();
        let b_data = b.as_f32_slice().unwrap();
        for i in 0..numel {
            result[i] = op(a_data[i], b_data[i]);
        }
    } else {
        // General broadcast path
        let a_cont = a.contiguous();
        let b_cont = b.contiguous();
        let a_data = a_cont.as_f32_slice().unwrap();
        let b_data = b_cont.as_f32_slice().unwrap();

        for i in 0..numel {
            let a_idx = broadcast_index(i, &out_shape, a.shape());
            let b_idx = broadcast_index(i, &out_shape, b.shape());
            result[i] = op(a_data[a_idx], b_data[b_idx]);
        }
    }

    Ok(Tensor::from_f32(&result, out_shape.dims()))
}

/// Compute the source index for a broadcasted element.
fn broadcast_index(flat_idx: usize, out_shape: &crate::Shape, src_shape: &crate::Shape) -> usize {
    let out_dims = out_shape.dims();
    let src_dims = src_shape.dims();
    let out_ndim = out_dims.len();
    let src_ndim = src_dims.len();

    let mut remaining = flat_idx;
    let mut src_idx = 0;
    let out_strides = out_shape.contiguous_strides();
    let src_strides = src_shape.contiguous_strides();

    for i in 0..out_ndim {
        let coord = remaining / out_strides[i];
        remaining %= out_strides[i];

        let src_dim_idx = i as isize - (out_ndim as isize - src_ndim as isize);
        if src_dim_idx >= 0 {
            let si = src_dim_idx as usize;
            if src_dims[si] > 1 {
                src_idx += coord * src_strides[si];
            }
            // If src_dims[si] == 1, it's broadcast — coord maps to 0
        }
    }

    src_idx
}

/// 2D matrix multiplication: [M, K] @ [K, N] → [M, N]
fn matmul_2d(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    let m = a_dims[0];
    let k1 = a_dims[1];
    let k2 = b_dims[0];
    let n = b_dims[1];

    if k1 != k2 {
        return Err(KoreError::MatmulDimMismatch { m, k1, k2, n });
    }

    let a_data = a.as_f32_slice().unwrap();
    let b_data = b.as_f32_slice().unwrap();
    let mut c_data = vec![0.0f32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k1 {
                sum += a_data[i * k1 + p] * b_data[p * n + j];
            }
            c_data[i * n + j] = sum;
        }
    }

    Ok(Tensor::from_f32(&c_data, &[m, n]))
}

/// Batched matrix multiplication: [B, M, K] @ [B, K, N] → [B, M, N]
fn matmul_batched(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    let batch = a_dims[0];
    let m = a_dims[1];
    let k1 = a_dims[2];
    let k2 = b_dims[1];
    let n = b_dims[2];

    if a_dims[0] != b_dims[0] {
        return Err(KoreError::ShapeMismatch {
            expected: a_dims.to_vec(),
            got: b_dims.to_vec(),
        });
    }
    if k1 != k2 {
        return Err(KoreError::MatmulDimMismatch { m, k1, k2, n });
    }

    let a_data = a.as_f32_slice().unwrap();
    let b_data = b.as_f32_slice().unwrap();
    let mut c_data = vec![0.0f32; batch * m * n];

    for bi in 0..batch {
        let a_off = bi * m * k1;
        let b_off = bi * k1 * n;
        let c_off = bi * m * n;
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k1 {
                    sum += a_data[a_off + i * k1 + p] * b_data[b_off + p * n + j];
                }
                c_data[c_off + i * n + j] = sum;
            }
        }
    }

    Ok(Tensor::from_f32(&c_data, &[batch, m, n]))
}

/// Matrix-vector multiplication: [M, K] @ [K] → [M]
fn matvec(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    let m = a_dims[0];
    let k1 = a_dims[1];
    let k2 = b_dims[0];

    if k1 != k2 {
        return Err(KoreError::MatmulDimMismatch {
            m,
            k1,
            k2,
            n: 1,
        });
    }

    let a_data = a.as_f32_slice().unwrap();
    let b_data = b.as_f32_slice().unwrap();
    let mut c_data = vec![0.0f32; m];

    for i in 0..m {
        let mut sum = 0.0f32;
        for p in 0..k1 {
            sum += a_data[i * k1 + p] * b_data[p];
        }
        c_data[i] = sum;
    }

    Ok(Tensor::from_f32(&c_data, &[m]))
}

// Operator overloads
impl std::ops::Add for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        self.add(rhs).expect("Add failed")
    }
}

impl std::ops::Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        Tensor::sub(self, rhs).expect("Sub failed")
    }
}

impl std::ops::Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        Tensor::mul(self, rhs).expect("Mul failed")
    }
}

impl std::ops::Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(self).expect("Neg failed")
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn test_add() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_f32(&[4.0, 5.0, 6.0], &[3]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.as_f32_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub() {
        let a = Tensor::from_f32(&[4.0, 5.0, 6.0], &[3]);
        let b = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        let c = a.sub(&b).unwrap();
        assert_eq!(c.as_f32_slice().unwrap(), &[3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_mul() {
        let a = Tensor::from_f32(&[2.0, 3.0], &[2]);
        let b = Tensor::from_f32(&[4.0, 5.0], &[2]);
        let c = a.mul(&b).unwrap();
        assert_eq!(c.as_f32_slice().unwrap(), &[8.0, 15.0]);
    }

    #[test]
    fn test_broadcast_add() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::from_f32(&[10.0, 20.0, 30.0], &[3]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.shape().dims(), &[2, 3]);
        assert_eq!(
            c.as_f32_slice().unwrap(),
            &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
        );
    }

    #[test]
    fn test_scalar_ops() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        let b = a.add_scalar(10.0).unwrap();
        assert_eq!(b.as_f32_slice().unwrap(), &[11.0, 12.0, 13.0]);

        let c = a.mul_scalar(2.0).unwrap();
        assert_eq!(c.as_f32_slice().unwrap(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_unary_ops() {
        let a = Tensor::from_f32(&[-1.0, 0.0, 1.0], &[3]);
        let b = a.abs().unwrap();
        assert_eq!(b.as_f32_slice().unwrap(), &[1.0, 0.0, 1.0]);

        let c = a.neg().unwrap();
        assert_eq!(c.as_f32_slice().unwrap(), &[1.0, 0.0, -1.0]);
    }

    #[test]
    fn test_exp_log() {
        let a = Tensor::from_f32(&[0.0, 1.0], &[2]);
        let b = a.exp().unwrap();
        let data = b.as_f32_slice().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - std::f32::consts::E).abs() < 1e-5);

        let c = b.log().unwrap();
        let data = c.as_f32_slice().unwrap();
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_2d() {
        // [2,3] @ [3,2] → [2,2]
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::from_f32(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape().dims(), &[2, 2]);
        assert_eq!(c.as_f32_slice().unwrap(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_matvec() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_f32(&[1.0, 1.0], &[2]);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape().dims(), &[2]);
        assert_eq!(c.as_f32_slice().unwrap(), &[3.0, 7.0]);
    }

    #[test]
    fn test_matmul_dim_mismatch() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3, 1]);
        assert!(a.matmul(&b).is_err());
    }

    #[test]
    fn test_operator_overloads() {
        let a = Tensor::from_f32(&[1.0, 2.0], &[2]);
        let b = Tensor::from_f32(&[3.0, 4.0], &[2]);
        let c = &a + &b;
        assert_eq!(c.as_f32_slice().unwrap(), &[4.0, 6.0]);

        let d = &a * &b;
        assert_eq!(d.as_f32_slice().unwrap(), &[3.0, 8.0]);

        let e = -&a;
        assert_eq!(e.as_f32_slice().unwrap(), &[-1.0, -2.0]);
    }

    #[test]
    fn test_clamp() {
        let a = Tensor::from_f32(&[-2.0, 0.5, 3.0], &[3]);
        let b = a.clamp(0.0, 1.0).unwrap();
        assert_eq!(b.as_f32_slice().unwrap(), &[0.0, 0.5, 1.0]);
    }
}
