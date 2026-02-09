//! Element-wise arithmetic operations on tensors.
//!
//! When inputs have `requires_grad=true` and gradient tracking is enabled,
//! operations automatically build a computation graph for backward().

use std::sync::Arc;

use rayon::prelude::*;

use crate::autograd::{self, GradNode, GradFn};
use crate::error::KoreError;
use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::Result;

/// Minimum number of elements before we use rayon parallelism.
const PAR_THRESHOLD: usize = 8192;

/// Check if any of the inputs track gradients.
fn any_tracks_grad(tensors: &[&Tensor]) -> bool {
    autograd::is_grad_enabled() && tensors.iter().any(|t| t.tracks_grad())
}

/// Collect GradNode inputs from tensors that track gradients.
fn grad_inputs(tensors: &[&Tensor]) -> Vec<Arc<GradNode>> {
    tensors.iter().filter_map(|t| t.grad_node().cloned()).collect()
}

/// Attach a GradNode to a result tensor if gradient tracking is active.
fn with_grad(result: Tensor, grad_fn: Box<dyn GradFn>, inputs: Vec<Arc<GradNode>>) -> Tensor {
    let node = GradNode::with_grad_fn(grad_fn, inputs);
    result.with_grad_node(node)
}

impl Tensor {
    /// Element-wise addition: self + other.
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.is_cuda() {
            return crate::ops::cuda_ops::cuda_binary_op(self, other, "add_f32");
        }
        let result = binary_op(self, other, |a, b| a + b)?;
        if any_tracks_grad(&[self, other]) {
            Ok(with_grad(result, Box::new(autograd::AddBackward), grad_inputs(&[self, other])))
        } else {
            Ok(result)
        }
    }

    /// Element-wise subtraction: self - other.
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.is_cuda() {
            return crate::ops::cuda_ops::cuda_binary_op(self, other, "sub_f32");
        }
        let result = binary_op(self, other, |a, b| a - b)?;
        if any_tracks_grad(&[self, other]) {
            Ok(with_grad(result, Box::new(autograd::SubBackward), grad_inputs(&[self, other])))
        } else {
            Ok(result)
        }
    }

    /// Element-wise multiplication: self * other.
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.is_cuda() {
            return crate::ops::cuda_ops::cuda_binary_op(self, other, "mul_f32");
        }
        let result = binary_op(self, other, |a, b| a * b)?;
        if any_tracks_grad(&[self, other]) {
            Ok(with_grad(result, Box::new(autograd::MulBackward {
                lhs: self.clone(), rhs: other.clone(),
            }), grad_inputs(&[self, other])))
        } else {
            Ok(result)
        }
    }

    /// Element-wise division: self / other.
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.is_cuda() {
            return crate::ops::cuda_ops::cuda_binary_op(self, other, "div_f32");
        }
        let result = binary_op(self, other, |a, b| a / b)?;
        if any_tracks_grad(&[self, other]) {
            Ok(with_grad(result, Box::new(autograd::DivBackward {
                lhs: self.clone(), rhs: other.clone(),
            }), grad_inputs(&[self, other])))
        } else {
            Ok(result)
        }
    }

    /// Element-wise negation: -self.
    pub fn neg(&self) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.is_cuda() {
            return crate::ops::cuda_ops::cuda_unary_op(self, "neg_f32");
        }
        let result = unary_op(self, |a| -a)?;
        if any_tracks_grad(&[self]) {
            Ok(with_grad(result, Box::new(autograd::NegBackward), grad_inputs(&[self])))
        } else {
            Ok(result)
        }
    }

    /// Element-wise absolute value.
    pub fn abs(&self) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.is_cuda() {
            return crate::ops::cuda_ops::cuda_unary_op(self, "abs_f32");
        }
        let result = unary_op(self, |a| a.abs())?;
        if any_tracks_grad(&[self]) {
            Ok(with_grad(result, Box::new(autograd::AbsBackward {
                input: self.clone(),
            }), grad_inputs(&[self])))
        } else {
            Ok(result)
        }
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.is_cuda() {
            return crate::ops::cuda_ops::cuda_unary_op(self, "sqrt_f32");
        }
        let result = unary_op(self, |a| a.sqrt())?;
        if any_tracks_grad(&[self]) {
            let bw = Box::new(autograd::SqrtBackward { output: result.clone() });
            Ok(with_grad(result, bw, grad_inputs(&[self])))
        } else {
            Ok(result)
        }
    }

    /// Element-wise reciprocal: 1/self.
    pub fn reciprocal(&self) -> Result<Tensor> {
        unary_op(self, |a| 1.0 / a)
    }

    /// Element-wise exponential: e^self.
    pub fn exp(&self) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.is_cuda() {
            return crate::ops::cuda_ops::cuda_unary_op(self, "exp_f32");
        }
        let result = unary_op(self, |a| a.exp())?;
        if any_tracks_grad(&[self]) {
            let bw = Box::new(autograd::ExpBackward { output: result.clone() });
            Ok(with_grad(result, bw, grad_inputs(&[self])))
        } else {
            Ok(result)
        }
    }

    /// Element-wise natural logarithm.
    pub fn log(&self) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.is_cuda() {
            return crate::ops::cuda_ops::cuda_unary_op(self, "log_f32");
        }
        let result = unary_op(self, |a| a.ln())?;
        if any_tracks_grad(&[self]) {
            Ok(with_grad(result, Box::new(autograd::LogBackward {
                input: self.clone(),
            }), grad_inputs(&[self])))
        } else {
            Ok(result)
        }
    }

    /// Element-wise power: self^exponent.
    pub fn pow_scalar(&self, exponent: f32) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.is_cuda() {
            return crate::ops::cuda_ops::cuda_scalar_op(self, exponent, "pow_scalar_f32");
        }
        let result = unary_op(self, |a| a.powf(exponent))?;
        if any_tracks_grad(&[self]) {
            Ok(with_grad(result, Box::new(autograd::PowScalarBackward {
                input: self.clone(), exponent,
            }), grad_inputs(&[self])))
        } else {
            Ok(result)
        }
    }

    /// Scalar addition: self + scalar.
    pub fn add_scalar(&self, scalar: f32) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.is_cuda() {
            return crate::ops::cuda_ops::cuda_scalar_op(self, scalar, "add_scalar_f32");
        }
        let result = unary_op(self, |a| a + scalar)?;
        if any_tracks_grad(&[self]) {
            Ok(with_grad(result, Box::new(autograd::AddScalarBackward), grad_inputs(&[self])))
        } else {
            Ok(result)
        }
    }

    /// Scalar multiplication: self * scalar.
    pub fn mul_scalar(&self, scalar: f32) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.is_cuda() {
            return crate::ops::cuda_ops::cuda_scalar_op(self, scalar, "mul_scalar_f32");
        }
        let result = unary_op(self, |a| a * scalar)?;
        if any_tracks_grad(&[self]) {
            Ok(with_grad(result, Box::new(autograd::MulScalarBackward { scalar }), grad_inputs(&[self])))
        } else {
            Ok(result)
        }
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

        let result = match (a_dims.len(), b_dims.len()) {
            (2, 2) => {
                #[cfg(feature = "cuda")]
                if a.is_cuda() {
                    return crate::ops::cuda_ops::cuda_matmul_2d(&a, &b);
                }
                matmul_2d(&a, &b)?
            }
            (3, 3) => matmul_batched(&a, &b)?,
            (2, 1) => matvec(&a, &b)?,
            _ => return Err(KoreError::ShapeMismatch {
                expected: a_dims.to_vec(),
                got: b_dims.to_vec(),
            }),
        };

        if any_tracks_grad(&[self, other]) {
            Ok(with_grad(result, Box::new(autograd::MatmulBackward {
                lhs: self.clone(), rhs: other.clone(),
            }), grad_inputs(&[self, other])))
        } else {
            Ok(result)
        }
    }

    /// Clamp all elements to [min, max].
    pub fn clamp(&self, min: f32, max: f32) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.is_cuda() {
            return crate::ops::cuda_ops::cuda_clamp(self, min, max);
        }
        let result = unary_op(self, |a| a.clamp(min, max))?;
        if any_tracks_grad(&[self]) {
            Ok(with_grad(result, Box::new(autograd::ClampBackward {
                input: self.clone(), min, max,
            }), grad_inputs(&[self])))
        } else {
            Ok(result)
        }
    }
}

// =========================================================================
// In-place operations (no allocation, for optimizer parameter updates)
// =========================================================================

impl Tensor {
    /// In-place addition: self += other.
    pub fn add_(&mut self, other: &Tensor) -> Result<()> {
        inplace_binary(self, other, |a, b| a + b)
    }

    /// In-place subtraction: self -= other.
    pub fn sub_(&mut self, other: &Tensor) -> Result<()> {
        inplace_binary(self, other, |a, b| a - b)
    }

    /// In-place multiplication: self *= other.
    pub fn mul_(&mut self, other: &Tensor) -> Result<()> {
        inplace_binary(self, other, |a, b| a * b)
    }

    /// In-place division: self /= other.
    pub fn div_(&mut self, other: &Tensor) -> Result<()> {
        inplace_binary(self, other, |a, b| a / b)
    }

    /// In-place scalar addition: self += scalar.
    pub fn add_scalar_(&mut self, scalar: f32) -> Result<()> {
        inplace_unary(self, |a| a + scalar)
    }

    /// In-place scalar multiplication: self *= scalar.
    pub fn mul_scalar_(&mut self, scalar: f32) -> Result<()> {
        inplace_unary(self, |a| a * scalar)
    }

    /// In-place clamp: self = clamp(self, min, max).
    pub fn clamp_(&mut self, min: f32, max: f32) -> Result<()> {
        inplace_unary(self, |a| a.clamp(min, max))
    }

    /// In-place ReLU: self = max(0, self).
    pub fn relu_(&mut self) -> Result<()> {
        inplace_unary(self, |a| a.max(0.0))
    }

    /// In-place zero: self = 0.
    pub fn zero_(&mut self) -> Result<()> {
        inplace_unary(self, |_| 0.0)
    }

    /// In-place fill: self = value.
    pub fn fill_(&mut self, value: f32) -> Result<()> {
        inplace_unary(self, |_| value)
    }

    /// In-place fused multiply-add: self = self + other * scalar.
    /// Common in optimizers: param += -lr * grad.
    pub fn add_scaled_(&mut self, other: &Tensor, scalar: f32) -> Result<()> {
        inplace_binary(self, other, |a, b| a + b * scalar)
    }
}

/// Apply an in-place unary operation (f32 only, contiguous required).
fn inplace_unary(a: &mut Tensor, op: impl Fn(f32) -> f32 + Sync) -> Result<()> {
    if a.dtype() != DType::F32 {
        return Err(KoreError::UnsupportedDType(a.dtype()));
    }
    if !a.is_contiguous() {
        return Err(KoreError::StorageError(
            "In-place op requires contiguous tensor".into(),
        ));
    }
    let data = a.as_f32_slice_mut().ok_or_else(|| {
        KoreError::StorageError("Failed to get mutable f32 slice".into())
    })?;
    if data.len() >= PAR_THRESHOLD {
        data.par_iter_mut().for_each(|v| *v = op(*v));
    } else {
        for v in data.iter_mut() {
            *v = op(*v);
        }
    }
    Ok(())
}

/// Apply an in-place binary operation (f32 only, same shape required).
fn inplace_binary(a: &mut Tensor, b: &Tensor, op: impl Fn(f32, f32) -> f32 + Sync) -> Result<()> {
    if a.dtype() != DType::F32 || b.dtype() != DType::F32 {
        return Err(KoreError::DTypeMismatch {
            expected: a.dtype(),
            got: b.dtype(),
        });
    }
    if a.shape().dims() != b.shape().dims() {
        return Err(KoreError::ShapeMismatch {
            expected: a.shape().dims().to_vec(),
            got: b.shape().dims().to_vec(),
        });
    }
    if !a.is_contiguous() {
        return Err(KoreError::StorageError(
            "In-place op requires contiguous tensor".into(),
        ));
    }
    let b = b.contiguous();
    let b_data = b.as_f32_slice().unwrap();
    let a_data = a.as_f32_slice_mut().ok_or_else(|| {
        KoreError::StorageError("Failed to get mutable f32 slice".into())
    })?;
    if a_data.len() >= PAR_THRESHOLD {
        a_data.par_iter_mut().zip(b_data.par_iter()).for_each(|(av, &bv)| {
            *av = op(*av, bv);
        });
    } else {
        for (av, &bv) in a_data.iter_mut().zip(b_data.iter()) {
            *av = op(*av, bv);
        }
    }
    Ok(())
}

/// Apply a unary operation element-wise (f32 only for now).
fn unary_op(a: &Tensor, op: impl Fn(f32) -> f32 + Sync) -> Result<Tensor> {
    if a.dtype() != DType::F32 {
        return Err(KoreError::UnsupportedDType(a.dtype()));
    }
    let a = a.contiguous();
    let a_data = a.as_f32_slice().unwrap();
    let result: Vec<f32> = if a_data.len() >= PAR_THRESHOLD {
        a_data.par_iter().map(|&v| op(v)).collect()
    } else {
        a_data.iter().map(|&v| op(v)).collect()
    };
    Ok(Tensor::from_f32(&result, a.shape().dims()))
}

/// Apply a binary operation element-wise with broadcasting (f32 only for now).
fn binary_op(a: &Tensor, b: &Tensor, op: impl Fn(f32, f32) -> f32 + Sync) -> Result<Tensor> {
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
        if numel >= PAR_THRESHOLD {
            result.par_iter_mut().enumerate().for_each(|(i, r)| {
                *r = op(a_data[i], b_data[i]);
            });
        } else {
            for i in 0..numel {
                result[i] = op(a_data[i], b_data[i]);
            }
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

/// Tile size for cache-friendly blocking.
/// 64×64 tiles fit comfortably in L1 cache (~32KB for f32).
const TILE_M: usize = 64;
const TILE_N: usize = 64;
const TILE_K: usize = 64;

/// 2D matrix multiplication: [M, K] @ [K, N] → [M, N]
///
/// Uses tiled algorithm with AVX2+FMA SIMD inner loop when available,
/// falling back to tiled scalar. ~5-20× faster than naive triple-loop.
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

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { tiled_matmul_avx2(a_data, b_data, &mut c_data, m, n, k1) };
            return Ok(Tensor::from_f32(&c_data, &[m, n]));
        }
    }

    tiled_matmul_scalar(a_data, b_data, &mut c_data, m, n, k1);
    Ok(Tensor::from_f32(&c_data, &[m, n]))
}

/// Scalar tiled matmul (fallback for non-x86 or missing AVX2).
fn tiled_matmul_scalar(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i0 in (0..m).step_by(TILE_M) {
        let i_end = (i0 + TILE_M).min(m);
        for j0 in (0..n).step_by(TILE_N) {
            let j_end = (j0 + TILE_N).min(n);
            for p0 in (0..k).step_by(TILE_K) {
                let p_end = (p0 + TILE_K).min(k);
                for i in i0..i_end {
                    for p in p0..p_end {
                        let a_val = a[i * k + p];
                        for j in j0..j_end {
                            c[i * n + j] += a_val * b[p * n + j];
                        }
                    }
                }
            }
        }
    }
}

/// AVX2+FMA tiled matmul — processes 8 floats at a time in the inner loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tiled_matmul_avx2(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    use std::arch::x86_64::*;

    for i0 in (0..m).step_by(TILE_M) {
        let i_end = (i0 + TILE_M).min(m);
        for j0 in (0..n).step_by(TILE_N) {
            let j_end = (j0 + TILE_N).min(n);
            for p0 in (0..k).step_by(TILE_K) {
                let p_end = (p0 + TILE_K).min(k);
                for i in i0..i_end {
                    for p in p0..p_end {
                        let a_val = _mm256_set1_ps(a[i * k + p]);
                        let mut j = j0;
                        while j + 8 <= j_end {
                            let c_ptr = c.as_mut_ptr().add(i * n + j);
                            let b_ptr = b.as_ptr().add(p * n + j);
                            let c_vec = _mm256_loadu_ps(c_ptr);
                            let b_vec = _mm256_loadu_ps(b_ptr);
                            let result = _mm256_fmadd_ps(a_val, b_vec, c_vec);
                            _mm256_storeu_ps(c_ptr, result);
                            j += 8;
                        }
                        // Scalar tail
                        while j < j_end {
                            c[i * n + j] += a[i * k + p] * b[p * n + j];
                            j += 1;
                        }
                    }
                }
            }
        }
    }
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
        let a_slice = &a_data[a_off..a_off + m * k1];
        let b_slice = &b_data[b_off..b_off + k1 * n];
        let c_slice = &mut c_data[c_off..c_off + m * n];

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe { tiled_matmul_avx2(a_slice, b_slice, c_slice, m, n, k1) };
                continue;
            }
        }
        tiled_matmul_scalar(a_slice, b_slice, c_slice, m, n, k1);
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

    // =========================================================================
    // Autograd integration tests
    // =========================================================================

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < tol, "elem {} differs: {} vs {} (tol={})", i, x, y, tol);
        }
    }

    #[test]
    fn test_autograd_add_backward() {
        // c = a + b, dc/da = 1, dc/db = 1
        let mut a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        let mut b = Tensor::from_f32(&[4.0, 5.0, 6.0], &[3]);
        a.set_requires_grad(true);
        b.set_requires_grad(true);

        let c = a.add(&b).unwrap();
        let loss = c.sum().unwrap();
        loss.backward().unwrap();

        assert_close(a.grad().unwrap().as_f32_slice().unwrap(), &[1.0, 1.0, 1.0], 1e-6);
        assert_close(b.grad().unwrap().as_f32_slice().unwrap(), &[1.0, 1.0, 1.0], 1e-6);
    }

    #[test]
    fn test_autograd_mul_backward() {
        // c = a * b, dc/da = b, dc/db = a
        let mut a = Tensor::from_f32(&[2.0, 3.0], &[2]);
        let mut b = Tensor::from_f32(&[4.0, 5.0], &[2]);
        a.set_requires_grad(true);
        b.set_requires_grad(true);

        let c = a.mul(&b).unwrap();
        let loss = c.sum().unwrap();
        loss.backward().unwrap();

        assert_close(a.grad().unwrap().as_f32_slice().unwrap(), &[4.0, 5.0], 1e-6);
        assert_close(b.grad().unwrap().as_f32_slice().unwrap(), &[2.0, 3.0], 1e-6);
    }

    #[test]
    fn test_autograd_matmul_backward() {
        // C = A @ B, dA = dC @ B^T, dB = A^T @ dC
        let mut a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let mut b = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        a.set_requires_grad(true);
        b.set_requires_grad(true);

        let c = a.matmul(&b).unwrap();
        let loss = c.sum().unwrap();
        loss.backward().unwrap();

        // dC = ones(2,2), dA = dC @ B^T, dB = A^T @ dC
        // B^T = [[5,7],[6,8]], dA = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
        let ga = a.grad().unwrap();
        assert_eq!(ga.shape().dims(), &[2, 2]);
        assert_close(ga.as_f32_slice().unwrap(), &[11.0, 15.0, 11.0, 15.0], 1e-4);

        // A^T = [[1,3],[2,4]], dB = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
        let gb = b.grad().unwrap();
        assert_eq!(gb.shape().dims(), &[2, 2]);
        assert_close(gb.as_f32_slice().unwrap(), &[4.0, 4.0, 6.0, 6.0], 1e-4);
    }

    #[test]
    fn test_autograd_chain_rule() {
        // loss = sum((a * b) + b), a=[2,3], b=[4,5]
        // d(loss)/da = b = [4, 5]
        // d(loss)/db = a + 1 = [3, 4]
        let mut a = Tensor::from_f32(&[2.0, 3.0], &[2]);
        let mut b = Tensor::from_f32(&[4.0, 5.0], &[2]);
        a.set_requires_grad(true);
        b.set_requires_grad(true);

        let ab = a.mul(&b).unwrap();
        let c = ab.add(&b).unwrap();
        let loss = c.sum().unwrap();
        loss.backward().unwrap();

        assert_close(a.grad().unwrap().as_f32_slice().unwrap(), &[4.0, 5.0], 1e-6);
        assert_close(b.grad().unwrap().as_f32_slice().unwrap(), &[3.0, 4.0], 1e-6);
    }

    #[test]
    fn test_autograd_no_grad_scope() {
        use crate::autograd::NoGradGuard;

        let mut a = Tensor::from_f32(&[1.0, 2.0], &[2]);
        a.set_requires_grad(true);

        // Under no_grad, ops should NOT build graph
        let result = {
            let _guard = NoGradGuard::new();
            a.mul_scalar(2.0).unwrap()
        };
        assert!(result.grad_node().is_none());
    }

    #[test]
    fn test_autograd_exp_backward() {
        // loss = sum(exp(a)), d(loss)/da = exp(a)
        let mut a = Tensor::from_f32(&[0.0, 1.0], &[2]);
        a.set_requires_grad(true);

        let e = a.exp().unwrap();
        let loss = e.sum().unwrap();
        loss.backward().unwrap();

        let ga = a.grad().unwrap();
        let expected = [0.0f32.exp(), 1.0f32.exp()]; // exp(0)=1, exp(1)=e
        assert_close(ga.as_f32_slice().unwrap(), &expected, 1e-5);
    }

    #[test]
    fn test_autograd_neg_backward() {
        // loss = sum(-a), d(loss)/da = -1
        let mut a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        a.set_requires_grad(true);

        let n = a.neg().unwrap();
        let loss = n.sum().unwrap();
        loss.backward().unwrap();

        assert_close(a.grad().unwrap().as_f32_slice().unwrap(), &[-1.0, -1.0, -1.0], 1e-6);
    }

    #[test]
    fn test_autograd_sub_backward() {
        // loss = sum(a - b), d/da = 1, d/db = -1
        let mut a = Tensor::from_f32(&[5.0, 6.0], &[2]);
        let mut b = Tensor::from_f32(&[1.0, 2.0], &[2]);
        a.set_requires_grad(true);
        b.set_requires_grad(true);

        let c = a.sub(&b).unwrap();
        let loss = c.sum().unwrap();
        loss.backward().unwrap();

        assert_close(a.grad().unwrap().as_f32_slice().unwrap(), &[1.0, 1.0], 1e-6);
        assert_close(b.grad().unwrap().as_f32_slice().unwrap(), &[-1.0, -1.0], 1e-6);
    }

    #[test]
    fn test_autograd_mul_scalar_backward() {
        // loss = sum(a * 3.0), d/da = 3.0
        let mut a = Tensor::from_f32(&[1.0, 2.0], &[2]);
        a.set_requires_grad(true);

        let c = a.mul_scalar(3.0).unwrap();
        let loss = c.sum().unwrap();
        loss.backward().unwrap();

        assert_close(a.grad().unwrap().as_f32_slice().unwrap(), &[3.0, 3.0], 1e-6);
    }

    // =========================================================================
    // In-place operation tests
    // =========================================================================

    #[test]
    fn test_inplace_add() {
        let mut a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_f32(&[4.0, 5.0, 6.0], &[3]);
        a.add_(&b).unwrap();
        assert_eq!(a.as_f32_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_inplace_sub() {
        let mut a = Tensor::from_f32(&[4.0, 5.0, 6.0], &[3]);
        let b = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        a.sub_(&b).unwrap();
        assert_eq!(a.as_f32_slice().unwrap(), &[3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_inplace_mul() {
        let mut a = Tensor::from_f32(&[2.0, 3.0], &[2]);
        let b = Tensor::from_f32(&[4.0, 5.0], &[2]);
        a.mul_(&b).unwrap();
        assert_eq!(a.as_f32_slice().unwrap(), &[8.0, 15.0]);
    }

    #[test]
    fn test_inplace_scalar_mul() {
        let mut a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        a.mul_scalar_(2.0).unwrap();
        assert_eq!(a.as_f32_slice().unwrap(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_inplace_relu() {
        let mut a = Tensor::from_f32(&[-1.0, 0.0, 1.0, -2.0, 3.0], &[5]);
        a.relu_().unwrap();
        assert_eq!(a.as_f32_slice().unwrap(), &[0.0, 0.0, 1.0, 0.0, 3.0]);
    }

    #[test]
    fn test_inplace_clamp() {
        let mut a = Tensor::from_f32(&[-2.0, 0.5, 3.0], &[3]);
        a.clamp_(0.0, 1.0).unwrap();
        assert_eq!(a.as_f32_slice().unwrap(), &[0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_inplace_zero() {
        let mut a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        a.zero_().unwrap();
        assert_eq!(a.as_f32_slice().unwrap(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_inplace_add_scaled() {
        // param += -lr * grad  →  add_scaled_(&grad, -lr)
        let mut param = Tensor::from_f32(&[1.0, 2.0], &[2]);
        let grad = Tensor::from_f32(&[0.5, 1.0], &[2]);
        param.add_scaled_(&grad, -0.1).unwrap();
        assert_close(param.as_f32_slice().unwrap(), &[0.95, 1.9], 1e-6);
    }

    #[test]
    fn test_inplace_shape_mismatch() {
        let mut a = Tensor::from_f32(&[1.0, 2.0], &[2]);
        let b = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3]);
        assert!(a.add_(&b).is_err());
    }
}
