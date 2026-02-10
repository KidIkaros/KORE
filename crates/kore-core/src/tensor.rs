use std::fmt;
use std::sync::Arc;

use smallvec::SmallVec;

use crate::autograd::GradNode;
use crate::dtype::DType;
use crate::device::Device;
use crate::error::KoreError;
use crate::shape::Shape;
use crate::storage::Storage;
use crate::Result;

/// A multi-dimensional array — the fundamental data structure in Kore.
///
/// Tensors support:
/// - Multiple dtypes including native Ternary and Quaternary
/// - Zero-copy views (reshape, transpose, slice share storage)
/// - CPU and CUDA devices
/// - Automatic gradient tracking (when used with kore-autograd)
///
/// # Examples
///
/// ```
/// use kore_core::Tensor;
///
/// // Create from f32 data
/// let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
/// assert_eq!(t.shape().dims(), &[2, 2]);
/// assert_eq!(t.numel(), 4);
///
/// // Reshape (zero-copy view)
/// let flat = t.reshape(&[4]).unwrap();
/// assert_eq!(flat.shape().dims(), &[4]);
/// ```
#[derive(Clone)]
pub struct Tensor {
    storage: Storage,
    shape: Shape,
    strides: SmallVec<[usize; 4]>,
    offset: usize,
    requires_grad: bool,
    grad_node: Option<Arc<GradNode>>,
}

impl Tensor {
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Create a tensor from f32 data with the given shape.
    pub fn from_f32(data: &[f32], shape: &[usize]) -> Self {
        let s = Shape::new(shape);
        assert_eq!(
            s.numel(),
            data.len(),
            "Shape {:?} requires {} elements, got {}",
            shape,
            s.numel(),
            data.len()
        );
        let strides = s.contiguous_strides();
        Self {
            storage: Storage::from_f32(data),
            shape: s,
            strides,
            offset: 0,
            requires_grad: false,
            grad_node: None,
        }
    }

    /// Create a tensor from f64 data with the given shape.
    pub fn from_f64(data: &[f64], shape: &[usize]) -> Self {
        let s = Shape::new(shape);
        assert_eq!(s.numel(), data.len());
        let strides = s.contiguous_strides();
        Self {
            storage: Storage::from_f64(data),
            shape: s,
            strides,
            offset: 0,
            requires_grad: false,
            grad_node: None,
        }
    }

    /// Create a tensor of zeros with the given shape and dtype.
    pub fn zeros(shape: &[usize], dtype: DType) -> Self {
        let s = Shape::new(shape);
        let strides = s.contiguous_strides();
        Self {
            storage: Storage::zeros(dtype, s.numel()),
            shape: s,
            strides,
            offset: 0,
            requires_grad: false,
            grad_node: None,
        }
    }

    /// Create a tensor of ones (f32).
    pub fn ones(shape: &[usize]) -> Self {
        let s = Shape::new(shape);
        let numel = s.numel();
        let data: Vec<f32> = vec![1.0; numel];
        Self::from_f32(&data, shape)
    }

    /// Create a tensor with random values from standard normal distribution N(0,1).
    pub fn randn(shape: &[usize]) -> Self {
        use rand::Rng;
        let s = Shape::new(shape);
        let numel = s.numel();
        let mut rng = rand::thread_rng();
        // Box-Muller transform for normal distribution
        let data: Vec<f32> = (0..numel)
            .map(|_| {
                let u1: f32 = rng.gen_range(1e-7f32..1.0f32);
                let u2: f32 = rng.gen_range(0.0f32..std::f32::consts::TAU);
                (-2.0 * u1.ln()).sqrt() * u2.cos()
            })
            .collect();
        Self::from_f32(&data, shape)
    }

    /// Create a tensor with random values uniformly distributed in [low, high).
    pub fn rand_uniform(shape: &[usize], low: f32, high: f32) -> Self {
        use rand::Rng;
        let s = Shape::new(shape);
        let numel = s.numel();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..numel).map(|_| rng.gen_range(low..high)).collect();
        Self::from_f32(&data, shape)
    }

    /// Create a 1-D tensor with values from `start` to `end` (exclusive).
    ///
    /// # Panics
    /// Panics if `step` is zero or if `step` direction doesn't match `start`→`end`.
    pub fn arange(start: f32, end: f32, step: f32) -> Self {
        assert!(step != 0.0, "arange: step must be non-zero");
        assert!((end - start) * step > 0.0 || (end - start).abs() < f32::EPSILON,
            "arange: step direction ({}) does not match start ({}) → end ({})", step, start, end);
        let mut data = Vec::new();
        let mut v = start;
        if step > 0.0 {
            while v < end {
                data.push(v);
                v += step;
            }
        } else {
            while v > end {
                data.push(v);
                v += step;
            }
        }
        let len = data.len();
        Self::from_f32(&data, &[len])
    }

    /// Create a scalar tensor from a single f32 value.
    pub fn scalar(value: f32) -> Self {
        Self {
            storage: Storage::from_f32(&[value]),
            shape: Shape::scalar(),
            strides: SmallVec::new(),
            offset: 0,
            requires_grad: false,
            grad_node: None,
        }
    }

    /// Create a tensor from pre-built Storage and shape.
    pub fn from_storage(storage: Storage, shape: &[usize]) -> Self {
        let s = Shape::new(shape);
        let strides = s.contiguous_strides();
        Self {
            storage,
            shape: s,
            strides,
            offset: 0,
            requires_grad: false,
            grad_node: None,
        }
    }

    /// Get a reference to the underlying storage (for CUDA dispatch).
    pub fn storage_ref(&self) -> &Storage {
        &self.storage
    }

    // =========================================================================
    // Properties
    // =========================================================================

    /// Shape of the tensor.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Data type.
    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    /// Device.
    pub fn device(&self) -> Device {
        self.storage.device()
    }

    /// Strides (in elements, not bytes).
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Whether this tensor requires gradient computation.
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Set whether this tensor requires gradient computation.
    /// When enabled, creates a leaf GradNode for this tensor.
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
        if requires_grad && self.grad_node.is_none() {
            self.grad_node = Some(GradNode::leaf());
        }
        if !requires_grad {
            self.grad_node = None;
        }
    }

    // =========================================================================
    // Autograd
    // =========================================================================

    /// Get the GradNode for this tensor (if tracking gradients).
    pub fn grad_node(&self) -> Option<&Arc<GradNode>> {
        self.grad_node.as_ref()
    }

    /// Attach a GradNode to this tensor (used by op dispatch).
    /// Also sets requires_grad=true since this tensor is part of the computation graph.
    pub fn with_grad_node(mut self, node: Arc<GradNode>) -> Self {
        self.grad_node = Some(node);
        self.requires_grad = true;
        self
    }

    /// Get the accumulated gradient for this tensor.
    pub fn grad(&self) -> Option<Tensor> {
        self.grad_node.as_ref().and_then(|n| n.get_grad())
    }

    /// Clear accumulated gradients.
    pub fn zero_grad(&self) {
        if let Some(ref node) = self.grad_node {
            node.zero_grad();
        }
    }

    /// Run backward pass from this tensor (must be scalar).
    pub fn backward(&self) -> Result<()> {
        if self.numel() != 1 {
            return Err(KoreError::ShapeMismatch {
                expected: vec![1],
                got: self.shape().dims().to_vec(),
            });
        }
        let node = self.grad_node.as_ref().ok_or_else(|| {
            KoreError::StorageError("backward() called on tensor without grad tracking".into())
        })?;
        crate::autograd::backward(node, Tensor::scalar(1.0));
        Ok(())
    }

    /// Check if any input requires grad (used by op dispatch to decide whether to build graph).
    pub fn tracks_grad(&self) -> bool {
        self.requires_grad && self.grad_node.is_some() && crate::autograd::is_grad_enabled()
    }

    /// Whether this tensor is contiguous in memory (row-major).
    pub fn is_contiguous(&self) -> bool {
        self.strides == self.shape.contiguous_strides() && self.offset == 0
    }

    // =========================================================================
    // Data access
    // =========================================================================

    /// Get the underlying f32 data as a slice (contiguous tensors only).
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        if !self.is_contiguous() {
            return None;
        }
        self.storage.as_f32_slice()
    }

    /// Get a mutable f32 slice (contiguous, copy-on-write).
    pub fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        if !self.is_contiguous() {
            return None;
        }
        self.storage.as_f32_slice_mut()
    }

    /// Get a single f32 element by flat index.
    pub fn get_f32(&self, flat_index: usize) -> Option<f32> {
        let slice = self.storage.as_f32_slice()?;
        let physical = self.flat_to_physical(flat_index)?;
        slice.get(physical).copied()
    }

    /// Convert multi-dimensional index to physical storage index.
    fn flat_to_physical(&self, flat_index: usize) -> Option<usize> {
        if self.shape.is_scalar() {
            return if flat_index == 0 {
                Some(self.offset)
            } else {
                None
            };
        }

        if flat_index >= self.numel() {
            return None;
        }

        // Convert flat index to multi-dimensional index
        let mut remaining = flat_index;
        let mut physical = self.offset;
        let contiguous_strides = self.shape.contiguous_strides();

        for (i, &cs) in contiguous_strides.iter().enumerate() {
            let idx = remaining / cs;
            remaining %= cs;
            physical += idx * self.strides[i];
        }

        Some(physical)
    }

    // =========================================================================
    // Shape operations (zero-copy views)
    // =========================================================================

    /// Reshape the tensor (zero-copy if contiguous).
    pub fn reshape(&self, new_shape: &[isize]) -> Result<Tensor> {
        let resolved = self.shape.resolve_reshape(new_shape).ok_or_else(|| {
            KoreError::InvalidReshape {
                numel: self.numel(),
                shape: new_shape.iter().map(|&d| d as usize).collect(),
            }
        })?;

        if !self.is_contiguous() {
            return Err(KoreError::StorageError(
                "Cannot reshape non-contiguous tensor (call .contiguous() first)".into(),
            ));
        }

        let strides = resolved.contiguous_strides();
        Ok(Tensor {
            storage: self.storage.clone(), // Arc clone — shared data
            shape: resolved,
            strides,
            offset: self.offset,
            requires_grad: self.requires_grad,
            grad_node: self.grad_node.clone(),
        })
    }

    /// Transpose the last two dimensions (zero-copy view).
    pub fn transpose(&self) -> Result<Tensor> {
        let new_shape = self.shape.transpose().ok_or_else(|| {
            KoreError::InvalidAxis {
                axis: 0,
                ndim: self.ndim(),
            }
        })?;

        let ndim = self.ndim();
        let mut new_strides = self.strides.clone();
        new_strides.swap(ndim - 2, ndim - 1);

        Ok(Tensor {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            requires_grad: self.requires_grad,
            grad_node: self.grad_node.clone(),
        })
    }

    // =========================================================================
    // Device transfer
    // =========================================================================

    /// Whether this tensor is on CPU.
    pub fn is_cpu(&self) -> bool {
        self.storage.is_cpu()
    }

    /// Whether this tensor is on a CUDA device.
    pub fn is_cuda(&self) -> bool {
        self.storage.is_cuda()
    }

    /// Move tensor to the specified device. No-op if already there.
    #[cfg(feature = "cuda")]
    pub fn to(&self, device: Device) -> Result<Tensor> {
        match device {
            Device::Cpu => {
                if self.device().is_cpu() {
                    return Ok(self.clone());
                }
                let cpu_storage = self.storage.to_cpu()?;
                Ok(Tensor {
                    storage: cpu_storage,
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                    offset: self.offset,
                    requires_grad: self.requires_grad,
                    grad_node: self.grad_node.clone(),
                })
            }
            Device::Cuda(idx) => {
                if let Device::Cuda(cur) = self.device() {
                    if cur == idx {
                        return Ok(self.clone());
                    }
                }
                // Must be contiguous before transfer
                let cont = self.contiguous();
                let cuda_storage = cont.storage.to_cuda(idx)?;
                Ok(Tensor {
                    storage: cuda_storage,
                    shape: cont.shape.clone(),
                    strides: cont.strides.clone(),
                    offset: 0,
                    requires_grad: cont.requires_grad,
                    grad_node: cont.grad_node.clone(),
                })
            }
        }
    }

    /// Move tensor to CUDA device (convenience for `.to(Device::Cuda(idx))`).
    #[cfg(feature = "cuda")]
    pub fn cuda(&self, device_idx: usize) -> Result<Tensor> {
        self.to(Device::Cuda(device_idx))
    }

    /// Move tensor to CPU (convenience for `.to(Device::Cpu)`).
    #[cfg(feature = "cuda")]
    pub fn cpu(&self) -> Result<Tensor> {
        self.to(Device::Cpu)
    }

    /// Return a contiguous copy of this tensor if it isn't already contiguous.
    pub fn contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            return self.clone();
        }

        // For now, only support F32 contiguous copy
        if self.dtype() == DType::F32 {
            let numel = self.numel();
            let mut data = vec![0.0f32; numel];
            for i in 0..numel {
                data[i] = self.get_f32(i)
                    .expect("contiguous: index out of bounds during copy");
            }
            let mut t = Tensor::from_f32(&data, self.shape.dims());
            t.requires_grad = self.requires_grad;
            t.grad_node = self.grad_node.clone();
            t
        } else {
            // Fallback: clone (already contiguous for fresh tensors)
            self.clone()
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(shape={}, dtype={}, device={}, contiguous={}, requires_grad={})",
            self.shape,
            self.dtype(),
            self.device(),
            self.is_contiguous(),
            self.requires_grad,
        )
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(data) = self.as_f32_slice() {
            if self.numel() <= 20 {
                write!(f, "tensor({:?}, shape={})", data, self.shape)
            } else {
                write!(
                    f,
                    "tensor([{:.4}, {:.4}, ..., {:.4}], shape={})",
                    data[0],
                    data[1],
                    data[self.numel() - 1],
                    self.shape
                )
            }
        } else {
            write!(f, "tensor(shape={}, dtype={})", self.shape, self.dtype())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_f32() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_eq!(t.shape().dims(), &[2, 3]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.dtype(), DType::F32);
        assert!(t.is_contiguous());
    }

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(&[3, 4], DType::F32);
        assert_eq!(t.numel(), 12);
        let data = t.as_f32_slice().unwrap();
        assert!(data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_ones() {
        let t = Tensor::ones(&[2, 2]);
        let data = t.as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_scalar() {
        let t = Tensor::scalar(3.14);
        assert!(t.shape().is_scalar());
        assert_eq!(t.numel(), 1);
        assert_eq!(t.get_f32(0), Some(3.14));
    }

    #[test]
    fn test_arange() {
        let t = Tensor::arange(0.0, 5.0, 1.0);
        assert_eq!(t.shape().dims(), &[5]);
        let data = t.as_f32_slice().unwrap();
        assert_eq!(data, &[0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let r = t.reshape(&[3, 2]).unwrap();
        assert_eq!(r.shape().dims(), &[3, 2]);
        assert_eq!(r.as_f32_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reshape_infer() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let r = t.reshape(&[-1, 2]).unwrap();
        assert_eq!(r.shape().dims(), &[3, 2]);
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let tr = t.transpose().unwrap();
        assert_eq!(tr.shape().dims(), &[3, 2]);
        assert!(!tr.is_contiguous());

        // Verify transposed element access
        assert_eq!(tr.get_f32(0), Some(1.0)); // [0,0]
        assert_eq!(tr.get_f32(1), Some(4.0)); // [0,1] → original [1,0]
        assert_eq!(tr.get_f32(2), Some(2.0)); // [1,0] → original [0,1]
    }

    #[test]
    fn test_contiguous() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let tr = t.transpose().unwrap();
        assert!(!tr.is_contiguous());

        let c = tr.contiguous();
        assert!(c.is_contiguous());
        assert_eq!(c.shape().dims(), &[3, 2]);
        let data = c.as_f32_slice().unwrap();
        assert_eq!(data, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_requires_grad() {
        let mut t = Tensor::zeros(&[2, 2], DType::F32);
        assert!(!t.requires_grad());
        t.set_requires_grad(true);
        assert!(t.requires_grad());
    }

    #[test]
    fn test_debug_display() {
        let t = Tensor::from_f32(&[1.0, 2.0], &[2]);
        let debug = format!("{:?}", t);
        assert!(debug.contains("Tensor"));
        assert!(debug.contains("f32"));

        let display = format!("{}", t);
        assert!(display.contains("tensor"));
    }
}
