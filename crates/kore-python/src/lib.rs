// PyO3 error conversions use .into() which clippy flags as useless when
// the source is already PyErr — this is an intentional PyO3 idiom.
#![allow(clippy::useless_conversion)]
//! # kore-python
//!
//! PyO3 bindings for Kore → `import kore` in Python.
//!
//! Provides:
//! - `kore.Tensor` — wraps `kore_core::Tensor` with NumPy interop
//! - `kore.nn.Linear` — wraps `kore_nn::Linear`
//! - `kore.optim.Adam` / `kore.optim.SGD` — optimizer bindings
//! - `kore.functional` — activation functions

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArrayDyn, PyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods, PyArrayMethods};

// ============================================================================
// Tensor wrapper
// ============================================================================

#[pyclass(name = "Tensor")]
#[derive(Clone)]
struct PyTensor {
    inner: kore_core::Tensor,
}

#[pymethods]
impl PyTensor {
    /// Create a tensor from a NumPy array.
    #[new]
    fn new(data: PyReadonlyArrayDyn<'_, f32>) -> PyResult<Self> {
        let slice = data.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let shape: Vec<usize> = data.shape().to_vec();
        Ok(Self {
            inner: kore_core::Tensor::from_f32(slice, &shape),
        })
    }

    /// Create a tensor of zeros.
    #[staticmethod]
    fn zeros(shape: Vec<usize>) -> Self {
        Self {
            inner: kore_core::Tensor::zeros(&shape, kore_core::DType::F32),
        }
    }

    /// Create a tensor of ones.
    #[staticmethod]
    fn ones(shape: Vec<usize>) -> Self {
        Self {
            inner: kore_core::Tensor::ones(&shape),
        }
    }

    /// Shape as a list.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().dims().to_vec()
    }

    /// Number of dimensions.
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Total number of elements.
    #[getter]
    fn numel(&self) -> usize {
        self.inner.numel()
    }

    /// Data type as string.
    #[getter]
    fn dtype(&self) -> String {
        format!("{}", self.inner.dtype())
    }

    /// Convert to NumPy array (copies data).
    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let data = self.inner.contiguous();
        let slice = data.as_f32_slice()
            .ok_or_else(|| PyValueError::new_err("Cannot convert non-f32 tensor to numpy"))?;
        let shape: Vec<usize> = data.shape().dims().to_vec();
        let flat = PyArray1::from_vec_bound(py, slice.to_vec());
        flat.reshape(shape)
            .map_err(|e| PyValueError::new_err(format!("numpy reshape failed: {}", e)))
    }

    /// Element-wise addition.
    fn add(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = self.inner.add(&other.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    /// Element-wise subtraction.
    fn sub(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = self.inner.sub(&other.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    /// Element-wise multiplication.
    fn mul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = self.inner.mul(&other.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    /// Matrix multiplication.
    fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = self.inner.matmul(&other.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    /// Reshape.
    fn reshape(&self, shape: Vec<isize>) -> PyResult<PyTensor> {
        let result = self.inner.reshape(&shape)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    /// Transpose last two dimensions.
    fn transpose(&self) -> PyResult<PyTensor> {
        let result = self.inner.transpose()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    /// Sum all elements.
    fn sum(&self) -> PyResult<PyTensor> {
        let result = self.inner.sum()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    /// Mean of all elements.
    fn mean(&self) -> PyResult<PyTensor> {
        let result = self.inner.mean()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __add__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.add(other)
    }

    fn __sub__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.sub(other)
    }

    fn __mul__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.mul(other)
    }

    fn __matmul__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.matmul(other)
    }

    /// Softmax over given axis.
    #[pyo3(signature = (axis=-1))]
    fn softmax(&self, axis: isize) -> PyResult<PyTensor> {
        let result = self.inner.softmax(axis)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    /// Log-softmax over given axis.
    #[pyo3(signature = (axis=-1))]
    fn log_softmax(&self, axis: isize) -> PyResult<PyTensor> {
        let result = self.inner.log_softmax(axis)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    /// Masked fill: replace elements where mask > 0 with value.
    fn masked_fill(&self, mask: &PyTensor, value: f32) -> PyResult<PyTensor> {
        let result = self.inner.masked_fill(&mask.inner, value)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    /// Upper triangular.
    #[pyo3(signature = (k=0))]
    fn triu(&self, k: isize) -> PyResult<PyTensor> {
        let result = self.inner.triu(k)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    /// Lower triangular.
    #[pyo3(signature = (k=0))]
    fn tril(&self, k: isize) -> PyResult<PyTensor> {
        let result = self.inner.tril(k)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    /// Gather elements along axis.
    fn gather(&self, axis: isize, index: &PyTensor) -> PyResult<PyTensor> {
        let result = self.inner.gather(axis, &index.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    /// Concatenate tensors along axis.
    #[staticmethod]
    fn cat(tensors: Vec<PyTensor>, axis: isize) -> PyResult<PyTensor> {
        let refs: Vec<&kore_core::Tensor> = tensors.iter().map(|t| &t.inner).collect();
        let result = kore_core::Tensor::cat(&refs, axis)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    /// Stack tensors along new axis.
    #[staticmethod]
    fn stack(tensors: Vec<PyTensor>, axis: isize) -> PyResult<PyTensor> {
        let refs: Vec<&kore_core::Tensor> = tensors.iter().map(|t| &t.inner).collect();
        let result = kore_core::Tensor::stack(&refs, axis)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    /// Absolute value.
    fn abs(&self) -> PyResult<PyTensor> {
        let result = self.inner.abs()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    /// Create a tensor with random normal values N(0,1).
    #[staticmethod]
    fn randn(shape: Vec<usize>) -> Self {
        Self {
            inner: kore_core::Tensor::randn(&shape),
        }
    }

    /// Create a tensor with uniform random values in [low, high).
    #[staticmethod]
    #[pyo3(signature = (shape, low=0.0, high=1.0))]
    fn rand_uniform(shape: Vec<usize>, low: f32, high: f32) -> Self {
        Self {
            inner: kore_core::Tensor::rand_uniform(&shape, low, high),
        }
    }

    /// Enable gradient tracking on this tensor.
    fn requires_grad_(&mut self, requires_grad: bool) {
        self.inner.set_requires_grad(requires_grad);
    }

    /// Whether this tensor requires gradient.
    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    /// Get the accumulated gradient (None if no grad).
    #[getter]
    fn grad(&self) -> Option<PyTensor> {
        self.inner.grad().map(|g| PyTensor { inner: g })
    }

    /// Zero the gradient.
    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    /// Run backward pass (tensor must be scalar).
    fn backward(&self) -> PyResult<()> {
        self.inner.backward()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

// ============================================================================
// nn module
// ============================================================================

#[pyclass(name = "Linear")]
struct PyLinear {
    inner: kore_nn::Linear,
}

#[pymethods]
impl PyLinear {
    #[new]
    #[pyo3(signature = (in_features, out_features, bias=true))]
    fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        Self {
            inner: kore_nn::Linear::new(in_features, out_features, bias),
        }
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let result = kore_nn::Module::forward(&self.inner, &input.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<PyTensor> {
        kore_nn::Module::parameters(&self.inner)
            .into_iter()
            .map(|t| PyTensor { inner: t.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("Linear(in_features={}, out_features={}, bias={})",
            self.inner.in_features(), self.inner.out_features(), self.inner.has_bias())
    }
}

#[pyclass(name = "LayerNorm")]
struct PyLayerNorm {
    inner: kore_nn::LayerNorm,
}

#[pymethods]
impl PyLayerNorm {
    #[new]
    #[pyo3(signature = (normalized_shape, eps=1e-5))]
    fn new(normalized_shape: usize, eps: f32) -> Self {
        Self {
            inner: kore_nn::LayerNorm::new(normalized_shape, eps),
        }
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let result = kore_nn::Module::forward(&self.inner, &input.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<PyTensor> {
        kore_nn::Module::parameters(&self.inner)
            .into_iter()
            .map(|t| PyTensor { inner: t.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("LayerNorm(normalized_shape={}, eps={})",
            self.inner.normalized_shape(), self.inner.eps())
    }
}

#[pyclass(name = "Embedding")]
struct PyEmbedding {
    inner: kore_nn::Embedding,
}

#[pymethods]
impl PyEmbedding {
    #[new]
    fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        Self {
            inner: kore_nn::Embedding::new(num_embeddings, embedding_dim),
        }
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let result = kore_nn::Module::forward(&self.inner, &input.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn lookup(&self, ids: Vec<usize>) -> PyTensor {
        PyTensor { inner: self.inner.lookup(&ids) }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        kore_nn::Module::parameters(&self.inner)
            .into_iter()
            .map(|t| PyTensor { inner: t.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("Embedding(num_embeddings={}, embedding_dim={})",
            self.inner.num_embeddings(), self.inner.embedding_dim())
    }
}

#[pyclass(name = "BitLinear")]
struct PyBitLinear {
    inner: kore_nn::BitLinear,
}

#[pymethods]
impl PyBitLinear {
    #[new]
    #[pyo3(signature = (in_features, out_features, bias=false, threshold=0.3))]
    fn new(in_features: usize, out_features: usize, bias: bool, threshold: f32) -> Self {
        let linear = kore_nn::Linear::new(in_features, out_features, bias);
        Self {
            inner: kore_nn::BitLinear::from_linear(&linear, threshold),
        }
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let result = kore_nn::Module::forward(&self.inner, &input.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn compression_ratio(&self) -> f32 {
        self.inner.compression_ratio()
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

#[pyclass(name = "QuatLinear")]
struct PyQuatLinear {
    inner: kore_nn::QuatLinear,
}

#[pymethods]
impl PyQuatLinear {
    #[new]
    #[pyo3(signature = (in_features, out_features, bias=false))]
    fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let linear = kore_nn::Linear::new(in_features, out_features, bias);
        Self {
            inner: kore_nn::QuatLinear::from_linear(&linear),
        }
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let result = kore_nn::Module::forward(&self.inner, &input.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn compression_ratio(&self) -> f32 {
        self.inner.compression_ratio()
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

#[pyclass(name = "LoraLinear")]
struct PyLoraLinear {
    inner: kore_nn::LoraLinear,
}

#[pymethods]
impl PyLoraLinear {
    #[new]
    #[pyo3(signature = (in_features, out_features, rank=8, alpha=8.0, bias=false))]
    fn new(in_features: usize, out_features: usize, rank: usize, alpha: f32, bias: bool) -> Self {
        Self {
            inner: kore_nn::LoraLinear::new(in_features, out_features, rank, alpha, bias),
        }
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let result = kore_nn::Module::forward(&self.inner, &input.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn trainable_params(&self) -> usize {
        self.inner.trainable_params()
    }

    fn total_params(&self) -> usize {
        self.inner.total_params()
    }

    fn parameters(&self) -> Vec<PyTensor> {
        kore_nn::Module::parameters(&self.inner)
            .into_iter()
            .map(|t| PyTensor { inner: t.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

#[pyclass(name = "Dropout")]
struct PyDropout {
    inner: kore_nn::Dropout,
}

#[pymethods]
impl PyDropout {
    #[new]
    #[pyo3(signature = (p=0.5))]
    fn new(p: f32) -> Self {
        Self {
            inner: kore_nn::Dropout::new(p),
        }
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let result = kore_nn::Module::forward(&self.inner, &input.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn train(&mut self, mode: bool) {
        kore_nn::Module::train(&mut self.inner, mode);
    }

    fn eval(&mut self) {
        kore_nn::Module::train(&mut self.inner, false);
    }

    fn __repr__(&self) -> String {
        format!("Dropout(p={})", self.inner.p())
    }
}

#[pyclass(name = "Conv2d")]
struct PyConv2d {
    inner: kore_nn::Conv2d,
}

#[pymethods]
impl PyConv2d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=1, padding=0, bias=true))]
    fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize, bias: bool) -> Self {
        Self {
            inner: kore_nn::Conv2d::new(in_channels, out_channels, kernel_size, stride, padding, bias),
        }
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let result = kore_nn::Module::forward(&self.inner, &input.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<PyTensor> {
        kore_nn::Module::parameters(&self.inner)
            .into_iter()
            .map(|t| PyTensor { inner: t.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("Conv2d(in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={})",
            self.inner.in_channels(), self.inner.out_channels(),
            self.inner.kernel_size(), self.inner.stride(), self.inner.padding())
    }
}

#[pyclass(name = "MaxPool2d")]
struct PyMaxPool2d {
    inner: kore_nn::MaxPool2d,
}

#[pymethods]
impl PyMaxPool2d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=2, padding=0))]
    fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            inner: kore_nn::MaxPool2d::new(kernel_size, stride, padding),
        }
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let result = kore_nn::Module::forward(&self.inner, &input.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn __repr__(&self) -> String {
        format!("MaxPool2d(kernel_size={}, stride={}, padding={})",
            self.inner.kernel_size(), self.inner.stride(), self.inner.padding())
    }
}

#[pyclass(name = "AvgPool2d")]
struct PyAvgPool2d {
    inner: kore_nn::AvgPool2d,
}

#[pymethods]
impl PyAvgPool2d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=2, padding=0))]
    fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            inner: kore_nn::AvgPool2d::new(kernel_size, stride, padding),
        }
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let result = kore_nn::Module::forward(&self.inner, &input.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn __repr__(&self) -> String {
        format!("AvgPool2d(kernel_size={}, stride={}, padding={})",
            self.inner.kernel_size(), self.inner.stride(), self.inner.padding())
    }
}

#[pyclass(name = "AdaptiveAvgPool2d")]
struct PyAdaptiveAvgPool2d {
    inner: kore_nn::AdaptiveAvgPool2d,
}

#[pymethods]
impl PyAdaptiveAvgPool2d {
    #[new]
    #[pyo3(signature = (output_h, output_w))]
    fn new(output_h: usize, output_w: usize) -> Self {
        Self {
            inner: kore_nn::AdaptiveAvgPool2d::new(output_h, output_w),
        }
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let result = kore_nn::Module::forward(&self.inner, &input.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: result })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn __repr__(&self) -> String {
        format!("AdaptiveAvgPool2d(output_size=({}, {}))",
            self.inner.output_h(), self.inner.output_w())
    }
}

// ============================================================================
// optim module
// ============================================================================

#[pyclass(name = "Adam")]
struct PyAdam {
    inner: kore_optim::Adam,
}

#[pymethods]
impl PyAdam {
    #[new]
    #[pyo3(signature = (lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0))]
    fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            inner: kore_optim::Adam::new(lr, beta1, beta2, eps, weight_decay),
        }
    }

    /// Update parameters in-place. Params list is mutated directly.
    ///
    /// Usage::
    ///
    ///     optimizer.step(params, grads)  # params are updated in-place
    fn step(&mut self, params: Bound<'_, pyo3::types::PyList>, grads: Vec<PyTensor>) -> PyResult<()> {
        let g: Vec<kore_core::Tensor> = grads.into_iter().map(|t| t.inner).collect();
        let len = params.len();
        // Extract inner tensors (takes ownership temporarily)
        let mut p: Vec<kore_core::Tensor> = Vec::with_capacity(len);
        for i in 0..len {
            let item = params.get_item(i)?;
            let t: PyRef<'_, PyTensor> = item.extract()?;
            p.push(t.inner.clone());
        }
        self.inner.step(&mut p, &g);
        // Write updated tensors back into the Python objects
        for i in 0..len {
            let item = params.get_item(i)?;
            let mut t: PyRefMut<'_, PyTensor> = item.extract()?;
            t.inner = p.remove(0);
        }
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("Adam(lr={}, betas=({}, {}), eps={}, weight_decay={})",
            self.inner.lr(), self.inner.beta1(), self.inner.beta2(),
            self.inner.eps(), self.inner.weight_decay())
    }
}

#[pyclass(name = "SGD")]
struct PySGD {
    inner: kore_optim::SGD,
}

#[pymethods]
impl PySGD {
    #[new]
    #[pyo3(signature = (lr=0.01, momentum=0.0, weight_decay=0.0))]
    fn new(lr: f32, momentum: f32, weight_decay: f32) -> Self {
        Self {
            inner: kore_optim::SGD::new(lr, momentum, weight_decay),
        }
    }

    /// Update parameters in-place. Params list is mutated directly.
    fn step(&mut self, params: Bound<'_, pyo3::types::PyList>, grads: Vec<PyTensor>) -> PyResult<()> {
        let g: Vec<kore_core::Tensor> = grads.into_iter().map(|t| t.inner).collect();
        let len = params.len();
        let mut p: Vec<kore_core::Tensor> = Vec::with_capacity(len);
        for i in 0..len {
            let item = params.get_item(i)?;
            let t: PyRef<'_, PyTensor> = item.extract()?;
            p.push(t.inner.clone());
        }
        self.inner.step(&mut p, &g);
        for i in 0..len {
            let item = params.get_item(i)?;
            let mut t: PyRefMut<'_, PyTensor> = item.extract()?;
            t.inner = p.remove(0);
        }
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("SGD(lr={}, momentum={}, weight_decay={})",
            self.inner.lr(), self.inner.momentum(), self.inner.weight_decay())
    }
}

// ============================================================================
// Functional API
// ============================================================================

/// ReLU activation.
#[pyfunction]
fn relu(input: &PyTensor) -> PyResult<PyTensor> {
    let result = kore_nn::activations::relu(&input.inner)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyTensor { inner: result })
}

/// GELU activation.
#[pyfunction]
fn gelu(input: &PyTensor) -> PyResult<PyTensor> {
    let result = kore_nn::activations::gelu(&input.inner)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyTensor { inner: result })
}

/// Softmax over last dimension.
#[pyfunction]
fn softmax(input: &PyTensor) -> PyResult<PyTensor> {
    let result = kore_nn::activations::softmax(&input.inner)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyTensor { inner: result })
}

/// Sigmoid activation.
#[pyfunction]
fn sigmoid(input: &PyTensor) -> PyResult<PyTensor> {
    let result = kore_nn::activations::sigmoid(&input.inner)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyTensor { inner: result })
}

/// Cross-entropy loss.
#[pyfunction]
fn cross_entropy_loss(logits: &PyTensor, targets: &PyTensor) -> PyResult<PyTensor> {
    let result = kore_nn::cross_entropy_loss(&logits.inner, &targets.inner)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyTensor { inner: result })
}

/// MSE loss.
#[pyfunction]
fn mse_loss(pred: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    let result = kore_nn::mse_loss(&pred.inner, &target.inner)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyTensor { inner: result })
}

/// L1 loss.
#[pyfunction]
fn l1_loss(pred: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    let result = kore_nn::l1_loss(&pred.inner, &target.inner)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyTensor { inner: result })
}

/// NLL loss.
#[pyfunction]
fn nll_loss(log_probs: &PyTensor, targets: &PyTensor) -> PyResult<PyTensor> {
    let result = kore_nn::nll_loss(&log_probs.inner, &targets.inner)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyTensor { inner: result })
}

/// Tanh activation.
#[pyfunction]
fn tanh(input: &PyTensor) -> PyResult<PyTensor> {
    let result = kore_nn::activations::tanh(&input.inner)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyTensor { inner: result })
}

/// SiLU (Swish) activation.
#[pyfunction]
fn silu(input: &PyTensor) -> PyResult<PyTensor> {
    let result = kore_nn::activations::silu(&input.inner)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyTensor { inner: result })
}

/// Save a state dict to a safetensors file.
#[pyfunction]
fn save_state_dict(state_dict: std::collections::HashMap<String, PyTensor>, path: String) -> PyResult<()> {
    let sd: std::collections::HashMap<String, kore_core::Tensor> = state_dict
        .into_iter()
        .map(|(k, v)| (k, v.inner))
        .collect();
    kore_nn::save_state_dict(&sd, std::path::Path::new(&path))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Load a state dict from a safetensors file.
#[pyfunction]
fn load_state_dict(path: String) -> PyResult<std::collections::HashMap<String, PyTensor>> {
    let sd = kore_nn::load_state_dict(std::path::Path::new(&path))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(sd.into_iter().map(|(k, v)| (k, PyTensor { inner: v })).collect())
}

// ============================================================================
// Module entry point
// ============================================================================

#[pymodule]
fn kore(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(gelu, m)?)?;
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, m)?)?;

    // nn submodule
    let nn = PyModule::new_bound(py, "nn")?;
    nn.add_class::<PyLinear>()?;
    nn.add_class::<PyBitLinear>()?;
    nn.add_class::<PyLoraLinear>()?;
    nn.add_class::<PyLayerNorm>()?;
    nn.add_class::<PyQuatLinear>()?;
    nn.add_class::<PyEmbedding>()?;
    nn.add_class::<PyDropout>()?;
    nn.add_class::<PyConv2d>()?;
    nn.add_class::<PyMaxPool2d>()?;
    nn.add_class::<PyAvgPool2d>()?;
    nn.add_class::<PyAdaptiveAvgPool2d>()?;
    m.add_submodule(&nn)?;

    // optim submodule
    let optim = PyModule::new_bound(py, "optim")?;
    optim.add_class::<PyAdam>()?;
    optim.add_class::<PySGD>()?;
    m.add_submodule(&optim)?;

    // functional submodule (loss functions + activations)
    let functional = PyModule::new_bound(py, "functional")?;
    functional.add_function(wrap_pyfunction!(relu, &functional)?)?;
    functional.add_function(wrap_pyfunction!(gelu, &functional)?)?;
    functional.add_function(wrap_pyfunction!(softmax, &functional)?)?;
    functional.add_function(wrap_pyfunction!(sigmoid, &functional)?)?;
    functional.add_function(wrap_pyfunction!(cross_entropy_loss, &functional)?)?;
    functional.add_function(wrap_pyfunction!(mse_loss, &functional)?)?;
    functional.add_function(wrap_pyfunction!(l1_loss, &functional)?)?;
    functional.add_function(wrap_pyfunction!(nll_loss, &functional)?)?;
    functional.add_function(wrap_pyfunction!(tanh, &functional)?)?;
    functional.add_function(wrap_pyfunction!(silu, &functional)?)?;
    m.add_submodule(&functional)?;

    // io functions
    m.add_function(wrap_pyfunction!(save_state_dict, m)?)?;
    m.add_function(wrap_pyfunction!(load_state_dict, m)?)?;

    Ok(())
}
