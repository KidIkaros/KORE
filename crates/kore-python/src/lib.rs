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

    /// Convert to NumPy array (zero-copy when possible).
    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let data = self.inner.contiguous();
        let slice = data.as_f32_slice()
            .ok_or_else(|| PyValueError::new_err("Cannot convert non-f32 tensor to numpy"))?;
        let shape: Vec<usize> = data.shape().dims().to_vec();
        let flat = PyArray1::from_vec_bound(py, slice.to_vec());
        Ok(flat.reshape(shape).unwrap())
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

    fn step(&mut self, params: Vec<PyTensor>, grads: Vec<PyTensor>) -> PyResult<Vec<PyTensor>> {
        let mut p: Vec<kore_core::Tensor> = params.into_iter().map(|t| t.inner).collect();
        let g: Vec<kore_core::Tensor> = grads.into_iter().map(|t| t.inner).collect();
        self.inner.step(&mut p, &g);
        Ok(p.into_iter().map(|t| PyTensor { inner: t }).collect())
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

    fn step(&mut self, params: Vec<PyTensor>, grads: Vec<PyTensor>) -> PyResult<Vec<PyTensor>> {
        let mut p: Vec<kore_core::Tensor> = params.into_iter().map(|t| t.inner).collect();
        let g: Vec<kore_core::Tensor> = grads.into_iter().map(|t| t.inner).collect();
        self.inner.step(&mut p, &g);
        Ok(p.into_iter().map(|t| PyTensor { inner: t }).collect())
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
    m.add_submodule(&functional)?;

    Ok(())
}
