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

// ============================================================================
// Module entry point
// ============================================================================

#[pymodule]
fn kore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(gelu, m)?)?;
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, m)?)?;
    Ok(())
}
