use smallvec::SmallVec;
use std::fmt;

/// Tensor shape with stack-allocated storage for ≤4 dimensions.
///
/// Most ML tensors are 1D-4D (scalars, vectors, matrices, batched matrices),
/// so we avoid heap allocation for the common case.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: SmallVec<[usize; 4]>,
}

impl Shape {
    /// Create a new shape from dimensions.
    pub fn new(dims: &[usize]) -> Self {
        Self {
            dims: SmallVec::from_slice(dims),
        }
    }

    /// Scalar shape (0 dimensions).
    pub fn scalar() -> Self {
        Self {
            dims: SmallVec::new(),
        }
    }

    /// Number of dimensions (rank).
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        if self.dims.is_empty() {
            1 // scalar
        } else {
            self.dims.iter().product()
        }
    }

    /// Get dimension sizes as a slice.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Get size of a specific dimension.
    pub fn dim(&self, axis: usize) -> Option<usize> {
        self.dims.get(axis).copied()
    }

    /// Whether this is a scalar (0-dimensional).
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// Compute default strides for a contiguous row-major layout.
    pub fn contiguous_strides(&self) -> SmallVec<[usize; 4]> {
        let ndim = self.dims.len();
        if ndim == 0 {
            return SmallVec::new();
        }
        let mut strides = SmallVec::from_elem(0usize, ndim);
        strides[ndim - 1] = 1;
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        strides
    }

    /// Attempt to broadcast this shape with another.
    /// Returns the broadcasted shape or None if incompatible.
    pub fn broadcast_with(&self, other: &Shape) -> Option<Shape> {
        let max_ndim = self.ndim().max(other.ndim());
        let mut result = SmallVec::with_capacity(max_ndim);

        for i in 0..max_ndim {
            let a = if i < self.ndim() {
                self.dims[self.ndim() - 1 - i]
            } else {
                1
            };
            let b = if i < other.ndim() {
                other.dims[other.ndim() - 1 - i]
            } else {
                1
            };

            if a == b {
                result.push(a);
            } else if a == 1 {
                result.push(b);
            } else if b == 1 {
                result.push(a);
            } else {
                return None;
            }
        }

        result.reverse();
        Some(Shape { dims: result })
    }

    /// Validate and compute a reshape target.
    /// At most one dimension can be -1 (inferred).
    pub fn resolve_reshape(&self, target: &[isize]) -> Option<Shape> {
        let numel = self.numel();
        let mut inferred_idx = None;
        let mut known_product: usize = 1;

        for (i, &d) in target.iter().enumerate() {
            if d == -1 {
                if inferred_idx.is_some() {
                    return None; // multiple -1s
                }
                inferred_idx = Some(i);
            } else if d <= 0 {
                return None; // invalid dimension
            } else {
                known_product = known_product.checked_mul(d as usize)?;
            }
        }

        let mut result: SmallVec<[usize; 4]> = target
            .iter()
            .map(|&d| if d == -1 { 0 } else { d as usize })
            .collect();

        if let Some(idx) = inferred_idx {
            if known_product == 0 || numel % known_product != 0 {
                return None;
            }
            result[idx] = numel / known_product;
        }

        let result_shape = Shape { dims: result };
        if result_shape.numel() != numel {
            return None;
        }
        Some(result_shape)
    }

    /// Compute the transposed shape (swap last two dimensions).
    pub fn transpose(&self) -> Option<Shape> {
        if self.ndim() < 2 {
            return None;
        }
        let mut dims = self.dims.clone();
        let n = dims.len();
        dims.swap(n - 2, n - 1);
        Some(Shape { dims })
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape({:?})", self.dims.as_slice())
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape::new(dims)
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape {
            dims: SmallVec::from_vec(dims),
        }
    }
}

macro_rules! impl_shape_from_array {
    ($($n:expr),*) => {
        $(
            impl From<[usize; $n]> for Shape {
                fn from(dims: [usize; $n]) -> Self {
                    Shape::new(&dims)
                }
            }
        )*
    };
}

impl_shape_from_array!(0, 1, 2, 3, 4, 5, 6);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar() {
        let s = Shape::scalar();
        assert_eq!(s.ndim(), 0);
        assert_eq!(s.numel(), 1);
        assert!(s.is_scalar());
    }

    #[test]
    fn test_basic_shape() {
        let s = Shape::new(&[2, 3, 4]);
        assert_eq!(s.ndim(), 3);
        assert_eq!(s.numel(), 24);
        assert_eq!(s.dim(0), Some(2));
        assert_eq!(s.dim(1), Some(3));
        assert_eq!(s.dim(2), Some(4));
        assert_eq!(s.dim(3), None);
    }

    #[test]
    fn test_contiguous_strides() {
        let s = Shape::new(&[2, 3, 4]);
        let strides = s.contiguous_strides();
        assert_eq!(strides.as_slice(), &[12, 4, 1]);
    }

    #[test]
    fn test_broadcast() {
        let a = Shape::new(&[3, 1]);
        let b = Shape::new(&[1, 4]);
        let c = a.broadcast_with(&b).unwrap();
        assert_eq!(c.dims(), &[3, 4]);

        let a = Shape::new(&[2, 3]);
        let b = Shape::new(&[3]);
        let c = a.broadcast_with(&b).unwrap();
        assert_eq!(c.dims(), &[2, 3]);

        let a = Shape::new(&[2, 3]);
        let b = Shape::new(&[4, 3]);
        assert!(a.broadcast_with(&b).is_none());
    }

    #[test]
    fn test_reshape() {
        let s = Shape::new(&[2, 3, 4]);
        let r = s.resolve_reshape(&[6, 4]).unwrap();
        assert_eq!(r.dims(), &[6, 4]);

        let r = s.resolve_reshape(&[-1, 4]).unwrap();
        assert_eq!(r.dims(), &[6, 4]);

        let r = s.resolve_reshape(&[2, -1]).unwrap();
        assert_eq!(r.dims(), &[2, 12]);

        assert!(s.resolve_reshape(&[-1, -1]).is_none());
        assert!(s.resolve_reshape(&[5, 5]).is_none());
    }

    #[test]
    fn test_transpose() {
        let s = Shape::new(&[2, 3, 4]);
        let t = s.transpose().unwrap();
        assert_eq!(t.dims(), &[2, 4, 3]);

        let s = Shape::new(&[5]);
        assert!(s.transpose().is_none());
    }

    #[test]
    fn test_from_array() {
        let s: Shape = [2, 3].into();
        assert_eq!(s.dims(), &[2, 3]);

        let s: Shape = [1, 2, 3, 4].into();
        assert_eq!(s.numel(), 24);
    }
}
