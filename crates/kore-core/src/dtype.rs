use std::fmt;

/// Data types supported by Kore tensors.
///
/// Includes standard IEEE floats, integers, and native quantized types
/// (Ternary and Quaternary) as first-class citizens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 16-bit IEEE 754 half-precision float
    F16,
    /// 16-bit Brain Float (same exponent range as F32, reduced mantissa)
    BF16,
    /// 32-bit IEEE 754 single-precision float
    F32,
    /// 64-bit IEEE 754 double-precision float
    F64,
    /// 8-bit signed integer
    I8,
    /// 8-bit unsigned integer
    U8,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// Balanced ternary: {-1, 0, +1}, packed 5 trits per byte (base-243)
    Ternary,
    /// Quaternary: {-3, -1, +1, +3}, packed 4 values per byte (2 bits each)
    Quaternary,
}

impl DType {
    /// Size in bytes of a single element, or None for packed types.
    pub fn element_size(&self) -> Option<usize> {
        match self {
            DType::F16 | DType::BF16 => Some(2),
            DType::F32 => Some(4),
            DType::F64 => Some(8),
            DType::I8 | DType::U8 => Some(1),
            DType::I32 => Some(4),
            DType::I64 => Some(8),
            // Packed types: size depends on element count, not per-element
            DType::Ternary => None,
            DType::Quaternary => None,
        }
    }

    /// Number of bytes needed to store `n` elements of this dtype.
    pub fn storage_bytes(&self, n: usize) -> usize {
        match self {
            DType::Ternary => {
                // 5 trits per byte (base-243 packing)
                (n + 4) / 5
            }
            DType::Quaternary => {
                // 4 values per byte (2 bits each)
                (n + 3) / 4
            }
            other => {
                other.element_size().unwrap() * n
            }
        }
    }

    /// Whether this dtype is a floating-point type.
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    }

    /// Whether this dtype is an integer type.
    pub fn is_integer(&self) -> bool {
        matches!(self, DType::I8 | DType::U8 | DType::I32 | DType::I64)
    }

    /// Whether this dtype is a quantized/packed type.
    pub fn is_quantized(&self) -> bool {
        matches!(self, DType::Ternary | DType::Quaternary)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::F32 => write!(f, "f32"),
            DType::F64 => write!(f, "f64"),
            DType::I8 => write!(f, "i8"),
            DType::U8 => write!(f, "u8"),
            DType::I32 => write!(f, "i32"),
            DType::I64 => write!(f, "i64"),
            DType::Ternary => write!(f, "ternary"),
            DType::Quaternary => write!(f, "quaternary"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_sizes() {
        assert_eq!(DType::F32.element_size(), Some(4));
        assert_eq!(DType::F64.element_size(), Some(8));
        assert_eq!(DType::F16.element_size(), Some(2));
        assert_eq!(DType::I8.element_size(), Some(1));
        assert_eq!(DType::Ternary.element_size(), None);
        assert_eq!(DType::Quaternary.element_size(), None);
    }

    #[test]
    fn test_storage_bytes() {
        assert_eq!(DType::F32.storage_bytes(10), 40);
        assert_eq!(DType::Ternary.storage_bytes(5), 1);
        assert_eq!(DType::Ternary.storage_bytes(6), 2);
        assert_eq!(DType::Quaternary.storage_bytes(4), 1);
        assert_eq!(DType::Quaternary.storage_bytes(5), 2);
    }

    #[test]
    fn test_dtype_categories() {
        assert!(DType::F32.is_float());
        assert!(!DType::F32.is_integer());
        assert!(!DType::F32.is_quantized());
        assert!(DType::I32.is_integer());
        assert!(DType::Ternary.is_quantized());
        assert!(DType::Quaternary.is_quantized());
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", DType::F32), "f32");
        assert_eq!(format!("{}", DType::Ternary), "ternary");
        assert_eq!(format!("{}", DType::Quaternary), "quaternary");
    }
}
