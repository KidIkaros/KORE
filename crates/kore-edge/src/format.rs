//! `.koref` model format — single-file, mmap-friendly model container.
//!
//! Layout:
//! ```text
//! ┌──────────────────────────────────┐
//! │ Magic: "KORF" (4 bytes)          │
//! │ Version: u32 LE                  │
//! │ Header size: u32 LE              │
//! ├──────────────────────────────────┤
//! │ Header JSON (config, tensor idx) │
//! ├──────────────────────────────────┤
//! │ Padding to 64-byte alignment     │
//! ├──────────────────────────────────┤
//! │ Weight blob (contiguous tensors) │
//! └──────────────────────────────────┘
//! ```

use std::collections::HashMap;

/// Magic bytes identifying a .koref file.
pub const MAGIC: &[u8; 4] = b"KORF";

/// Current format version.
pub const VERSION: u32 = 1;

/// Alignment for the weight blob (SIMD-friendly).
const BLOB_ALIGNMENT: usize = 64;

/// Data type for tensors in the .koref format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EdgeDType {
    F32,
    F16,
    Ternary,
    Quaternary,
    I8,
    U8,
}

impl EdgeDType {
    /// Bytes needed to store `n` elements.
    pub fn storage_bytes(&self, n: usize) -> usize {
        match self {
            EdgeDType::F32 => n * 4,
            EdgeDType::F16 => n * 2,
            EdgeDType::I8 | EdgeDType::U8 => n,
            EdgeDType::Ternary => n.div_ceil(5),
            EdgeDType::Quaternary => n.div_ceil(4),
        }
    }

    /// String tag for JSON serialization.
    pub fn as_str(&self) -> &'static str {
        match self {
            EdgeDType::F32 => "f32",
            EdgeDType::F16 => "f16",
            EdgeDType::Ternary => "ternary",
            EdgeDType::Quaternary => "quaternary",
            EdgeDType::I8 => "i8",
            EdgeDType::U8 => "u8",
        }
    }

    /// Parse from string tag.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "f32" => Some(EdgeDType::F32),
            "f16" => Some(EdgeDType::F16),
            "ternary" => Some(EdgeDType::Ternary),
            "quaternary" => Some(EdgeDType::Quaternary),
            "i8" => Some(EdgeDType::I8),
            "u8" => Some(EdgeDType::U8),
            _ => None,
        }
    }
}

/// Index entry for a single tensor in the weight blob.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TensorEntry {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub nbytes: usize,
}

impl TensorEntry {
    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Operator type in the execution graph.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OpType {
    Embedding,
    Linear,
    RMSNorm,
    LayerNorm,
    RoPE,
    Attention,
    FeedForward,
    Softmax,
    Residual,
    Final,
}

/// A single operation in the model graph.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OpNode {
    pub op: OpType,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub params: HashMap<String, String>,
}

/// Header metadata for a .koref model.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KorefHeader {
    pub model_type: String,
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub norm_eps: f32,
    pub rope_base: f32,
    pub tensors: HashMap<String, TensorEntry>,
    pub ops: Vec<OpNode>,
}

/// A loaded .koref model — header + weight data.
pub struct KorefModel {
    pub header: KorefHeader,
    pub weights: Vec<u8>,
}

impl KorefModel {
    /// Load a .koref model from raw bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, FormatError> {
        if data.len() < 12 {
            return Err(FormatError::TooSmall);
        }

        // Check magic
        if &data[0..4] != MAGIC {
            return Err(FormatError::BadMagic);
        }

        // Version
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version != VERSION {
            return Err(FormatError::UnsupportedVersion(version));
        }

        // Header size
        let header_size = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        if data.len() < 12 + header_size {
            return Err(FormatError::TooSmall);
        }

        // Parse header JSON
        let header_bytes = &data[12..12 + header_size];
        let header_str = std::str::from_utf8(header_bytes)
            .map_err(|_| FormatError::InvalidHeader("not valid UTF-8".into()))?;

        #[cfg(feature = "serde")]
        let header: KorefHeader = serde_json::from_str(header_str)
            .map_err(|e| FormatError::InvalidHeader(e.to_string()))?;

        #[cfg(not(feature = "serde"))]
        return Err(FormatError::InvalidHeader("serde feature required".into()));

        // Weight blob starts at next 64-byte aligned offset
        let blob_start = align_up(12 + header_size, BLOB_ALIGNMENT);
        if data.len() < blob_start {
            return Err(FormatError::TooSmall);
        }

        let weights = data[blob_start..].to_vec();

        Ok(KorefModel { header, weights })
    }

    /// Serialize this model to .koref bytes.
    #[cfg(feature = "serde")]
    pub fn to_bytes(&self) -> Vec<u8> {
        let header_json = serde_json::to_string(&self.header).unwrap();
        let header_bytes = header_json.as_bytes();
        let header_size = header_bytes.len();

        let blob_start = align_up(12 + header_size, BLOB_ALIGNMENT);
        let padding = blob_start - (12 + header_size);

        let total = blob_start + self.weights.len();
        let mut buf = Vec::with_capacity(total);

        // Magic
        buf.extend_from_slice(MAGIC);
        // Version
        buf.extend_from_slice(&VERSION.to_le_bytes());
        // Header size
        buf.extend_from_slice(&(header_size as u32).to_le_bytes());
        // Header JSON
        buf.extend_from_slice(header_bytes);
        // Padding
        buf.extend(std::iter::repeat_n(0u8, padding));
        // Weight blob
        buf.extend_from_slice(&self.weights);

        buf
    }

    /// Get raw bytes for a named tensor.
    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let entry = self.header.tensors.get(name)?;
        if entry.offset + entry.nbytes > self.weights.len() {
            return None;
        }
        Some(&self.weights[entry.offset..entry.offset + entry.nbytes])
    }

    /// Get tensor data as f32 slice (only valid for F32 tensors).
    pub fn tensor_f32(&self, name: &str) -> Option<&[f32]> {
        let entry = self.header.tensors.get(name)?;
        if entry.dtype != "f32" {
            return None;
        }
        let data = self.tensor_data(name)?;
        let ptr = data.as_ptr() as *const f32;
        let len = data.len() / 4;
        Some(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    /// Get per-row scales for a quantized tensor (stored as "{name}.scales").
    pub fn tensor_scales(&self, name: &str) -> Option<&[f32]> {
        let scale_name = format!("{}.scales", name);
        self.tensor_f32(&scale_name)
    }
}

/// Errors from .koref parsing.
#[derive(Debug)]
pub enum FormatError {
    TooSmall,
    BadMagic,
    UnsupportedVersion(u32),
    InvalidHeader(String),
    IoError(String),
}

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FormatError::TooSmall => write!(f, "file too small for .koref format"),
            FormatError::BadMagic => write!(f, "invalid magic bytes (expected KORF)"),
            FormatError::UnsupportedVersion(v) => write!(f, "unsupported version: {}", v),
            FormatError::InvalidHeader(e) => write!(f, "invalid header: {}", e),
            FormatError::IoError(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for FormatError {}

/// Round `n` up to the next multiple of `align`.
fn align_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

/// Builder for creating .koref models programmatically.
pub struct KorefBuilder {
    header: KorefHeader,
    weight_buf: Vec<u8>,
    current_offset: usize,
}

impl KorefBuilder {
    /// Create a new builder with model config.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_type: &str,
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
        n_layers: usize,
        d_ff: usize,
        max_seq_len: usize,
        norm_eps: f32,
        rope_base: f32,
    ) -> Self {
        Self {
            header: KorefHeader {
                model_type: model_type.to_string(),
                vocab_size,
                d_model,
                n_heads,
                n_kv_heads,
                n_layers,
                d_ff,
                max_seq_len,
                norm_eps,
                rope_base,
                tensors: HashMap::new(),
                ops: Vec::new(),
            },
            weight_buf: Vec::new(),
            current_offset: 0,
        }
    }

    /// Add a tensor (raw bytes) to the model.
    pub fn add_tensor(&mut self, name: &str, dtype: EdgeDType, shape: &[usize], data: &[u8]) {
        // Align to 64 bytes
        let padding = align_up(self.current_offset, 64) - self.current_offset;
        self.weight_buf.extend(std::iter::repeat_n(0u8, padding));
        self.current_offset += padding;

        let entry = TensorEntry {
            dtype: dtype.as_str().to_string(),
            shape: shape.to_vec(),
            offset: self.current_offset,
            nbytes: data.len(),
        };

        self.header.tensors.insert(name.to_string(), entry);
        self.weight_buf.extend_from_slice(data);
        self.current_offset += data.len();
    }

    /// Add an f32 tensor.
    pub fn add_f32(&mut self, name: &str, shape: &[usize], data: &[f32]) {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        self.add_tensor(name, EdgeDType::F32, shape, bytes);
    }

    /// Add an operation to the graph.
    pub fn add_op(&mut self, op: OpNode) {
        self.header.ops.push(op);
    }

    /// Build the final KorefModel.
    pub fn build(self) -> KorefModel {
        KorefModel {
            header: self.header,
            weights: self.weight_buf,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
    }

    #[test]
    fn test_edge_dtype_storage() {
        assert_eq!(EdgeDType::F32.storage_bytes(10), 40);
        assert_eq!(EdgeDType::F16.storage_bytes(10), 20);
        assert_eq!(EdgeDType::Ternary.storage_bytes(5), 1);
        assert_eq!(EdgeDType::Quaternary.storage_bytes(4), 1);
    }

    #[test]
    fn test_edge_dtype_roundtrip() {
        for dt in &[EdgeDType::F32, EdgeDType::F16, EdgeDType::Ternary, EdgeDType::Quaternary, EdgeDType::I8, EdgeDType::U8] {
            assert_eq!(EdgeDType::from_str(dt.as_str()), Some(*dt));
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_roundtrip() {
        let mut builder = KorefBuilder::new("test", 256, 64, 4, 4, 2, 128, 128, 1e-5, 10000.0);

        let w = vec![1.0f32, 2.0, 3.0, 4.0];
        builder.add_f32("embed.weight", &[2, 2], &w);

        let model = builder.build();
        let bytes = model.to_bytes();

        let loaded = KorefModel::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.header.model_type, "test");
        assert_eq!(loaded.header.vocab_size, 256);
        assert_eq!(loaded.header.d_model, 64);
        assert!(loaded.header.tensors.contains_key("embed.weight"));

        let data = loaded.tensor_f32("embed.weight").unwrap();
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_bad_magic() {
        let data = b"BADXxxxxxxxx";
        assert!(matches!(KorefModel::from_bytes(data), Err(FormatError::BadMagic)));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_too_small() {
        let data = b"KORF";
        assert!(matches!(KorefModel::from_bytes(data), Err(FormatError::TooSmall)));
    }
}
