//! Minimal bounded GGUF v3 reader for architectures oxillama-gguf cannot parse
//! (e.g. PrismML custom tensor types like Q2_0 id 42).
//!
//! Memory-safe by construction: mmap-backed, every read bounds-checked against
//! the mapped length, no allocation sized by untrusted counts. Large tokenizer
//! arrays are walked but not stored.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use kore_core::{KoreError, Result};
use memmap2::Mmap;

/// GGUF metadata value (only types we decode are stored).
#[derive(Debug, Clone)]
pub enum MetaVal {
    I64(i64),
    U64(u64),
    F64(f64),
    Bool(bool),
    Str(String),
    StrArr(Vec<String>),
    F32Arr(Vec<f32>),
    I64Arr(Vec<i64>),
}

impl MetaVal {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U64(v) => Some(*v),
            Self::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F64(v) => Some(*v),
            Self::I64(v) => Some(*v as f64),
            Self::U64(v) => Some(*v as f64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_str_arr(&self) -> Option<&[String]> {
        match self {
            Self::StrArr(v) => Some(v),
            _ => None,
        }
    }
    pub fn as_f32_arr(&self) -> Option<&[f32]> {
        match self {
            Self::F32Arr(v) => Some(v),
            _ => None,
        }
    }
    pub fn as_i64_arr(&self) -> Option<&[i64]> {
        match self {
            Self::I64Arr(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub name: String,
    pub type_id: u32,
    /// GGUF dims, ne[0] contiguous.
    pub dims: Vec<usize>,
    /// Absolute offset of tensor data within the file.
    pub abs_offset: usize,
    /// Byte length of tensor payload.
    pub nbytes: usize,
}

impl TensorEntry {
    pub fn n_elements(&self) -> usize {
        self.dims.iter().product()
    }
}

/// Parsed GGUF file: metadata + tensor table + mmap for payload access.
pub struct MiniGguf {
    map: Mmap,
    pub meta: HashMap<String, MetaVal>,
    pub tensors: Vec<TensorEntry>,
    by_name: HashMap<String, usize>,
}

const ALIGN: u64 = 32;

struct Rd<'a> {
    d: &'a [u8],
    pos: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| err("offset overflow"))?;
        if end > self.d.len() {
            return Err(err("read past end of file"));
        }
        let s = &self.d[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        if n > 1 << 24 {
            return Err(err("string too long"));
        }
        let b = self.take(n)?;
        String::from_utf8(b.to_vec()).map_err(|_| err("invalid utf8 in string"))
    }
}

fn err(msg: &str) -> KoreError {
    KoreError::StorageError(format!("gguf_min: {msg}"))
}

/// Payload geometry `(elements per block, bytes per block)` for known type ids;
/// returns None for unknown ids.
///
/// This table is what lets the reader *size* a tensor table, so a model cannot
/// be loaded until its quant geometry is here -- even if the compute kernels
/// already exist. It covers the standard ggml types so that real HuggingFace
/// GGUFs (which ship K-quants and IQ-quants, not Q2_0) can at least be parsed.
/// Reading the *bytes* is a separate concern (`tensor_f32` / the kernels).
///
/// Block sizes: `QK_K = 256` for the K-quants. The 32-wide legacy quants carry a
/// single f16 scale; the K-quants carry super-block scales plus 6-bit sub-scales.
fn type_block(type_id: u32) -> Option<(usize, usize)> {
    // (elements per block, bytes per block)
    Some(match type_id {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        30 => (1, 2),     // BF16
        2 => (32, 18),    // Q4_0     : f16 d + 16 B of 4-bit
        3 => (32, 20),    // Q4_1     : f16 d + f16 m + 16 B
        6 => (32, 22),    // Q5_0     : f16 d + 4 B qh + 16 B
        7 => (32, 24),    // Q5_1     : f16 d + f16 m + 4 B qh + 16 B
        8 => (32, 34),    // Q8_0     : f16 d + 32 B
        9 => (32, 36),    // Q8_1     : f16 d + f16 s + 32 B
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        15 => (256, 292), // Q8_K
        16 => (256, 66),  // IQ2_XXS
        17 => (256, 74),  // IQ2_XS
        18 => (256, 98),  // IQ3_XXS
        19 => (256, 50),  // IQ1_S
        20 => (32, 18),   // IQ4_NL   : like Q4_0 but a non-linear codebook
        21 => (256, 110), // IQ3_S
        22 => (256, 82),  // IQ2_S
        23 => (256, 136), // IQ4_XS
        24 => (256, 56),  // IQ1_M
        42 => (64, 18),   // Q2_0 (PrismML): f16 d + 16 B of 2-bit
        _ => return None,
    })
}

fn tensor_nbytes(type_id: u32, dims: &[usize]) -> Result<usize> {
    let n: usize = dims.iter().product();
    match type_block(type_id) {
        Some((elems, bytes)) => Ok(n.div_ceil(elems) * bytes),
        None => Err(err(&format!("unsupported tensor type id {type_id}"))),
    }
}

/// Skip a metadata value of the given type without storing it.
fn skip_value(rd: &mut Rd, vtype: u32) -> Result<()> {
    match vtype {
        0..=1 | 7 => rd.take(1).map(|_| ()),
        2..=3 => rd.take(2).map(|_| ()),
        4..=6 => rd.take(4).map(|_| ()),
        10..=12 => rd.take(8).map(|_| ()),
        8 => {
            let n = rd.u64()?;
            if n > 1 << 24 {
                return Err(err("string too long"));
            }
            rd.take(n as usize).map(|_| ())
        }
        9 => {
            let et = rd.u32()?;
            let count = rd.u64()?;
            if count > (1 << 30) {
                return Err(err("array count insane"));
            }
            if et == 8 {
                for _ in 0..count {
                    let n = rd.u64()?;
                    if n > 1 << 24 {
                        return Err(err("array string too long"));
                    }
                    rd.take(n as usize)?;
                }
                Ok(())
            } else {
                let esz = match et {
                    0..=1 | 7 => 1u64,
                    2..=3 => 2,
                    4..=6 => 4,
                    10..=12 => 8,
                    _ => return Err(err("bad array elem type")),
                };
                let total = count
                    .checked_mul(esz)
                    .ok_or_else(|| err("array size overflow"))?;
                rd.take(total as usize).map(|_| ())
            }
        }
        _ => Err(err(&format!("bad value type {vtype}"))),
    }
}

/// Read a metadata value; keys starting with a prefix in `skip_keys` are
/// skipped without storage (tokenizer arrays are huge).
fn read_value(rd: &mut Rd, vtype: u32, store: bool) -> Result<Option<MetaVal>> {
    if !store {
        skip_value(rd, vtype)?;
        return Ok(None);
    }
    let v = match vtype {
        0 => MetaVal::U64(rd.u8()? as u64),
        1 => MetaVal::I64(rd.u8()? as i8 as i64),
        2 => MetaVal::U64(rd.u16()? as u64),
        3 => MetaVal::I64(rd.u16()? as i16 as i64),
        4 => MetaVal::U64(rd.u32()? as u64),
        5 => MetaVal::I64(rd.i32()? as i64),
        6 => MetaVal::F64(rd.f32()? as f64),
        7 => MetaVal::Bool(rd.u8()? != 0),
        8 => MetaVal::Str(rd.string()?),
        9 => {
            let et = rd.u32()?;
            let count = rd.u64()? as usize;
            if count > (1 << 24) {
                return Err(err("stored array count insane"));
            }
            match et {
                8 => {
                    let mut v = Vec::with_capacity(count.min(1 << 20));
                    for _ in 0..count {
                        v.push(rd.string()?);
                    }
                    MetaVal::StrArr(v)
                }
                6 => {
                    let mut v = Vec::with_capacity(count.min(1 << 24));
                    for _ in 0..count {
                        v.push(rd.f32()?);
                    }
                    MetaVal::F32Arr(v)
                }
                4 | 10 => {
                    let mut v = Vec::with_capacity(count.min(1 << 24));
                    for _ in 0..count {
                        v.push(if et == 4 {
                            rd.u32()? as i64
                        } else {
                            rd.u64()? as i64
                        });
                    }
                    MetaVal::I64Arr(v)
                }
                5 | 11 => {
                    let mut v = Vec::with_capacity(count.min(1 << 24));
                    for _ in 0..count {
                        v.push(if et == 5 { rd.i32()? as i64 } else { rd.i64()? });
                    }
                    MetaVal::I64Arr(v)
                }
                _ => {
                    // unsupported elem type: skip remaining elements
                    let esz = match et {
                        0..=1 | 7 => 1usize,
                        2..=3 => 2,
                        _ => return Err(err("bad array elem type")),
                    };
                    rd.take(count * esz)?;
                    MetaVal::I64Arr(vec![])
                }
            }
        }
        10 => MetaVal::U64(rd.u64()?),
        11 => MetaVal::I64(rd.i64()?),
        12 => MetaVal::F64(rd.f64()?),
        _ => return Err(err(&format!("bad value type {vtype}"))),
    };
    Ok(Some(v))
}

impl MiniGguf {
    /// Parse GGUF header, metadata and tensor table; mmap the file.
    /// Tensor payloads are not touched.
    pub fn load(path: &Path) -> Result<Arc<Self>> {
        let file =
            std::fs::File::open(path).map_err(|e| err(&format!("open {}: {e}", path.display())))?;
        let map =
            unsafe { Mmap::map(&file).map_err(|e| err(&format!("mmap {}: {e}", path.display())))? };
        let mut rd = Rd { d: &map, pos: 0 };

        if rd.u32()? != 0x4655_4747 {
            return Err(err("bad magic"));
        }
        let version = rd.u32()?;
        if version != 3 {
            return Err(err(&format!("unsupported version {version}")));
        }
        let tensor_count = rd.u64()? as usize;
        let kv_count = rd.u64()? as usize;
        if tensor_count > 1 << 22 || kv_count > 1 << 20 {
            return Err(err("insane counts"));
        }

        let mut meta = HashMap::with_capacity(kv_count.min(4096));
        for _ in 0..kv_count {
            let key = rd.string()?;
            let vtype = rd.u32()?;
            let store = !key.starts_with("tokenizer.");
            if let Some(v) = read_value(&mut rd, vtype, store)? {
                meta.insert(key, v);
            }
        }

        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = rd.string()?;
            let n_dims = rd.u32()? as usize;
            if n_dims > 4 {
                return Err(err("tensor with >4 dims"));
            }
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                let d = rd.u64()? as usize;
                dims.push(d);
            }
            let type_id = rd.u32()?;
            let offset = rd.u64()?;
            tensors.push(TensorEntry {
                name,
                type_id,
                dims,
                abs_offset: 0, // fixed below
                nbytes: 0,
                // store relative offset in abs_offset for now
            });
            tensors.last_mut().unwrap().abs_offset = offset as usize;
            tensors.last_mut().unwrap().nbytes =
                tensor_nbytes(type_id, &tensors.last().unwrap().dims).unwrap_or(0);
        }

        let data_base = (rd.pos as u64).div_ceil(ALIGN) * ALIGN;
        for t in &mut tensors {
            let abs = data_base
                .checked_add(t.abs_offset as u64)
                .ok_or_else(|| err("tensor offset overflow"))?;
            let end = abs
                .checked_add(t.nbytes as u64)
                .ok_or_else(|| err("tensor end overflow"))?;
            if end > map.len() as u64 {
                return Err(err(&format!(
                    "tensor '{}' out of bounds (end {end} > file {})",
                    t.name,
                    map.len()
                )));
            }
            t.abs_offset = abs as usize;
        }

        let by_name = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();

        Ok(Arc::new(Self {
            map,
            meta,
            tensors,
            by_name,
        }))
    }

    pub fn get(&self, key: &str) -> Option<&MetaVal> {
        self.meta.get(key)
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(|v| v.as_u64())
    }

    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(|v| v.as_f64())
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(|v| v.as_str())
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorEntry> {
        self.by_name
            .get(name)
            .map(|&i| &self.tensors[i])
            .ok_or_else(|| err(&format!("missing tensor '{name}'")))
    }

    /// Raw bytes of a tensor's payload (mmap slice, zero-copy).
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let t = self.tensor(name)?;
        Ok(&self.map[t.abs_offset..t.abs_offset + t.nbytes])
    }

    /// F32 tensor decoded to Vec<f32>.
    /// Read a tensor as f32, dequantizing if the file stored it packed.
    ///
    /// These are the small auxiliary tensors -- norms, conv taps, GDN gate
    /// weights -- that the runtime keeps as plain f32 buffers. Real files do
    /// quantize some of them: `POCKET-35B-Q2_K` ships `ssm_alpha`/`ssm_beta`
    /// as Q2_K, which an f32-only reader cannot load at all.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        let bytes = self.tensor_bytes(name)?;
        crate::dequant::tensor_to_f32(t.type_id, bytes, t.n_elements())
            .map_err(|e| err(&format!("tensor '{name}': {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_geometry_of_common_quants() {
        // Legacy 32-wide quants.
        assert_eq!(type_block(8), Some((32, 34))); // Q8_0
        assert_eq!(type_block(2), Some((32, 18))); // Q4_0
        assert_eq!(type_block(3), Some((32, 20))); // Q4_1

        // K-quants are 256-wide.
        assert_eq!(type_block(12), Some((256, 144))); // Q4_K
        assert_eq!(type_block(14), Some((256, 210))); // Q6_K
        assert_eq!(type_block(10), Some((256, 84))); // Q2_K

        // The PrismML type this reader was originally written for.
        assert_eq!(type_block(42), Some((64, 18))); // Q2_0

        // Unquantized.
        assert_eq!(type_block(0), Some((1, 4)));
        assert_eq!(type_block(1), Some((1, 2)));
        assert_eq!(type_block(30), Some((1, 2)));
    }

    #[test]
    fn nbytes_rolls_partial_blocks_up() {
        // 33 elements of a 32-wide quant still occupies two full blocks.
        assert_eq!(tensor_nbytes(8, &[33]).unwrap(), 2 * 34);
        // Exactly one block.
        assert_eq!(tensor_nbytes(8, &[32]).unwrap(), 34);
    }

    #[test]
    fn nbytes_of_a_k_quant_matrix() {
        // A 4096x2048 Q4_K weight: 8_388_608 elements = 32_768 blocks of 256.
        assert_eq!(tensor_nbytes(12, &[4096, 2048]).unwrap(), 32_768 * 144);
        // 4.5 bits/weight effective, which is the whole point of Q4_K.
        let bits = (32_768 * 144 * 8) as f64 / 8_388_608 as f64;
        assert!(
            (bits - 4.5).abs() < 1e-9,
            "Q4_K is 4.5 bits/weight, got {bits}"
        );
    }

    #[test]
    fn unknown_type_is_an_error_not_a_guess() {
        // Better to refuse than to size a tensor wrongly and read the wrong bytes.
        assert_eq!(type_block(9999), None);
        assert!(tensor_nbytes(9999, &[256]).is_err());
    }
}
