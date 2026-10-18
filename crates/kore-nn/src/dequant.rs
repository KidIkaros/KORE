//! Turn a packed tensor -- or one row of one -- into f32.
//!
//! Every packed GGML format is a sequence of fixed-size blocks, so a slice that
//! starts on a block boundary decodes on its own. That is what lets a whole
//! tensor and a single row of it share one decoder.
//!
//! This is the single place that knows *which* implementation owns a format:
//! [`kore_kernels::k_quant`] for the K-quants it implements (it is the written
//! specification, and the only one that matches llama.cpp's Q3_K scale packing),
//! `oxillama` for the legacy formats whose kernels are validated against the
//! same fixtures, and [`kore_kernels::q2_0`] for KORE's own ternary layout.

use kore_core::KoreError;
use oxillama_gguf::GgufTensorType;

fn err(msg: impl Into<String>) -> KoreError {
    KoreError::StorageError(msg.into())
}

/// Decode `bytes` (a whole tensor, or one block-aligned row of one) into `n`
/// f32 values.
pub fn tensor_to_f32(type_id: u32, bytes: &[u8], n: usize) -> Result<Vec<f32>, KoreError> {
    match type_id {
        0 => {
            let need = n
                .checked_mul(4)
                .ok_or_else(|| err(format!("dequant: {n} f32 values overflow")))?;
            if bytes.len() < need {
                return Err(err(format!(
                    "dequant: f32 needs {need} bytes for {n} values, got {}",
                    bytes.len()
                )));
            }
            Ok(bytes[..need]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        1 => {
            let need = n
                .checked_mul(2)
                .ok_or_else(|| err(format!("dequant: {n} f16 values overflow")))?;
            if bytes.len() < need {
                return Err(err(format!(
                    "dequant: f16 needs {need} bytes for {n} values, got {}",
                    bytes.len()
                )));
            }
            Ok(bytes[..need]
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        }
        kore_kernels::q2_0::Q2_0_TYPE_ID => kore_kernels::q2_0::dequantize_q2_0(bytes, n),
        2 | 6 | 8 => {
            let ty = GgufTensorType::from_u32(type_id)
                .ok_or_else(|| err(format!("dequant: unknown GGML type {type_id}")))?;
            oxillama_quant::dequantize_to_f32(bytes, ty, n)
                .map_err(|e| err(format!("dequant: GGML type {type_id}: {e}")))
        }
        t @ 10..=14 => kore_kernels::k_quant::dequantize(t, bytes, n),
        other => Err(err(format!(
            "dequant: no decoder for GGML type {other} \
             (have F32=0, F16=1, Q4_0=2, Q5_0=6, Q8_0=8, \
             Q2_K=10, Q3_K=11, Q4_K=12, Q5_K=13, Q6_K=14, Q2_0=42)"
        ))),
    }
}
