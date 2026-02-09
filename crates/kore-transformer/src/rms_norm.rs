//! Root Mean Square Layer Normalization (used in LLaMA, Mistral, etc.)

use kore_core::{KoreError, Tensor};

/// RMSNorm: x * weight / sqrt(mean(x^2) + eps)
pub struct RMSNorm {
    /// Learnable scale: [d_model]
    pub weight: Tensor,
    pub eps: f32,
    pub d_model: usize,
}

impl RMSNorm {
    pub fn new(d_model: usize, eps: f32) -> Self {
        let weight = Tensor::ones(&[d_model]);
        Self { weight, eps, d_model }
    }

    /// Normalize input of shape [seq_len, d_model] or [batch, seq_len, d_model].
    /// Normalizes over the last dimension.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, KoreError> {
        let x_data = x.as_f32_slice().ok_or(KoreError::StorageError("expected f32 tensor".into()))?;
        let w_data = self.weight.as_f32_slice().ok_or(KoreError::StorageError("expected f32 weight".into()))?;
        let dims = x.shape().dims().to_vec();
        let d = *dims.last().ok_or(KoreError::StorageError("empty shape".into()))?;

        if d != self.d_model {
            return Err(KoreError::StorageError(
                format!("last dim {} != d_model {}", d, self.d_model),
            ));
        }

        let n_rows = x_data.len() / d;
        let mut out = vec![0.0f32; x_data.len()];

        for row in 0..n_rows {
            let start = row * d;
            let slice = &x_data[start..start + d];

            // Compute RMS
            let ms: f32 = slice.iter().map(|v| v * v).sum::<f32>() / d as f32;
            let rms = (ms + self.eps).sqrt();
            let inv_rms = 1.0 / rms;

            for j in 0..d {
                out[start + j] = slice[j] * inv_rms * w_data[j];
            }
        }

        Ok(Tensor::from_f32(&out, &dims))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_identity() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = Tensor::ones(&[2, 4]);
        let out = norm.forward(&x).unwrap();
        let data = out.as_f32_slice().unwrap();
        // ones normalized: each val = 1 / sqrt(1 + eps) ≈ 1.0
        for &v in data {
            assert!((v - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_rms_norm_shape() {
        let norm = RMSNorm::new(64, 1e-5);
        let x = Tensor::from_f32(&vec![0.5; 8 * 64], &[8, 64]);
        let out = norm.forward(&x).unwrap();
        assert_eq!(out.shape().dims(), &[8, 64]);
    }
}
