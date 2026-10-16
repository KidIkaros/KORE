use rand::Rng;

use kore_core::{DType, Tensor};

use crate::module::Module;

enum LinearKind {
    F32 {
        weight: Option<Tensor>,
        weight_t: Tensor,
    },
    Q4K {
        raw: Tensor,
        in_features: usize,
        out_features: usize,
    },
    Q6K {
        raw: Tensor,
        in_features: usize,
        out_features: usize,
    },
}

/// Packed quantized weight format stored in a `Linear`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PackedQuant {
    Q4K,
    Q6K,
}

/// Fully connected linear layer: y = x @ W^T + b
pub struct Linear {
    kind: LinearKind,
    bias: Option<Tensor>,
    training: bool,
}

impl Linear {
    /// Create a new Linear layer with Xavier uniform initialization.
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        // Xavier uniform: U(-sqrt(6/(in+out)), sqrt(6/(in+out)))
        let limit = (6.0 / (in_features + out_features) as f32).sqrt();
        let mut rng = rand::thread_rng();
        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| rng.gen_range(-limit..limit))
            .collect();

        let mut weight = Tensor::from_f32(&weight_data, &[out_features, in_features]);
        weight.set_requires_grad(true);

        let bias_tensor = if bias {
            let mut b = Tensor::zeros(&[out_features], DType::F32);
            b.set_requires_grad(true);
            Some(b)
        } else {
            None
        };

        let weight_t = weight
            .transpose()
            .expect("Linear weight transpose failed")
            .contiguous();

        Self {
            kind: LinearKind::F32 {
                weight: Some(weight),
                weight_t,
            },
            bias: bias_tensor,
            training: true,
        }
    }

    /// Create a Linear layer from existing weight and optional bias tensors.
    pub fn from_weight(weight: Tensor, bias: Option<Tensor>) -> Self {
        let weight_t = weight
            .transpose()
            .expect("Linear weight transpose failed")
            .contiguous();
        Self {
            kind: LinearKind::F32 {
                weight: Some(weight),
                weight_t,
            },
            bias,
            training: false,
        }
    }

    /// Create an inference-optimized Linear layer directly from a transposed weight matrix `[in_features, out_features]`.
    /// Does not store duplicate non-transposed weight, cutting memory in half.
    pub fn from_weight_transposed(weight_t: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            kind: LinearKind::F32 {
                weight: None,
                weight_t,
            },
            bias,
            training: false,
        }
    }

    /// Create an inference-optimized Linear layer directly from raw Q4_K packed bytes.
    /// Eliminates CPU dequantization and stores weights in 4.5 bits/weight directly in GPU VRAM.
    pub fn from_q4_k(raw_bytes: Vec<u8>, in_features: usize, out_features: usize, bias: Option<Tensor>) -> Self {
        let storage = kore_core::Storage::from_packed_bytes(DType::U8, raw_bytes.len(), raw_bytes);
        let raw = Tensor::from_storage(storage, &[out_features, (in_features / 256) * 144]);
        Self {
            kind: LinearKind::Q4K {
                raw,
                in_features,
                out_features,
            },
            bias,
            training: false,
        }
    }

    /// Create an inference-optimized Linear layer directly from raw Q6_K packed bytes.
    /// Eliminates CPU dequantization and stores weights in 6.5 bits/weight directly in GPU VRAM.
    pub fn from_q6_k(raw_bytes: Vec<u8>, in_features: usize, out_features: usize, bias: Option<Tensor>) -> Self {
        let storage = kore_core::Storage::from_packed_bytes(DType::U8, raw_bytes.len(), raw_bytes);
        let raw = Tensor::from_storage(storage, &[out_features, (in_features / 256) * 210]);
        Self {
            kind: LinearKind::Q6K {
                raw,
                in_features,
                out_features,
            },
            bias,
            training: false,
        }
    }

    /// Move linear layer parameters to the target device.
    pub fn to(&mut self, device: kore_core::Device) -> kore_core::Result<()> {
        match &mut self.kind {
            LinearKind::F32 { weight, weight_t } => {
                if let Some(ref mut w) = weight {
                    *w = w.to(device)?;
                }
                *weight_t = weight_t.to(device)?;
            }
            LinearKind::Q4K { raw, .. } | LinearKind::Q6K { raw, .. } => {
                *raw = raw.to(device)?;
            }
        }
        if let Some(ref mut b) = self.bias {
            *b = b.to(device)?;
        }
        Ok(())
    }

    /// Get the weight tensor.
    pub fn weight(&self) -> &Tensor {
        match &self.kind {
            LinearKind::F32 { weight, weight_t } => weight.as_ref().unwrap_or(weight_t),
            LinearKind::Q4K { raw, .. } | LinearKind::Q6K { raw, .. } => raw,
        }
    }

    /// Get the bias tensor (if present).
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// Input feature dimension.
    pub fn in_features(&self) -> usize {
        match &self.kind {
            LinearKind::F32 { weight_t, .. } => weight_t.shape().dims()[0],
            LinearKind::Q4K { in_features, .. } | LinearKind::Q6K { in_features, .. } => *in_features,
        }
    }

    /// Output feature dimension.
    pub fn out_features(&self) -> usize {
        match &self.kind {
            LinearKind::F32 { weight_t, .. } => weight_t.shape().dims()[1],
            LinearKind::Q4K { out_features, .. } | LinearKind::Q6K { out_features, .. } => *out_features,
        }
    }

    /// Whether this layer has a bias term.
    pub fn has_bias(&self) -> bool {
        self.bias.is_some()
    }

    /// Packed quant kind + raw weight tensor, if this Linear is quantized.
    pub fn packed(&self) -> Option<(PackedQuant, &Tensor)> {
        match &self.kind {
            LinearKind::Q4K { raw, .. } => Some((PackedQuant::Q4K, raw)),
            LinearKind::Q6K { raw, .. } => Some((PackedQuant::Q6K, raw)),
            LinearKind::F32 { .. } => None,
        }
    }

    /// Concatenate the packed weights of three same-kind Linears into one raw
    /// tensor (`rows = a|b|c`). Used to build fused Q|K|V projections.
    /// Returns `None` if kinds differ, any has bias, or inputs aren't on CPU.
    pub fn concat_packed3(a: &Self, b: &Self, c: &Self) -> Option<(Tensor, PackedQuant)> {
        let (ka, ra) = a.packed()?;
        let (kb, rb) = b.packed()?;
        let (kc, rc) = c.packed()?;
        if ka != kb || kb != kc || a.has_bias() || b.has_bias() || c.has_bias()
            || a.in_features() != b.in_features() || b.in_features() != c.in_features()
        {
            return None;
        }
        if !ra.device().is_cpu() || !rb.device().is_cpu() || !rc.device().is_cpu() {
            return None;
        }
        let ba = ra.storage_ref().as_bytes();
        let bb = rb.storage_ref().as_bytes();
        let bc = rc.storage_ref().as_bytes();
        let mut combined = Vec::with_capacity(ba.len() + bb.len() + bc.len());
        combined.extend_from_slice(ba);
        combined.extend_from_slice(bb);
        combined.extend_from_slice(bc);
        let rows = a.out_features() + b.out_features() + c.out_features();
        let row_bytes = ba.len() / a.out_features();
        let storage = kore_core::Storage::from_packed_bytes(DType::U8, combined.len(), combined);
        Some((
            Tensor::from_storage(storage, &[rows, row_bytes]),
            ka,
        ))
    }

    /// Attempt to fuse two Linear layers sharing the same input `[M, K]` into a single
    /// `[M, out_a + out_b]` projection (e.g., `gate_proj + up_proj` in SwiGLU MLP).
    pub fn try_concat(a: &Self, b: &Self) -> Option<Self> {
        if a.has_bias() || b.has_bias() || a.in_features() != b.in_features() {
            return None;
        }
        match (&a.kind, &b.kind) {
            (
                LinearKind::Q4K {
                    raw: raw_a,
                    in_features,
                    out_features: out_a,
                },
                LinearKind::Q4K {
                    raw: raw_b,
                    out_features: out_b,
                    ..
                },
            ) => {
                if !raw_a.device().is_cpu() || !raw_b.device().is_cpu() {
                    return None;
                }
                let bytes_a = raw_a.storage_ref().as_bytes();
                let bytes_b = raw_b.storage_ref().as_bytes();
                let mut combined = Vec::with_capacity(bytes_a.len() + bytes_b.len());
                combined.extend_from_slice(bytes_a);
                combined.extend_from_slice(bytes_b);
                Some(Self::from_q4_k(
                    combined,
                    *in_features,
                    *out_a + *out_b,
                    None,
                ))
            }
            (
                LinearKind::Q6K {
                    raw: raw_a,
                    in_features,
                    out_features: out_a,
                },
                LinearKind::Q6K {
                    raw: raw_b,
                    out_features: out_b,
                    ..
                },
            ) => {
                if !raw_a.device().is_cpu() || !raw_b.device().is_cpu() {
                    return None;
                }
                let bytes_a = raw_a.storage_ref().as_bytes();
                let bytes_b = raw_b.storage_ref().as_bytes();
                let mut combined = Vec::with_capacity(bytes_a.len() + bytes_b.len());
                combined.extend_from_slice(bytes_a);
                combined.extend_from_slice(bytes_b);
                Some(Self::from_q6_k(
                    combined,
                    *in_features,
                    *out_a + *out_b,
                    None,
                ))
            }
            _ => None,
        }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> kore_core::Result<Tensor> {
        let mut output = match &self.kind {
            LinearKind::F32 { weight_t, .. } => input.matmul(weight_t)?,
            LinearKind::Q4K { raw, in_features, out_features } => {
                if input.device().is_cpu() {
                    let raw_bytes = raw.storage_ref().as_bytes();
                    kore_kernels::cpu_matmul::gemv_q4_k_cpu(input, raw_bytes, *out_features, *in_features)?
                } else {
                    let backend = kore_core::device::get_backend(input.device()).ok_or_else(|| {
                        kore_core::KoreError::StorageError(format!("No backend for {}", input.device()))
                    })?;
                    backend.gemv_q4_k(input, raw, *out_features, *in_features)?
                }
            }
            LinearKind::Q6K { raw, in_features, out_features } => {
                if input.device().is_cpu() {
                    let raw_bytes = raw.storage_ref().as_bytes();
                    kore_kernels::cpu_matmul::gemv_q6_k_cpu(input, raw_bytes, *out_features, *in_features)?
                } else {
                    let backend = kore_core::device::get_backend(input.device()).ok_or_else(|| {
                        kore_core::KoreError::StorageError(format!("No backend for {}", input.device()))
                    })?;
                    backend.gemv_q6_k(input, raw, *out_features, *in_features)?
                }
            }
        };

        if let Some(ref bias) = self.bias {
            output = output.add(bias)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        match &self.kind {
            LinearKind::F32 { weight, weight_t } => {
                if let Some(ref w) = weight {
                    params.push(w);
                } else {
                    params.push(weight_t);
                }
            }
            LinearKind::Q4K { raw, .. } | LinearKind::Q6K { raw, .. } => {
                params.push(raw);
            }
        }
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(&str, &Tensor)> {
        let mut params = Vec::new();
        match &self.kind {
            LinearKind::F32 { weight, weight_t } => {
                if let Some(ref w) = weight {
                    params.push(("weight", w));
                } else {
                    params.push(("weight", weight_t));
                }
            }
            LinearKind::Q4K { raw, .. } | LinearKind::Q6K { raw, .. } => {
                params.push(("weight", raw));
            }
        }
        if let Some(ref b) = self.bias {
            params.push(("bias", b));
        }
        params
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_creation() {
        let layer = Linear::new(4, 3, true);
        assert_eq!(layer.weight().shape().dims(), &[3, 4]);
        assert_eq!(layer.bias().unwrap().shape().dims(), &[3]);
        assert!(layer.weight().requires_grad());
    }

    #[test]
    fn test_linear_forward() {
        let layer = Linear::new(3, 2, false);
        let input = Tensor::ones(&[1, 3]);
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 2]);
    }

    #[test]
    fn test_linear_parameters() {
        let layer = Linear::new(4, 3, true);
        assert_eq!(layer.parameters().len(), 2); // weight + bias

        let layer_no_bias = Linear::new(4, 3, false);
        assert_eq!(layer_no_bias.parameters().len(), 1);
    }

    #[test]
    fn test_linear_state_dict() {
        let layer = Linear::new(4, 3, true);
        let sd = layer.state_dict();
        assert!(sd.contains_key("weight"));
        assert!(sd.contains_key("bias"));
    }
}
