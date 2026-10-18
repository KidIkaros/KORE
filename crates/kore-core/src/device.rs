use std::fmt;

/// Compute device for tensor storage and operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// CPU with optional SIMD acceleration (AVX2, AVX-512, NEON)
    #[default]
    Cpu,
    /// CUDA GPU with device index
    Cuda(usize),
    /// Vulkan GPU with device index
    Vulkan(usize),
}

impl Device {
    /// Whether this is a CPU device.
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// Whether this is a CUDA device.
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }

    /// Get the CUDA device index, if applicable.
    pub fn cuda_index(&self) -> Option<usize> {
        match self {
            Device::Cuda(idx) => Some(*idx),
            _ => None,
        }
    }

    /// Whether this is a Vulkan device.
    pub fn is_vulkan(&self) -> bool {
        matches!(self, Device::Vulkan(_))
    }

    /// Get the Vulkan device index, if applicable.
    pub fn vulkan_index(&self) -> Option<usize> {
        match self {
            Device::Vulkan(idx) => Some(*idx),
            _ => None,
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(idx) => write!(f, "cuda:{idx}"),
            Device::Vulkan(idx) => write!(f, "vulkan:{idx}"),
        }
    }
}

impl std::str::FromStr for Device {
    type Err = crate::KoreError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let s = s.trim().to_lowercase();
        if s == "cpu" {
            Ok(Device::Cpu)
        } else if s == "auto" || s == "gpu" {
            if get_backend(Device::Vulkan(0)).is_some() {
                Ok(Device::Vulkan(0))
            } else if get_backend(Device::Cuda(0)).is_some() {
                Ok(Device::Cuda(0))
            } else {
                Ok(Device::Cpu)
            }
        } else if let Some(idx_str) = s.strip_prefix("gpu:") {
            let idx = idx_str.parse::<usize>().map_err(|_| {
                crate::KoreError::StorageError(format!("Invalid GPU device index: {}", idx_str))
            })?;
            if get_backend(Device::Vulkan(idx)).is_some() {
                Ok(Device::Vulkan(idx))
            } else if get_backend(Device::Cuda(idx)).is_some() {
                Ok(Device::Cuda(idx))
            } else {
                Ok(Device::Vulkan(idx))
            }
        } else if s == "cuda" {
            Ok(Device::Cuda(0))
        } else if let Some(idx_str) = s.strip_prefix("cuda:") {
            let idx = idx_str.parse::<usize>().map_err(|_| {
                crate::KoreError::StorageError(format!("Invalid CUDA device index: {}", idx_str))
            })?;
            Ok(Device::Cuda(idx))
        } else if s == "vulkan" {
            Ok(Device::Vulkan(0))
        } else if let Some(idx_str) = s.strip_prefix("vulkan:") {
            let idx = idx_str.parse::<usize>().map_err(|_| {
                crate::KoreError::StorageError(format!("Invalid Vulkan device index: {}", idx_str))
            })?;
            Ok(Device::Vulkan(idx))
        } else {
            let available: Vec<String> = available_devices()
                .into_iter()
                .map(|d| d.to_string())
                .collect();
            Err(crate::KoreError::StorageError(format!(
                "Unknown device: '{}'. Supported devices: auto, gpu, cpu, vulkan[:idx], cuda[:idx]. Available on this machine: [{}]",
                s,
                available.join(", ")
            )))
        }
    }
}

/// Device backend abstraction for heterogeneous hardware support (Vulkan, CUDA, WebGPU, etc.).
///
/// Implementors handle memory allocation, H2D/D2H transfers, and operation dispatch
/// for specific compute targets without polluting core entities.
pub trait DeviceBackend: Send + Sync {
    /// Device kind identifier (e.g. "cuda", "vulkan").
    fn device_name(&self) -> &'static str;

    /// Whether this backend supports the given device instance.
    fn supports(&self, device: Device) -> bool;

    /// Copy storage from CPU to this device.
    fn copy_to_device(
        &self,
        storage: &crate::Storage,
        target: Device,
    ) -> crate::Result<crate::Storage>;

    /// Copy storage from this device back to CPU.
    fn copy_to_cpu(&self, storage: &crate::Storage) -> crate::Result<crate::Storage>;

    /// Upload a host byte slice to `target` without first materialising an owned
    /// CPU `Storage`.
    ///
    /// The default implementation is the honest two-step -- own the bytes, then
    /// `copy_to_device` -- which every backend already supports. A backend whose
    /// upload path can read straight from the caller's slice should override it.
    /// This matters for weight streaming: an offloading engine re-uploads whole
    /// matrices on every cache miss, and at that rate the extra full-size copy,
    /// plus the fresh allocation behind it, stops being free. Measured at ~20%
    /// of the step in a miss-heavy MoE configuration.
    fn upload_u8(&self, bytes: &[u8], target: crate::Device) -> crate::Result<crate::Storage> {
        let storage = crate::Storage::from_bytes(crate::DType::U8, bytes.len(), bytes.to_vec())?;
        self.copy_to_device(&storage, target)
    }

    /// Matrix multiplication on device: C = A @ B
    fn matmul(&self, _a: &crate::Tensor, _b: &crate::Tensor) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "matmul not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Elementwise addition: C = A + B
    fn add(&self, _a: &crate::Tensor, _b: &crate::Tensor) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "add not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Elementwise subtraction: C = A - B
    fn sub(&self, _a: &crate::Tensor, _b: &crate::Tensor) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "sub not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Elementwise multiplication: C = A * B
    fn mul(&self, _a: &crate::Tensor, _b: &crate::Tensor) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "mul not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Elementwise division: C = A / B
    fn div(&self, _a: &crate::Tensor, _b: &crate::Tensor) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "div not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Scalar multiplication: C = A * scalar
    fn mul_scalar(&self, _a: &crate::Tensor, _scalar: f32) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "mul_scalar not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Scalar addition: C = A + scalar
    fn add_scalar(&self, _a: &crate::Tensor, _scalar: f32) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "add_scalar not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// RMSNorm: output[row,:] = (input[row,:] / rms) * weight[:]
    fn rms_norm(
        &self,
        _input: &crate::Tensor,
        _weight: &crate::Tensor,
        _eps: f32,
    ) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "rms_norm not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Softmax along the last dimension
    fn softmax(&self, _input: &crate::Tensor) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "softmax not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Make a non-contiguous tensor contiguous on device.
    fn contiguous(&self, _input: &crate::Tensor) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "contiguous not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Slice contiguous rows `[start_row .. start_row + num_rows, :]` from a 2D tensor on device.
    fn slice_rows(
        &self,
        _input: &crate::Tensor,
        _start_row: usize,
        _num_rows: usize,
    ) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "slice_rows not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Write the single row `src` (`[cols]`) into row `row` of the 2-D `dst`
    /// (`[rows, cols]`), in place: the inverse of [`Self::slice_rows`].
    fn scatter_row(
        &self,
        _dst: &crate::Tensor,
        _row: usize,
        _src: &crate::Tensor,
    ) -> crate::Result<()> {
        Err(crate::KoreError::StorageError(format!(
            "scatter_row not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Copy every element of `src` into `dst`, in place on device.
    ///
    /// Needed because some states are updated *in place* (the DeltaNet
    /// recurrence writes its `state` buffer directly), so a caller that has to
    /// rewind a batch needs a real copy of the value before it is overwritten.
    fn copy_buffer(&self, _dst: &crate::Tensor, _src: &crate::Tensor) -> crate::Result<()> {
        Err(crate::KoreError::StorageError(format!(
            "copy_buffer not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Append `new_rows` (`[num_new, cols]`) into preallocated `cache_buf` (`[max_rows, cols]`)
    /// at `row_offset`, returning a tensor view of shape `[row_offset + num_new, cols]`.
    fn kv_cache_append(
        &self,
        _cache_buf: &crate::Tensor,
        _new_rows: &crate::Tensor,
        _row_offset: usize,
    ) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "kv_cache_append not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Normalized Fast Walsh-Hadamard Transform along the last dimension: y = H x / sqrt(D).
    fn hadamard_transform(&self, _input: &crate::Tensor) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "hadamard_transform not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Quantized matrix multiplication on device: C[M, N] = A[M, K] @ dequant(B_packed[N, row_bytes], scales[N])^T
    /// `format`: 0 = Ternary (1.58-bit base-243, 5 trits/byte), 1 = Quaternary (2-bit, 4 values/byte).
    fn quantized_matmul(
        &self,
        _a: &crate::Tensor,
        _b_packed: &crate::Tensor,
        _scales: &crate::Tensor,
        _format: u32,
    ) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "quantized_matmul not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Fused Flash-Decoding attention for small query batches (M <= 4) without causal mask:
    /// output[M, d_v] = softmax(Q[M, d_k] @ K[seq_k, d_k]^T * scale) @ V[seq_k, d_v].
    fn fused_attention_decode(
        &self,
        _q: &crate::Tensor,
        _k: &crate::Tensor,
        _v: &crate::Tensor,
        _scale: f32,
    ) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "fused_attention_decode not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Fused Flash-Decoding Grouped Query Attention (GQA) for small query batches (M <= 4):
    /// Q: [M, num_heads * head_dim]
    /// K: [seq_k, num_kv_heads * head_dim]
    /// V: [seq_k, num_kv_heads * head_dim]
    /// Output: [M, num_heads * head_dim]
    // The argument list is the kernel's binding table: every backend
    // marshals exactly these, and bundling them in a struct would only move
    // the same fields to a call site.
    #[allow(clippy::too_many_arguments)]
    fn fused_attention_decode_gqa(
        &self,
        _q: &crate::Tensor,
        _k: &crate::Tensor,
        _v: &crate::Tensor,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _scale: f32,
    ) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "fused_attention_decode_gqa not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Fused GQA decode attention (quantized; K/V are INT8 + per-token scales).
    ///
    /// `seq_k` is the number of leading keys of `k_q8` this call may attend to.
    /// It normally equals `k_q8`'s leading extent; passing a smaller value lets a
    /// caller attend an *earlier* position against a cache that already holds
    /// later ones, which is how a multi-column verify pass reproduces the
    /// key window (and therefore the reduction) of sequential decode.
    #[allow(clippy::too_many_arguments)]
    fn fused_attention_decode_gqa_q8(
        &self,
        _q: &crate::Tensor,
        _k_q8: &crate::Tensor,
        _v_q8: &crate::Tensor,
        _k_scales: &crate::Tensor,
        _v_scales: &crate::Tensor,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _scale: f32,
        _seq_k: usize,
    ) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "fused_attention_decode_gqa_q8 not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Split-KV GQA decode attention (`seq_q == 1`).
    ///
    /// Same result as [`Self::fused_attention_decode_gqa_q8`], but the KV window
    /// (`seq_k` keys) is sharded across `n_splits * num_heads` workgroups so the
    /// GPU has enough parallel work to hide memory latency. Backends without
    /// sharding fall back to the single-group kernel, ignoring `n_splits`.
    #[allow(clippy::too_many_arguments)]
    fn fused_attention_decode_gqa_q8_split(
        &self,
        q: &crate::Tensor,
        k_q8: &crate::Tensor,
        v_q8: &crate::Tensor,
        k_scales: &crate::Tensor,
        v_scales: &crate::Tensor,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        _n_splits: usize,
        seq_k: usize,
    ) -> crate::Result<crate::Tensor> {
        self.fused_attention_decode_gqa_q8(
            q,
            k_q8,
            v_q8,
            k_scales,
            v_scales,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            seq_k,
        )
    }

    /// Copy quantized INT8 K/V + per-token INT8 scales into the quantized KV cache
    /// and the parallel scale buffers.
    #[allow(clippy::too_many_arguments)]
    fn kv_cache_append_q8(
        &self,
        _k_cache_buf: &crate::Tensor,
        _v_cache_buf: &crate::Tensor,
        _k_scales_buf: &crate::Tensor,
        _v_scales_buf: &crate::Tensor,
        _k_new: &crate::Tensor,
        _v_new: &crate::Tensor,
        _k_scales_new: &crate::Tensor,
        _v_scales_new: &crate::Tensor,
        _row_offset: usize,
    ) -> crate::Result<(crate::Tensor, crate::Tensor)> {
        Err(crate::KoreError::StorageError(format!(
            "kv_cache_append_q8 not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Quantize F32 rows to symmetric INT8 + per-row scales on device.
    /// In: `[rows, cols]` F32 (each row independent).
    /// Out: `([rows, cols]` packed INT8 bytes, `[rows]` F32 scales).
    fn quantize_q8_rows(
        &self,
        _x: &crate::Tensor,
    ) -> crate::Result<(crate::Tensor, crate::Tensor)> {
        Err(crate::KoreError::StorageError(format!(
            "quantize_q8_rows not implemented for backend '{}'",
            self.device_name()
        )))
    }
    /// Rotary Position Embedding (RoPE) on device.
    /// In: [M, num_heads * head_dim]
    /// Output: [M, num_heads * head_dim]
    fn rope(
        &self,
        _x: &crate::Tensor,
        _pos: usize,
        _num_heads: usize,
        _head_dim: usize,
        _rope_theta: f32,
    ) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "rope not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Quantized Q2_K matrix-vector multiplication on device:
    /// y = x @ W^T
    /// x: [M, K] in F32 (where M is 1 during decode)
    /// w_raw: [N * (K / 256) * 84] raw Q2_K packed bytes (or U8 tensor)
    /// n: output dimension (number of rows in W)
    /// k: input dimension (number of columns in W)
    /// Output: [M, N] in F32
    fn gemv_q2_k(
        &self,
        _x: &crate::Tensor,
        _w_raw: &crate::Tensor,
        _n: usize,
        _k: usize,
    ) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "gemv_q2_k not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Quantized Q3_K matrix-vector multiplication on device:
    /// y = x @ W^T
    /// x: [M, K] in F32 (where M is 1 during decode)
    /// w_raw: [N * (K / 256) * 112] padded Q3_K packed bytes (or U8 tensor)
    /// n: output dimension (number of rows in W)
    /// k: input dimension (number of columns in W)
    /// Output: [M, N] in F32
    fn gemv_q3_k(
        &self,
        _x: &crate::Tensor,
        _w_raw: &crate::Tensor,
        _n: usize,
        _k: usize,
    ) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "gemv_q3_k not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Quantized Q4_K matrix-vector multiplication on device:
    /// y = x @ W^T
    /// x: [M, K] in F32 (where M is 1 during decode)
    /// w_raw: [N * (K / 256) * 144] raw Q4_K packed bytes (or U8 tensor)
    /// n: output dimension (number of rows in W)
    /// k: input dimension (number of columns in W)
    /// Output: [M, N] in F32
    fn gemv_q4_k(
        &self,
        _x: &crate::Tensor,
        _w_raw: &crate::Tensor,
        _n: usize,
        _k: usize,
    ) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "gemv_q4_k not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Quantized Q5_K matrix-vector multiplication on device:
    /// y = x @ W^T
    /// x: [M, K] in F32 (where M is 1 during decode)
    /// w_raw: [N * (K / 256) * 176] raw Q5_K packed bytes (or U8 tensor)
    /// n: output dimension (number of rows in W)
    /// k: input dimension (number of columns in W)
    /// Output: [M, N] in F32
    fn gemv_q5_k(
        &self,
        _x: &crate::Tensor,
        _w_raw: &crate::Tensor,
        _n: usize,
        _k: usize,
    ) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "gemv_q5_k not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Quantized Q6_K matrix-vector multiplication on device:
    /// y = x @ W^T
    /// x: [M, K] in F32 (where M is 1 during decode)
    /// w_raw: [N * (K / 256) * 210] raw Q6_K packed bytes (or U8 tensor)
    /// n: output dimension (number of rows in W)
    /// k: input dimension (number of columns in W)
    /// Output: [M, N] in F32
    fn gemv_q6_k(
        &self,
        _x: &crate::Tensor,
        _w_raw: &crate::Tensor,
        _n: usize,
        _k: usize,
    ) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "gemv_q6_k not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Fused Q|K|V projection: one dispatch over a concatenated Q4_K weight
    /// buffer (`w` rows = nq | nk | nv) producing three `[M, n*]` outputs.
    fn gemv_q4_k_split3(
        &self,
        _x: &crate::Tensor,
        _w_raw: &crate::Tensor,
        _nq: usize,
        _nk: usize,
        _nv: usize,
        _k: usize,
    ) -> crate::Result<(crate::Tensor, crate::Tensor, crate::Tensor)> {
        Err(crate::KoreError::StorageError(format!(
            "gemv_q4_k_split3 not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Fused Q|K|V projection over a concatenated Q6_K weight buffer.
    fn gemv_q6_k_split3(
        &self,
        _x: &crate::Tensor,
        _w_raw: &crate::Tensor,
        _nq: usize,
        _nk: usize,
        _nv: usize,
        _k: usize,
    ) -> crate::Result<(crate::Tensor, crate::Tensor, crate::Tensor)> {
        Err(crate::KoreError::StorageError(format!(
            "gemv_q6_k_split3 not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Fused RoPE for Q and K in a single dispatch.
    // The argument list is the kernel's binding table: every backend
    // marshals exactly these, and bundling them in a struct would only move
    // the same fields to a call site.
    #[allow(clippy::too_many_arguments)]
    fn rope_pair(
        &self,
        _q: &crate::Tensor,
        _k: &crate::Tensor,
        _pos: usize,
        _num_heads_q: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_theta: f32,
    ) -> crate::Result<(crate::Tensor, crate::Tensor)> {
        Err(crate::KoreError::StorageError(format!(
            "rope_pair not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Fused KV-cache append for K and V in a single dispatch.
    /// Returns `[row_offset + num_new, cols]` views over the resident caches.
    fn kv_cache_append_pair(
        &self,
        _k_cache: &crate::Tensor,
        _v_cache: &crate::Tensor,
        _k_new: &crate::Tensor,
        _v_new: &crate::Tensor,
        _row_offset: usize,
    ) -> crate::Result<(crate::Tensor, crate::Tensor)> {
        Err(crate::KoreError::StorageError(format!(
            "kv_cache_append_pair not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Fused residual add + RMSNorm: returns `(a + b, rms_norm(a + b) * weight)`.
    fn add_rms_norm(
        &self,
        _a: &crate::Tensor,
        _b: &crate::Tensor,
        _weight: &crate::Tensor,
        _eps: f32,
    ) -> crate::Result<(crate::Tensor, crate::Tensor)> {
        Err(crate::KoreError::StorageError(format!(
            "add_rms_norm not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Fused "pre-op + 512-point signed Hadamard" decode kernel.
    /// `mode`: 0 = add+rms_norm, 1 = rms_norm, 2 = swiglu, 3 = sigmoid_mul.
    /// `had`: when false the sign table and the transform are skipped and
    /// `out_had` is the pre-op value as-is. That is what a model whose weights
    /// are *not* folded through the transform needs: the same fused pre-op and
    /// no rotation. `signs` is only read when `had`.
    /// Returns `(out_res, out_norm, out_had)`; unused outputs are uninitialized.
    /// `weight` is only read for modes 0/1 (pass `signs` for modes 2/3).
    // The argument list is the kernel's binding table: every backend
    // marshals exactly these, and bundling them in a struct would only move
    // the same fields to a call site.
    #[allow(clippy::too_many_arguments)]
    fn had_pre(
        &self,
        _mode: u32,
        _had: bool,
        _a: &crate::Tensor,
        _b: &crate::Tensor,
        _weight: &crate::Tensor,
        _signs: &crate::Tensor,
        _eps: f32,
    ) -> crate::Result<(crate::Tensor, crate::Tensor, crate::Tensor)> {
        Err(crate::KoreError::StorageError(format!(
            "had_pre not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Argmax along the last dimension of a 2D `[rows, cols]` tensor on device, returning `rows` token IDs.
    fn argmax_last_dim(&self, _input: &crate::Tensor) -> crate::Result<Vec<u32>> {
        Err(crate::KoreError::StorageError(format!(
            "argmax_last_dim not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Elementwise SiLU activation: C = silu(A) = A / (1 + exp(-A))
    fn silu(&self, _a: &crate::Tensor) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "silu not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Fused SwiGLU: C = silu(gate) * up
    fn swiglu(&self, _gate: &crate::Tensor, _up: &crate::Tensor) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "swiglu not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Fused SwiGLU from a concatenated `[M, 2 * intermediate_size]` gate+up tensor:
    /// `C[m, i] = silu(gate_up[m, i]) * gate_up[m, intermediate_size + i]`.
    fn swiglu_split(
        &self,
        _gate_up: &crate::Tensor,
        _intermediate_size: usize,
    ) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "swiglu_split not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Elementwise attention gating: `C = A * sigmoid(B)`.
    fn sigmoid_mul(&self, _a: &crate::Tensor, _b: &crate::Tensor) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "sigmoid_mul not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Q2_0 quantized GEMV: `y[m,n] = x[m,k] @ W^T`, W packed [n,k] blocks of
    /// 64 elems / 18 bytes (fp16 scale + 2-bit {-1,0,1,2} codes).
    fn gemv_q2_0(
        &self,
        _x: &crate::Tensor,
        _w_raw: &crate::Tensor,
        _n: usize,
        _k: usize,
    ) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "gemv_q2_0 not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Signed block-wise Hadamard transform.
    /// mode 0: `out = WHT(signs * x) / sqrt(block)` (forward projection input);
    /// mode 1: `out = signs * WHT(x) / sqrt(block)` (embedding restore).
    fn hadamard_signed(
        &self,
        _x: &crate::Tensor,
        _signs: &crate::Tensor,
        _block_size: usize,
        _mode: u32,
    ) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "hadamard_signed not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Gated-DeltaNet depthwise causal conv + SiLU; `hist` mutated in place.
    fn gdn_conv(
        &self,
        _x: &crate::Tensor,
        _w: &crate::Tensor,
        _hist: &crate::Tensor,
        _conv_dim: usize,
        _conv_k: usize,
    ) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "gdn_conv not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Fused DeltaNet gates: `beta = sigmoid(W_b h)`,
    /// `gate = softplus(W_a h + dt_bias) * a`. Returns `(beta[n_v], gate[n_v])`.
    // The argument list is the kernel's binding table: every backend
    // marshals exactly these, and bundling them in a struct would only move
    // the same fields to a call site.
    #[allow(clippy::too_many_arguments)]
    fn gdn_gates(
        &self,
        _h: &crate::Tensor,
        _w_alpha: &crate::Tensor,
        _w_beta: &crate::Tensor,
        _dt_bias: &crate::Tensor,
        _ssm_a: &crate::Tensor,
        _hidden: usize,
        _n_v: usize,
    ) -> crate::Result<(crate::Tensor, crate::Tensor)> {
        Err(crate::KoreError::StorageError(format!(
            "gdn_gates not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Fused DeltaNet recurrent step: L2-normed q/k, in-place delta-rule state
    /// update, output readout, gated RMSNorm with z. Returns `[n_v*d_v]`.
    ///
    /// `n_v` value heads read `n_k` key heads: value head `hv` borrows key head
    /// `hv * n_k / n_v`, so a file whose q/k projections carry fewer heads than
    /// v (GQA over the linear attention) needs no reshuffle of the weights.
    // The argument list is the kernel's binding table: every backend
    // marshals exactly these, and bundling them in a struct would only move
    // the same fields to a call site.
    #[allow(clippy::too_many_arguments)]
    fn gdn_step(
        &self,
        _qkv: &crate::Tensor,
        _z: &crate::Tensor,
        _state: &crate::Tensor,
        _beta: &crate::Tensor,
        _gate: &crate::Tensor,
        _norm_w: &crate::Tensor,
        _n_v: usize,
        _n_k: usize,
        _d_k: usize,
        _d_v: usize,
        _eps: f32,
    ) -> crate::Result<crate::Tensor> {
        Err(crate::KoreError::StorageError(format!(
            "gdn_step not implemented for backend '{}'",
            self.device_name()
        )))
    }

    /// Per-head Q/K RMSNorm + gate split + partial NEOX RoPE (Qwen3.5).
    /// Returns `(q_roped, k_roped, gate)`.
    // The argument list is the kernel's binding table: every backend
    // marshals exactly these, and bundling them in a struct would only move
    // the same fields to a call site.
    #[allow(clippy::too_many_arguments)]
    fn qknorm_rope(
        &self,
        _qg: &crate::Tensor,
        _kk: &crate::Tensor,
        _q_norm: &crate::Tensor,
        _k_norm: &crate::Tensor,
        _n_heads: usize,
        _n_kv: usize,
        _head_dim: usize,
        _rope_dim: usize,
        _pos: usize,
        _theta: f32,
        _eps: f32,
    ) -> crate::Result<(crate::Tensor, crate::Tensor, crate::Tensor)> {
        Err(crate::KoreError::StorageError(format!(
            "qknorm_rope not implemented for backend '{}'",
            self.device_name()
        )))
    }
}

static BACKEND_REGISTRY: std::sync::OnceLock<
    parking_lot::RwLock<Vec<std::sync::Arc<dyn DeviceBackend>>>,
> = std::sync::OnceLock::new();

fn get_registry() -> &'static parking_lot::RwLock<Vec<std::sync::Arc<dyn DeviceBackend>>> {
    BACKEND_REGISTRY.get_or_init(|| parking_lot::RwLock::new(Vec::new()))
}

/// Register a new device backend with the global runtime.
pub fn register_backend(backend: std::sync::Arc<dyn DeviceBackend>) {
    let mut reg = get_registry().write();
    reg.retain(|b| b.device_name() != backend.device_name());
    reg.push(backend);
}

/// Look up the registered backend for a given device.
pub fn get_backend(device: Device) -> Option<std::sync::Arc<dyn DeviceBackend>> {
    let reg = get_registry().read();
    reg.iter().find(|b| b.supports(device)).cloned()
}

/// Returns the names of all registered device backends.
pub fn registered_backend_names() -> Vec<&'static str> {
    let reg = get_registry().read();
    reg.iter().map(|b| b.device_name()).collect()
}

/// Returns the list of currently available compute devices on this system.
pub fn available_devices() -> Vec<Device> {
    let mut devices = vec![Device::Cpu];
    if get_backend(Device::Vulkan(0)).is_some() {
        devices.push(Device::Vulkan(0));
    }
    if get_backend(Device::Cuda(0)).is_some() {
        devices.push(Device::Cuda(0));
    }
    devices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_properties() {
        assert!(Device::Cpu.is_cpu());
        assert!(!Device::Cpu.is_cuda());
        assert!(Device::Cuda(0).is_cuda());
        assert_eq!(Device::Cuda(1).cuda_index(), Some(1));
        assert_eq!(Device::Cpu.cuda_index(), None);

        assert!(Device::Vulkan(0).is_vulkan());
        assert!(!Device::Cpu.is_vulkan());
        assert_eq!(Device::Vulkan(2).vulkan_index(), Some(2));
        assert_eq!(Device::Cpu.vulkan_index(), None);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Device::Cpu), "cpu");
        assert_eq!(format!("{}", Device::Cuda(0)), "cuda:0");
        assert_eq!(format!("{}", Device::Vulkan(0)), "vulkan:0");
    }

    #[test]
    fn test_default() {
        assert_eq!(Device::default(), Device::Cpu);
    }
}
