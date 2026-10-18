//! Vulkan / GPU compute backend for KORE operations using wgpu.

use crate::buffer::KoreVulkanBuffer;
use crate::error::{Result, VulkanError};
use kore_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

static BACKENDS: OnceLock<Mutex<HashMap<usize, VulkanBackend>>> = OnceLock::new();

/// Opt-in per-op dispatch census, enabled by `KORE_OP_COUNT=1`.
///
/// A decode step is a strict dependency chain of ~300 dispatches, so the cost
/// is `count x exposed-latency` per op. Counting the dispatches by label is what
/// turns "the small ops are a quarter of the token" into a target list, without
/// needing GPU timestamp queries (which this adapter does not expose). The
/// counter is off unless the env var is set; when off the cost is one
/// `OnceLock` read per dispatch.
static OP_COUNTS: OnceLock<Mutex<Vec<(&'static str, u64)>>> = OnceLock::new();
static OP_COUNT_ON: OnceLock<bool> = OnceLock::new();

fn op_counts_mut() -> std::sync::MutexGuard<'static, Vec<(&'static str, u64)>> {
    OP_COUNTS
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn op_count_enabled() -> bool {
    *OP_COUNT_ON.get_or_init(|| {
        std::env::var("KORE_OP_COUNT")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

/// Dispatch counts by label, most frequent first. Empty unless `KORE_OP_COUNT=1`.
pub fn op_counts() -> Vec<(&'static str, u64)> {
    let mut v = op_counts_mut().clone();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    v
}

/// Clear the dispatch census (e.g. after warmup, before the measured steps).
pub fn reset_op_counts() {
    op_counts_mut().clear();
}

fn bump_op_count(label: &'static str) {
    let mut v = op_counts_mut();
    match v.iter_mut().find(|(l, _)| *l == label) {
        Some(e) => e.1 += 1,
        None => v.push((label, 1)),
    }
}

/// Opt-in dispatch ablation, enabled by `KORE_SKIP_OPS=label[,label...]`.
///
/// The census says *how many* dispatches there are; it cannot say what they
/// cost, because this adapter exposes no GPU timestamp queries. Skipping a group
/// and re-measuring the step is the substitute: the chain still runs end to end
/// with the same shapes and the same dependencies (on stale data, so the output
/// is garbage -- this is a timing probe, never a correctness one), and the wall
/// time delta is that group's contribution to the step.
///
/// The special value `all` skips every dispatch, which measures the *host* side
/// of the chain -- the model loop, buffer draws and uniform writes -- with no
/// GPU work and no submissions at all. That is the number that says whether a
/// step is host-bound or GPU-bound. Like the census, the list is read once and
/// costs an `OnceLock` read per dispatch when set.
static SKIP_OPS: OnceLock<Vec<String>> = OnceLock::new();

fn skip_op(label: &'static str) -> bool {
    let list = SKIP_OPS.get_or_init(|| {
        std::env::var("KORE_SKIP_OPS")
            .map(|v| {
                v.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default()
    });
    list.iter().any(|s| s == "all" || s == label)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmUniforms {
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    beta: f32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ElementwiseUniforms {
    numel: u32,
    op_type: u32,
    scalar: f32,
    a_numel: u32,
    b_numel: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QuantizedUniforms {
    m: u32,
    n: u32,
    k: u32,
    format: u32,
    alpha: f32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RMSNormUniforms {
    rows: u32,
    cols: u32,
    eps: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxUniforms {
    rows: u32,
    cols: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TransposeUniforms {
    out_rows: u32,
    out_cols: u32,
    offset: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HadamardUniforms {
    total_blocks: u32,
    block_size: u32,
    scale: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RepackQ6KUniforms {
    blocks_per_row: u32,
    n_rows: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashDecodeUniforms {
    seq_q: u32,
    seq_k: u32,
    d: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashDecodeGqaUniforms {
    seq_q: u32,
    seq_k: u32,
    head_dim: u32,
    scale: f32,
    num_heads: u32,
    num_kv_heads: u32,
    kv_dim: u32,
    q_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RopeUniforms {
    pos: u32,
    num_heads: u32,
    head_dim: u32,
    half_dim: u32,
    rope_theta: f32,
    total_pairs: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ArgmaxUniforms {
    rows: u32,
    cols: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Q4KUniforms {
    m: u32,
    n: u32,
    k: u32,
    blocks_per_row: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Q5KUniforms {
    m: u32,
    n: u32,
    k: u32,
    blocks_per_row: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Q6KUniforms {
    m: u32,
    n: u32,
    k: u32,
    blocks_per_row: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GemvSplit3Uniforms {
    m: u32,
    n: u32,
    k: u32,
    blocks_per_row: u32,
    nq: u32,
    nk: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RopePairUniforms {
    pos: u32,
    head_dim: u32,
    half_dim: u32,
    rope_theta: f32,
    num_heads_q: u32,
    num_heads_k: u32,
    total_pairs_q: u32,
    total_pairs_k: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct KvAppendPairUniforms {
    numel_k: u32,
    numel_v: u32,
    k_dst_off: u32,
    v_dst_off: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct KvAppendQ8Uniforms {
    numel_k: u32,
    dst_k_offset: u32,
    src_k_offset: u32,
    numel_v: u32,
    dst_v_offset: u32,
    src_v_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Q8GqaUniforms {
    seq_q: u32,
    seq_k: u32,
    head_dim: u32,
    scale: f32,
    num_heads: u32,
    num_kv_heads: u32,
    kv_dim: u32,
    q_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Q8GqaSplitUniforms {
    seq_q: u32,
    seq_k: u32,
    head_dim: u32,
    scale: f32,
    num_heads: u32,
    num_kv_heads: u32,
    kv_dim: u32,
    q_dim: u32,
    n_splits: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Q8CombineUniforms {
    head_dim: u32,
    n_splits: u32,
    num_heads: u32,
    _pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QuantQ8Uniforms {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AddRmsNormUniforms {
    rows: u32,
    cols: u32,
    eps: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Q2GemvUniforms {
    m: u32,
    n: u32,
    k: u32,
    blocks_per_row: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HadamardSignedUniforms {
    total_blocks: u32,
    block_size: u32,
    scale: f32,
    mode: u32,
    /// Row width. The transform and its sign pattern act on 512-blocks *within a
    /// row*, so for an `[rows, cols]` input the sign index is row-local
    /// (`block base % cols`). At `rows = 1` that is the identity, `base % cols == base`.
    cols: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HadPreUniforms {
    rows: u32,
    cols: u32,
    blocks: u32,
    mode: u32,
    eps: f32,
    scale: f32,
    /// 1 = apply signs + Walsh-Hadamard, 0 = `out_had` is the pre-op value.
    had: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GdnConvUniforms {
    conv_dim: u32,
    conv_k: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GdnGatesUniforms {
    hidden: u32,
    n_v: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GdnStepUniforms {
    n_v: u32,
    /// Key/value head counts. They differ whenever the q/k projections carry
    /// fewer heads than v; each value head then borrows key head
    /// `hv % n_k`. Padding keeps the buffer a multiple of 16 bytes, as the
    /// uniform address space requires.
    n_k: u32,
    d_k: u32,
    d_v: u32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QkNormRopeUniforms {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    rope_dim: u32,
    /// Position of row 0; row `r` uses `pos + r`.
    pos: u32,
    theta: f32,
    eps: f32,
    rows: u32,
}

struct QueuedDispatch {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group: wgpu::BindGroup,
    workgroups: (u32, u32, u32),
}

/// Batched compute dispatches, uniform ring buffer, and pipeline layout cache state.
struct PendingBatch {
    dispatches: Vec<QueuedDispatch>,
    uniform_cursor: usize,
    uniform_buffers: Vec<Arc<wgpu::Buffer>>,
    uniform_data: Vec<Vec<u8>>,
    layout_cache: HashMap<usize, Arc<wgpu::BindGroupLayout>>,
}

impl PendingBatch {
    /// Encode all queued compute dispatches into a SINGLE `wgpu::ComputePass` on a `CommandEncoder`.
    fn take_encoder(&mut self, device: &wgpu::Device) -> Option<wgpu::CommandEncoder> {
        if self.dispatches.is_empty() {
            return None;
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("kore_batched_encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("kore_merged_compute_pass"),
                timestamp_writes: None,
            });
            for cmd in self.dispatches.drain(..) {
                cpass.set_pipeline(&cmd.pipeline);
                cpass.set_bind_group(0, &cmd.bind_group, &[]);
                cpass.dispatch_workgroups(cmd.workgroups.0, cmd.workgroups.1, cmd.workgroups.2);
            }
        }
        Some(encoder)
    }
}

/// Vulkan / GPU compute backend for KORE.
///
/// Executes pure-Rust WGSL compute shaders via Vulkan on Linux / Metal on macOS / DX12 on Windows.
#[derive(Clone)]
pub struct VulkanBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    device_index: usize,
    gemm_pipeline: Arc<wgpu::ComputePipeline>,
    gemv_pipeline: Arc<wgpu::ComputePipeline>,
    gemv_q4_k_pipeline: Arc<wgpu::ComputePipeline>,
    gemv_q5_k_pipeline: Arc<wgpu::ComputePipeline>,
    gemv_q6_k_pipeline: Arc<wgpu::ComputePipeline>,
    repack_q6_k_pipeline: Arc<wgpu::ComputePipeline>,
    elementwise_pipeline: Arc<wgpu::ComputePipeline>,
    quantized_pipeline: Arc<wgpu::ComputePipeline>,
    rmsnorm_pipeline: Arc<wgpu::ComputePipeline>,
    softmax_pipeline: Arc<wgpu::ComputePipeline>,
    transpose_pipeline: Arc<wgpu::ComputePipeline>,
    hadamard_pipeline: Arc<wgpu::ComputePipeline>,
    flash_decode_pipeline: Arc<wgpu::ComputePipeline>,
    flash_decode_gqa_pipeline: Arc<wgpu::ComputePipeline>,
    rope_pipeline: Arc<wgpu::ComputePipeline>,
    argmax_pipeline: Arc<wgpu::ComputePipeline>,
    gemv_q4_k_split3_pipeline: Arc<wgpu::ComputePipeline>,
    gemv_q6_k_split3_pipeline: Arc<wgpu::ComputePipeline>,
    rope_pair_pipeline: Arc<wgpu::ComputePipeline>,
    kv_append_pair_pipeline: Arc<wgpu::ComputePipeline>,
    kv_cache_append_q8_pipeline: Arc<wgpu::ComputePipeline>,
    flash_decode_gqa_q8_pipeline: Arc<wgpu::ComputePipeline>,
    flash_decode_gqa_q8_split_pipeline: Arc<wgpu::ComputePipeline>,
    flash_decode_gqa_q8_combine_pipeline: Arc<wgpu::ComputePipeline>,
    quantize_q8_rows_pipeline: Arc<wgpu::ComputePipeline>,
    add_rms_norm_pipeline: Arc<wgpu::ComputePipeline>,
    gemv_q2_0_pipeline: Arc<wgpu::ComputePipeline>,
    hadamard_signed_pipeline: Arc<wgpu::ComputePipeline>,
    had_pre_pipeline: Arc<wgpu::ComputePipeline>,
    gdn_conv_pipeline: Arc<wgpu::ComputePipeline>,
    gdn_gates_pipeline: Arc<wgpu::ComputePipeline>,
    gdn_step_pipeline: Arc<wgpu::ComputePipeline>,
    qknorm_rope_pipeline: Arc<wgpu::ComputePipeline>,
    batch: Arc<parking_lot::Mutex<PendingBatch>>,
    /// Lazily padded Q6_K weight buffers (210B -> 224B blocks), keyed by source buffer
    /// pointer; the source Arc is held so its address can never be recycled by the pool.
    q6k_pad_cache: Arc<
        parking_lot::Mutex<
            std::collections::HashMap<usize, (Arc<wgpu::Buffer>, Arc<wgpu::Buffer>)>,
        >,
    >,
}

impl VulkanBackend {
    /// Create new Vulkan backend with default device.
    pub fn new() -> Result<Self> {
        Self::with_device(0)
    }

    /// Create backend with specific device index.
    pub fn with_device(device_index: usize) -> Result<Self> {
        let backends = BACKENDS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut map = backends.lock().unwrap();
        if let Some(backend) = map.get(&device_index) {
            return Ok(backend.clone());
        }

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::empty(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .map_err(|_| VulkanError::NoDevice)?;

        let adapter_info = adapter.get_info();
        eprintln!(
            "[kore-vulkan] adapter: '{}' backend={:?} device_type={:?}",
            adapter_info.name, adapter_info.backend, adapter_info.device_type
        );

        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("KORE Vulkan Device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| VulkanError::Initialization(e.to_string()))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Compile compute pipelines
        let gemm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_gemm_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gemm.wgsl").into()),
        });
        let gemm_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_gemm_pipeline"),
                layout: None,
                module: &gemm_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let gemv_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_gemv_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gemv.wgsl").into()),
        });
        let gemv_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_gemv_pipeline"),
                layout: None,
                module: &gemv_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let elem_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_elementwise_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/elementwise.wgsl").into()),
        });
        let elementwise_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_elementwise_pipeline"),
                layout: None,
                module: &elem_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let quant_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_quantized_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/quantized_matmul.wgsl").into()),
        });
        let quantized_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_quantized_pipeline"),
                layout: None,
                module: &quant_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let rms_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_rmsnorm_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/rms_norm.wgsl").into()),
        });
        let rmsnorm_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_rmsnorm_pipeline"),
                layout: None,
                module: &rms_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let softmax_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_softmax_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/softmax.wgsl").into()),
        });
        let softmax_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_softmax_pipeline"),
                layout: None,
                module: &softmax_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let transpose_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_transpose_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/transpose.wgsl").into()),
        });
        let transpose_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_transpose_pipeline"),
                layout: None,
                module: &transpose_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let hadamard_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_hadamard_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/hadamard.wgsl").into()),
        });
        let hadamard_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_hadamard_pipeline"),
                layout: None,
                module: &hadamard_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let flash_decode_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_flash_decode_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/flash_decode.wgsl").into()),
        });
        let flash_decode_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_flash_decode_pipeline"),
                layout: None,
                module: &flash_decode_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let flash_decode_gqa_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_flash_decode_gqa_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/flash_decode_gqa.wgsl").into()),
        });
        let flash_decode_gqa_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_flash_decode_gqa_pipeline"),
                layout: None,
                module: &flash_decode_gqa_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));
        let flash_decode_gqa_q8_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("kore_flash_decode_gqa_q8_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/flash_decode_gqa_q8.wgsl").into(),
                ),
            });
        let flash_decode_gqa_q8_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_flash_decode_gqa_q8_pipeline"),
                layout: None,
                module: &flash_decode_gqa_q8_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let flash_decode_gqa_q8_split_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("kore_flash_decode_gqa_q8_split_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/flash_decode_gqa_q8_split.wgsl").into(),
                ),
            });
        let flash_decode_gqa_q8_split_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_flash_decode_gqa_q8_split_pipeline"),
                layout: None,
                module: &flash_decode_gqa_q8_split_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let flash_decode_gqa_q8_combine_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("kore_flash_decode_gqa_q8_combine_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/flash_decode_gqa_q8_combine.wgsl").into(),
                ),
            });
        let flash_decode_gqa_q8_combine_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_flash_decode_gqa_q8_combine_pipeline"),
                layout: None,
                module: &flash_decode_gqa_q8_combine_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let rope_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_rope_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/rope.wgsl").into()),
        });
        let rope_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_rope_pipeline"),
                layout: None,
                module: &rope_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let argmax_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_argmax_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/argmax.wgsl").into()),
        });
        let argmax_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_argmax_pipeline"),
                layout: None,
                module: &argmax_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let gemv_q4_k_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_gemv_q4_k_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gemv_q4_k.wgsl").into()),
        });
        let gemv_q4_k_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_gemv_q4_k_pipeline"),
                layout: None,
                module: &gemv_q4_k_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let gemv_q5_k_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_gemv_q5_k_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gemv_q5_k.wgsl").into()),
        });
        let gemv_q5_k_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_gemv_q5_k_pipeline"),
                layout: None,
                module: &gemv_q5_k_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let gemv_q6_k_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_gemv_q6_k_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gemv_q6_k.wgsl").into()),
        });
        let gemv_q6_k_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_gemv_q6_k_pipeline"),
                layout: None,
                module: &gemv_q6_k_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let repack_q6_k_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_repack_q6_k_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/repack_q6_k.wgsl").into()),
        });
        let repack_q6_k_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_repack_q6_k_pipeline"),
                layout: None,
                module: &repack_q6_k_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let gemv_q4_k_split3_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_gemv_q4_k_split3_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gemv_q4_k_split3.wgsl").into()),
        });
        let gemv_q4_k_split3_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_gemv_q4_k_split3_pipeline"),
                layout: None,
                module: &gemv_q4_k_split3_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let gemv_q6_k_split3_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_gemv_q6_k_split3_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gemv_q6_k_split3.wgsl").into()),
        });
        let gemv_q6_k_split3_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_gemv_q6_k_split3_pipeline"),
                layout: None,
                module: &gemv_q6_k_split3_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let rope_pair_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_rope_pair_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/rope_pair.wgsl").into()),
        });
        let rope_pair_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_rope_pair_pipeline"),
                layout: None,
                module: &rope_pair_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let kv_append_pair_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_kv_append_pair_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/kv_append_pair.wgsl").into()),
        });
        let kv_append_pair_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_kv_append_pair_pipeline"),
                layout: None,
                module: &kv_append_pair_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let kv_cache_append_q8_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_kv_cache_append_q8_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/kv_cache_append_q8.wgsl").into(),
            ),
        });
        let kv_cache_append_q8_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_kv_cache_append_q8_pipeline"),
                layout: None,
                module: &kv_cache_append_q8_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let quantize_q8_rows_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_quantize_q8_rows_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/quantize_q8_rows.wgsl").into()),
        });
        let quantize_q8_rows_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_quantize_q8_rows_pipeline"),
                layout: None,
                module: &quantize_q8_rows_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let add_rms_norm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_add_rms_norm_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/add_rms_norm.wgsl").into()),
        });
        let add_rms_norm_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_add_rms_norm_pipeline"),
                layout: None,
                module: &add_rms_norm_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let q2_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_gemv_q2_0_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gemv_q2_0.wgsl").into()),
        });
        let gemv_q2_0_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_gemv_q2_0_pipeline"),
                layout: None,
                module: &q2_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let hads_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_hadamard_signed_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/hadamard_signed.wgsl").into()),
        });
        let hadamard_signed_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_hadamard_signed_pipeline"),
                layout: None,
                module: &hads_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let had_pre_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_had_pre_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/had_pre.wgsl").into()),
        });
        let had_pre_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_had_pre_pipeline"),
                layout: None,
                module: &had_pre_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let gdn_conv_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_gdn_conv_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gdn_conv.wgsl").into()),
        });
        let gdn_conv_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_gdn_conv_pipeline"),
                layout: None,
                module: &gdn_conv_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let gdn_gates_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_gdn_gates_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gdn_gates.wgsl").into()),
        });
        let gdn_gates_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_gdn_gates_pipeline"),
                layout: None,
                module: &gdn_gates_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let gdn_step_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_gdn_step_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gdn_step.wgsl").into()),
        });
        let gdn_step_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_gdn_step_pipeline"),
                layout: None,
                module: &gdn_step_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let qknr_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kore_qknorm_rope_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/qknorm_rope.wgsl").into()),
        });
        let qknorm_rope_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("kore_qknorm_rope_pipeline"),
                layout: None,
                module: &qknr_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let batch = Arc::new(parking_lot::Mutex::new(PendingBatch {
            dispatches: Vec::with_capacity(1024),
            uniform_cursor: 0,
            uniform_buffers: Vec::with_capacity(2048),
            uniform_data: Vec::with_capacity(2048),
            layout_cache: HashMap::with_capacity(16),
        }));

        let backend = Self {
            device,
            queue,
            device_index,
            gemm_pipeline,
            gemv_pipeline,
            gemv_q4_k_pipeline,
            gemv_q5_k_pipeline,
            gemv_q6_k_pipeline,
            repack_q6_k_pipeline,
            elementwise_pipeline,
            quantized_pipeline,
            rmsnorm_pipeline,
            softmax_pipeline,
            transpose_pipeline,
            hadamard_pipeline,
            flash_decode_pipeline,
            flash_decode_gqa_pipeline,
            flash_decode_gqa_q8_pipeline,
            flash_decode_gqa_q8_split_pipeline,
            flash_decode_gqa_q8_combine_pipeline,
            rope_pipeline,
            argmax_pipeline,
            gemv_q4_k_split3_pipeline,
            gemv_q6_k_split3_pipeline,
            rope_pair_pipeline,
            kv_append_pair_pipeline,
            kv_cache_append_q8_pipeline,
            quantize_q8_rows_pipeline,
            add_rms_norm_pipeline,
            gemv_q2_0_pipeline,
            hadamard_signed_pipeline,
            had_pre_pipeline,
            gdn_conv_pipeline,
            gdn_gates_pipeline,
            gdn_step_pipeline,
            qknorm_rope_pipeline,
            batch,
            q6k_pad_cache: Arc::new(parking_lot::Mutex::new(std::collections::HashMap::new())),
        };
        kore_core::device::register_backend(Arc::new(backend.clone()));
        map.insert(device_index, backend.clone());
        Ok(backend)
    }

    /// Queue a compute dispatch into the merged single-ComputePass batch.
    fn record_dispatch(
        &self,
        pipeline: &Arc<wgpu::ComputePipeline>,
        uniform_bytes: &[u8],
        storage_buffers: &[&Arc<wgpu::Buffer>],
        workgroups: (u32, u32, u32),
        label: &'static str,
    ) {
        const RING_SIZE: usize = 2048;
        if op_count_enabled() {
            bump_op_count(label);
        }
        // Ablation probe (see `skip_op`): drop this dispatch before it costs
        // anything, leaving the shapes and dependencies intact.
        if skip_op(label) {
            return;
        }
        let mut batch = self.batch.lock();
        if batch.dispatches.len() >= 48 {
            if let Some(enc) = batch.take_encoder(&self.device) {
                crate::buffer::submit_with_reclaim(&self.queue, enc);
            }
        }

        let idx = batch.uniform_cursor % RING_SIZE;
        batch.uniform_cursor = batch.uniform_cursor.wrapping_add(1);

        let u_size = ((uniform_bytes.len() + 15) & !15).max(32) as u64;
        while batch.uniform_buffers.len() <= idx {
            let buf = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("kore_uniform_ring_buf"),
                size: 32,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            batch.uniform_buffers.push(buf);
            batch.uniform_data.push(Vec::new());
        }
        if batch.uniform_buffers[idx].size() < u_size {
            batch.uniform_buffers[idx] =
                Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: u_size,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            batch.uniform_data[idx].clear();
        }

        let u_buf = Arc::clone(&batch.uniform_buffers[idx]);
        if batch.uniform_data[idx].as_slice() != uniform_bytes {
            self.queue.write_buffer(&u_buf, 0, uniform_bytes);
            batch.uniform_data[idx].clear();
            batch.uniform_data[idx].extend_from_slice(uniform_bytes);
        }

        let pipe_ptr = Arc::as_ptr(pipeline) as usize;
        let layout = batch
            .layout_cache
            .entry(pipe_ptr)
            .or_insert_with(|| Arc::new(pipeline.get_bind_group_layout(0)))
            .clone();

        let mut entries = Vec::with_capacity(1 + storage_buffers.len());
        entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: u_buf.as_entire_binding(),
        });
        for (i, b) in storage_buffers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (i + 1) as u32,
                resource: b.as_entire_binding(),
            });
        }

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &layout,
            entries: &entries,
        });

        batch.dispatches.push(QueuedDispatch {
            pipeline: Arc::clone(pipeline),
            bind_group,
            workgroups,
        });
    }

    /// Flush any pending batched GPU commands to the queue.
    pub fn flush(&self) {
        let mut batch = self.batch.lock();
        if let Some(enc) = batch.take_encoder(&self.device) {
            crate::buffer::submit_with_reclaim(&self.queue, enc);
        }
    }

    /// Flush any pending batched GPU commands and wait for completion on the device.
    pub fn sync(&self) {
        let mut batch = self.batch.lock();
        if let Some(enc) = batch.take_encoder(&self.device) {
            crate::buffer::submit_with_reclaim(&self.queue, enc);
        }
        batch.uniform_cursor = 0;
        drop(batch);
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        crate::buffer::reclaim_in_flight_buffers();
    }

    /// Get the device this backend manages.
    pub fn device(&self) -> Device {
        Device::Vulkan(self.device_index)
    }

    /// Access wgpu Device reference.
    pub fn wgpu_device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Access wgpu Queue reference.
    pub fn wgpu_queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Matrix multiplication: C = A @ B
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(VulkanError::UnsupportedOperation(
                "Only 2D matmul supported on GPU".into(),
            ));
        }

        if a_shape[1] != b_shape[0] {
            return Err(VulkanError::ShapeMismatch {
                op: "matmul".into(),
                lhs: a_shape.to_vec(),
                rhs: b_shape.to_vec(),
            });
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        let a_buf =
            KoreVulkanBuffer::from_storage(a.storage_ref(), a_shape, &self.device, &self.queue)?;
        let b_buf =
            KoreVulkanBuffer::from_storage(b.storage_ref(), b_shape, &self.device, &self.queue)?;

        let out_bytes = (m * n * std::mem::size_of::<f32>()).max(4);
        let c_buffer =
            crate::buffer::alloc_storage_buffer(&self.device, out_bytes, "kore_gemm_out");

        let uniforms = GemmUniforms {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            alpha: 1.0,
            beta: 0.0,
            _pad: [0; 3],
        };

        let (pipeline, workgroups, label) = if (1..=4).contains(&m) {
            (
                &self.gemv_pipeline,
                ((n as u32).div_ceil(64).max(1), 1, 1),
                "gemv_pass",
            )
        } else {
            (
                &self.gemm_pipeline,
                (
                    (n as u32).div_ceil(16).max(1),
                    (m as u32).div_ceil(16).max(1),
                    1,
                ),
                "gemm_pass",
            )
        };

        self.record_dispatch(
            pipeline,
            bytemuck::bytes_of(&uniforms),
            &[a_buf.arc_buffer(), b_buf.arc_buffer(), &c_buffer],
            workgroups,
            label,
        );

        let c_kore_buf = KoreVulkanBuffer::new(c_buffer, DType::F32, vec![m, n]);
        Ok(self.make_tensor(c_kore_buf, &[m, n]))
    }

    fn make_tensor(&self, buf: KoreVulkanBuffer, shape: &[usize]) -> Tensor {
        let numel: usize = shape.iter().product();
        let storage = kore_core::Storage::from_device(DType::F32, numel, Arc::new(buf));
        Tensor::from_storage(storage, shape)
    }

    /// Element-wise addition: C = A + B
    pub fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.elementwise_binary(a, b, 0)
    }

    /// Element-wise multiplication: C = A * B
    pub fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.elementwise_binary(a, b, 1)
    }

    /// Generic elementwise binary dispatch (0=add, 1=mul, 2=sub, 3=div).
    pub fn elementwise_binary(&self, a: &Tensor, b: &Tensor, op_type: u32) -> Result<Tensor> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        let out_shape =
            a_shape
                .broadcast_with(b_shape)
                .ok_or_else(|| VulkanError::ShapeMismatch {
                    op: "elementwise".into(),
                    lhs: a_shape.dims().to_vec(),
                    rhs: b_shape.dims().to_vec(),
                })?;

        let numel = out_shape.numel();
        let a_buf = KoreVulkanBuffer::from_storage(
            a.storage_ref(),
            a.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        let b_buf = KoreVulkanBuffer::from_storage(
            b.storage_ref(),
            b.shape().dims(),
            &self.device,
            &self.queue,
        )?;

        let out_bytes = (numel * std::mem::size_of::<f32>()).max(4);
        let c_buffer =
            crate::buffer::alloc_storage_buffer(&self.device, out_bytes, "kore_elem_out");

        let uniforms = ElementwiseUniforms {
            numel: numel as u32,
            op_type,
            scalar: 0.0,
            a_numel: a.numel() as u32,
            b_numel: b.numel() as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let workgroups = (numel as u32).div_ceil(256).max(1);
        self.record_dispatch(
            &self.elementwise_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[a_buf.arc_buffer(), b_buf.arc_buffer(), &c_buffer],
            (workgroups, 1, 1),
            "elem_compute_pass",
        );

        let c_kore_buf = KoreVulkanBuffer::new(c_buffer, DType::F32, out_shape.dims().to_vec());
        Ok(self.make_tensor(c_kore_buf, out_shape.dims()))
    }

    /// Element-wise operation with scalar: C = A (op) scalar (4=add_scalar, 5=mul_scalar).
    pub fn elementwise_scalar(&self, a: &Tensor, scalar: f32, op_type: u32) -> Result<Tensor> {
        let a_shape = a.shape().dims();
        let numel = a.numel();
        let a_buf =
            KoreVulkanBuffer::from_storage(a.storage_ref(), a_shape, &self.device, &self.queue)?;

        let out_bytes = (numel * std::mem::size_of::<f32>()).max(4);
        let c_buffer =
            crate::buffer::alloc_storage_buffer(&self.device, out_bytes, "kore_elem_scalar_out");

        let uniforms = ElementwiseUniforms {
            numel: numel as u32,
            op_type,
            scalar,
            a_numel: numel as u32,
            b_numel: numel as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let workgroups = (numel as u32).div_ceil(256).max(1);
        self.record_dispatch(
            &self.elementwise_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[a_buf.arc_buffer(), a_buf.arc_buffer(), &c_buffer],
            (workgroups, 1, 1),
            "elem_scalar_compute_pass",
        );

        let c_kore_buf = KoreVulkanBuffer::new(c_buffer, DType::F32, a_shape.to_vec());
        Ok(self.make_tensor(c_kore_buf, a_shape))
    }

    /// Element-wise scalar addition: C = A + scalar
    pub fn add_scalar(&self, a: &Tensor, scalar: f32) -> Result<Tensor> {
        self.elementwise_scalar(a, scalar, 4)
    }

    /// Element-wise scalar multiplication: C = A * scalar
    pub fn mul_scalar(&self, a: &Tensor, scalar: f32) -> Result<Tensor> {
        self.elementwise_scalar(a, scalar, 5)
    }

    /// Element-wise SiLU activation: C = silu(A) = A / (1 + exp(-A))
    pub fn silu(&self, a: &Tensor) -> Result<Tensor> {
        self.elementwise_scalar(a, 0.0, 6)
    }

    /// Fused SwiGLU: C = silu(gate) * up = (gate / (1 + exp(-gate))) * up
    pub fn swiglu(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        self.elementwise_binary(gate, up, 7)
    }

    /// Fused SwiGLU on a concatenated `[M, 2 * intermediate_size]` tensor (`gate_up`),
    /// producing `[M, intermediate_size]` in a single GPU dispatch with zero slicing.
    pub fn swiglu_split(&self, gate_up: &Tensor, intermediate_size: usize) -> Result<Tensor> {
        let dims = gate_up.shape().dims();
        let m = if dims.len() == 2 { dims[0] } else { 1 };
        let numel = m * intermediate_size;
        let in_buf =
            KoreVulkanBuffer::from_storage(gate_up.storage_ref(), dims, &self.device, &self.queue)?;
        let out_bytes = (numel * std::mem::size_of::<f32>()).max(4);
        let c_buffer =
            crate::buffer::alloc_storage_buffer(&self.device, out_bytes, "kore_swiglu_split_out");

        let uniforms = ElementwiseUniforms {
            numel: numel as u32,
            op_type: 9,
            scalar: 0.0,
            a_numel: intermediate_size as u32,
            b_numel: numel as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let workgroups = (numel as u32).div_ceil(256).max(1);
        self.record_dispatch(
            &self.elementwise_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[in_buf.arc_buffer(), in_buf.arc_buffer(), &c_buffer],
            (workgroups, 1, 1),
            "swiglu_split_pass",
        );

        let c_kore_buf = KoreVulkanBuffer::new(c_buffer, DType::F32, vec![m, intermediate_size]);
        Ok(self.make_tensor(c_kore_buf, &[m, intermediate_size]))
    }

    /// Quantized matrix multiplication for ternary (1.58b) and quaternary (2b) formats.
    ///
    /// Supports both row-packed `[N, K]` layout (`BitLinear` / `QuatLinear`, where `b.shape() == [N, K]`
    /// and `scales.numel() == N`) and flat `[K, N]` layout.
    pub fn quantized_matmul(&self, a: &Tensor, b: &Tensor, scales: &Tensor) -> Result<Tensor> {
        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(VulkanError::ShapeMismatch {
                op: "quantized_matmul".into(),
                lhs: a_shape.to_vec(),
                rhs: b_shape.to_vec(),
            });
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let s_len = scales.numel();

        // Determine layout: BitLinear/QuatLinear store weights as row-packed [N, K] with scales [N].
        let (n, is_row_packed) = if b_shape[1] == k && b_shape[0] == s_len {
            (b_shape[0], true)
        } else if b_shape[0] == k && b_shape[1] == s_len {
            (b_shape[1], false)
        } else {
            return Err(VulkanError::ShapeMismatch {
                op: "quantized_matmul".into(),
                lhs: a_shape.to_vec(),
                rhs: b_shape.to_vec(),
            });
        };

        let base_format = match b.dtype() {
            DType::Ternary => 0u32,
            DType::Quaternary => 1u32,
            _ => return Err(VulkanError::UnsupportedDType(b.dtype())),
        };
        let format = base_format | if is_row_packed { 2u32 } else { 0u32 };

        let a_buf =
            KoreVulkanBuffer::from_storage(a.storage_ref(), a_shape, &self.device, &self.queue)?;
        let b_buf =
            KoreVulkanBuffer::from_storage(b.storage_ref(), b_shape, &self.device, &self.queue)?;
        let s_buf = KoreVulkanBuffer::from_storage(
            scales.storage_ref(),
            scales.shape().dims(),
            &self.device,
            &self.queue,
        )?;

        let out_bytes = (m * n * std::mem::size_of::<f32>()).max(4);
        let c_buffer =
            crate::buffer::alloc_storage_buffer(&self.device, out_bytes, "kore_quant_gemm_out");

        let uniforms = QuantizedUniforms {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            format,
            alpha: 1.0,
            _pad: [0; 3],
        };

        let workgroups_x = (n as u32).div_ceil(64).max(1);
        let workgroups_y = (m as u32).div_ceil(4).max(1);
        self.record_dispatch(
            &self.quantized_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[
                a_buf.arc_buffer(),
                b_buf.arc_buffer(),
                s_buf.arc_buffer(),
                &c_buffer,
            ],
            (workgroups_x, workgroups_y, 1),
            "quant_compute_pass",
        );

        let c_kore_buf = KoreVulkanBuffer::new(c_buffer, DType::F32, vec![m, n]);
        Ok(self.make_tensor(c_kore_buf, &[m, n]))
    }

    /// Softmax along the last dimension.
    pub fn softmax(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape().dims();
        let numel = input.numel();
        let cols = shape[shape.len() - 1];
        let rows = numel / cols;

        let in_buf =
            KoreVulkanBuffer::from_storage(input.storage_ref(), shape, &self.device, &self.queue)?;

        let out_bytes = (numel * std::mem::size_of::<f32>()).max(4);
        let out_buffer =
            crate::buffer::alloc_storage_buffer(&self.device, out_bytes, "kore_softmax_out");

        let uniforms = SoftmaxUniforms {
            rows: rows as u32,
            cols: cols as u32,
            _pad: [0; 2],
        };

        let workgroups = (rows as u32).div_ceil(64).max(1);
        self.record_dispatch(
            &self.softmax_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[in_buf.arc_buffer(), &out_buffer],
            (workgroups, 1, 1),
            "softmax_compute_pass",
        );

        let out_kore_buf = KoreVulkanBuffer::new(out_buffer, DType::F32, shape.to_vec());
        Ok(self.make_tensor(out_kore_buf, shape))
    }

    /// RMS Normalization on input with given weights and epsilon.
    pub fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let shape = input.shape().dims();
        let numel = input.numel();
        let cols = shape[shape.len() - 1];
        let rows = numel / cols;

        let in_buf =
            KoreVulkanBuffer::from_storage(input.storage_ref(), shape, &self.device, &self.queue)?;
        let w_buf = KoreVulkanBuffer::from_storage(
            weight.storage_ref(),
            weight.shape().dims(),
            &self.device,
            &self.queue,
        )?;

        let out_bytes = (numel * std::mem::size_of::<f32>()).max(4);
        let out_buffer =
            crate::buffer::alloc_storage_buffer(&self.device, out_bytes, "kore_rmsnorm_out");

        let uniforms = RMSNormUniforms {
            rows: rows as u32,
            cols: cols as u32,
            eps,
            _pad: 0,
        };

        let workgroups = (rows as u32).max(1);
        self.record_dispatch(
            &self.rmsnorm_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[in_buf.arc_buffer(), w_buf.arc_buffer(), &out_buffer],
            (workgroups, 1, 1),
            "rmsnorm_compute_pass",
        );

        let out_kore_buf = KoreVulkanBuffer::new(out_buffer, DType::F32, shape.to_vec());
        Ok(self.make_tensor(out_kore_buf, shape))
    }

    /// 2D Transpose on GPU: output[d, s] = input[s, d]
    pub fn transpose_2d(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape().dims();
        if shape.len() != 2 {
            return Err(VulkanError::UnsupportedOperation(
                "transpose_2d expects 2D tensor".into(),
            ));
        }
        let out_rows = shape[0];
        let out_cols = shape[1];
        let in_buf =
            KoreVulkanBuffer::from_storage(input.storage_ref(), shape, &self.device, &self.queue)?;
        let offset = 0u32;

        let out_bytes = (out_rows * out_cols * 4).max(4);
        let out_buffer = crate::buffer::alloc_storage_buffer(
            &self.device,
            out_bytes,
            "kore_vulkan_transpose_out",
        );

        let uniforms = TransposeUniforms {
            out_rows: out_rows as u32,
            out_cols: out_cols as u32,
            offset,
            _pad: 0,
        };

        let workgroups_x = (out_cols as u32).div_ceil(16).max(1);
        let workgroups_y = (out_rows as u32).div_ceil(16).max(1);
        self.record_dispatch(
            &self.transpose_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[in_buf.arc_buffer(), &out_buffer],
            (workgroups_x, workgroups_y, 1),
            "transpose_pass",
        );

        let out_kore_buf = KoreVulkanBuffer::new(out_buffer, DType::F32, vec![out_rows, out_cols]);
        Ok(self.make_tensor(out_kore_buf, &[out_rows, out_cols]))
    }

    /// Slice contiguous rows `[start_row .. start_row + num_rows, :]` on GPU via compute shader copy.
    pub fn slice_rows(&self, input: &Tensor, start_row: usize, num_rows: usize) -> Result<Tensor> {
        let shape = input.shape().dims();
        if shape.len() != 2 {
            return Err(VulkanError::UnsupportedOperation(
                "slice_rows expects 2D tensor".into(),
            ));
        }
        let cols = shape[1];
        let in_buf =
            KoreVulkanBuffer::from_storage(input.storage_ref(), shape, &self.device, &self.queue)?;

        let numel = num_rows * cols;
        let copy_bytes = (numel * 4).max(4);
        let out_buffer =
            crate::buffer::alloc_storage_buffer(&self.device, copy_bytes, "kore_vulkan_slice_rows");

        let uniforms = ElementwiseUniforms {
            numel: numel as u32,
            op_type: 8,
            scalar: 0.0,
            a_numel: numel as u32,
            b_numel: numel as u32,
            _pad0: 0,
            _pad1: (start_row * cols) as u32,
            _pad2: 0,
        };
        let workgroups = (numel as u32).div_ceil(256).max(1);
        self.record_dispatch(
            &self.elementwise_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[in_buf.arc_buffer(), in_buf.arc_buffer(), &out_buffer],
            (workgroups, 1, 1),
            "slice_rows_pass",
        );

        let out_kore_buf = KoreVulkanBuffer::new(out_buffer, DType::F32, vec![num_rows, cols]);
        Ok(self.make_tensor(out_kore_buf, &[num_rows, cols]))
    }

    /// Write the single row `src` (`[cols]`) into row `row` of the 2-D `dst`
    /// (`[rows, cols]`), in place, on the GPU.
    ///
    /// The inverse of [`Self::slice_rows`]. A batched op that has to be fed one
    /// row at a time -- the DeltaNet recurrent scan, whose state carries across
    /// rows -- uses this to reassemble its per-row outputs into the `[rows, cols]`
    /// layout the batched ops downstream expect, without a host round-trip.
    pub fn scatter_row(&self, dst: &Tensor, row: usize, src: &Tensor) -> Result<()> {
        let shape = dst.shape().dims();
        if shape.len() != 2 {
            return Err(VulkanError::UnsupportedOperation(
                "scatter_row expects a 2D destination".into(),
            ));
        }
        let (rows, cols) = (shape[0], shape[1]);
        if row >= rows || src.numel() != cols {
            return Err(VulkanError::ShapeMismatch {
                op: "scatter_row".into(),
                lhs: vec![rows, cols],
                rhs: vec![row, src.numel()],
            });
        }
        let dst_buf =
            KoreVulkanBuffer::from_storage(dst.storage_ref(), shape, &self.device, &self.queue)?;
        let src_buf = KoreVulkanBuffer::from_storage(
            src.storage_ref(),
            src.shape().dims(),
            &self.device,
            &self.queue,
        )?;

        let uniforms = ElementwiseUniforms {
            numel: cols as u32,
            op_type: 8,
            scalar: 0.0,
            a_numel: cols as u32,
            b_numel: cols as u32,
            _pad0: (row * cols) as u32,
            _pad1: 0,
            _pad2: 0,
        };
        let workgroups = (cols as u32).div_ceil(256).max(1);
        self.record_dispatch(
            &self.elementwise_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[
                src_buf.arc_buffer(),
                src_buf.arc_buffer(),
                dst_buf.arc_buffer(),
            ],
            (workgroups, 1, 1),
            "scatter_row_pass",
        );
        Ok(())
    }

    /// Copy every element of `src` into `dst`, in place, on the GPU.
    ///
    /// A full-tensor memcpy -- the same elementwise `copy_offset` dispatch
    /// [`Self::scatter_row`] uses, with both offsets at zero. It exists because a
    /// recurrent state (`gdn_step`) is updated **in place**, so a caller that
    /// needs to keep the value it had before a batch has to take a real copy.
    pub fn copy_buffer(&self, dst: &Tensor, src: &Tensor) -> Result<()> {
        let numel = dst.numel();
        if src.numel() != numel {
            return Err(VulkanError::ShapeMismatch {
                op: "copy_buffer".into(),
                lhs: vec![numel],
                rhs: vec![src.numel()],
            });
        }
        if numel == 0 {
            return Ok(());
        }
        if numel > u32::MAX as usize {
            return Err(VulkanError::UnsupportedOperation(
                "copy_buffer: numel exceeds the u32 index space".into(),
            ));
        }
        let dst_buf = KoreVulkanBuffer::from_storage(
            dst.storage_ref(),
            dst.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        let src_buf = KoreVulkanBuffer::from_storage(
            src.storage_ref(),
            src.shape().dims(),
            &self.device,
            &self.queue,
        )?;

        let uniforms = ElementwiseUniforms {
            numel: numel as u32,
            op_type: 8,
            scalar: 0.0,
            a_numel: numel as u32,
            b_numel: numel as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let workgroups = (numel as u32).div_ceil(256);
        self.record_dispatch(
            &self.elementwise_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[
                src_buf.arc_buffer(),
                src_buf.arc_buffer(),
                dst_buf.arc_buffer(),
            ],
            (workgroups, 1, 1),
            "copy_buffer_pass",
        );
        Ok(())
    }

    /// Append `new_rows` (`[num_new, cols]`) into preallocated `cache_buf` (`[max_rows, cols]`)
    /// at `row_offset` on GPU, returning a zero-copy view of shape `[row_offset + num_new, cols]`.
    pub fn kv_cache_append(
        &self,
        cache_buf: &Tensor,
        new_rows: &Tensor,
        row_offset: usize,
    ) -> Result<Tensor> {
        let cache_shape = cache_buf.shape().dims();
        let new_shape = new_rows.shape().dims();
        if cache_shape.len() != 2 || new_shape.len() != 2 || cache_shape[1] != new_shape[1] {
            return Err(VulkanError::ShapeMismatch {
                op: "kv_cache_append".into(),
                lhs: cache_shape.to_vec(),
                rhs: new_shape.to_vec(),
            });
        }
        let num_new = new_shape[0];
        let cols = new_shape[1];
        let total_rows = row_offset + num_new;

        let dst_buf = KoreVulkanBuffer::from_storage(
            cache_buf.storage_ref(),
            cache_shape,
            &self.device,
            &self.queue,
        )?;
        let src_buf = KoreVulkanBuffer::from_storage(
            new_rows.storage_ref(),
            new_shape,
            &self.device,
            &self.queue,
        )?;

        let numel = num_new * cols;
        if numel > 0 {
            let uniforms = ElementwiseUniforms {
                numel: numel as u32,
                op_type: 8,
                scalar: 0.0,
                a_numel: numel as u32,
                b_numel: numel as u32,
                _pad0: (row_offset * cols) as u32,
                _pad1: 0,
                _pad2: 0,
            };
            let workgroups = (numel as u32).div_ceil(256).max(1);
            self.record_dispatch(
                &self.elementwise_pipeline,
                bytemuck::bytes_of(&uniforms),
                &[
                    src_buf.arc_buffer(),
                    src_buf.arc_buffer(),
                    dst_buf.arc_buffer(),
                ],
                (workgroups, 1, 1),
                "kv_cache_append_pass",
            );
        }

        let view_buf = KoreVulkanBuffer::new(
            Arc::clone(dst_buf.arc_buffer()),
            DType::F32,
            vec![total_rows, cols],
        );
        Ok(self.make_tensor(view_buf, &[total_rows, cols]))
    }

    /// Q8 (quantized INT8) KV-cache append: writes a single-token U8 K/V plus
    /// per-token f32 scales into the U8 caches and scale buffers. The decode
    /// shader dequantizes K/V inline (see `fused_attention_decode_gqa_q8`).
    pub fn kv_cache_append_q8(
        &self,
        k_cache_buf: &Tensor,
        v_cache_buf: &Tensor,
        k_scales_buf: &Tensor,
        v_scales_buf: &Tensor,
        k_new: &Tensor,
        v_new: &Tensor,
        k_scales_new: &Tensor,
        v_scales_new: &Tensor,
        row_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let kc_shape = k_cache_buf.shape().dims();
        let vc_shape = v_cache_buf.shape().dims();
        let kn_shape = k_new.shape().dims();
        let vn_shape = v_new.shape().dims();
        let ks_shape = k_scales_new.shape().dims();
        let vs_shape = v_scales_new.shape().dims();
        if kc_shape.len() != 2
            || vc_shape.len() != 2
            || kn_shape.len() != 2
            || vn_shape.len() != 2
            || ks_shape.len() != 1
            || vs_shape.len() != 1
            || kn_shape[1] != kc_shape[1]
            || vn_shape[1] != vc_shape[1]
            || kn_shape[0] != vn_shape[0]
            || kn_shape[0] != ks_shape[0]
        {
            return Err(VulkanError::ShapeMismatch {
                op: "kv_cache_append_q8".into(),
                lhs: kc_shape.to_vec(),
                rhs: kn_shape.to_vec(),
            });
        }

        let num_new = kn_shape[0];
        let cols_k = kc_shape[1];
        let cols_v = vc_shape[1];

        // INT8 K/V are bound as packed `array<u32>` in WGSL (naga has no 8-bit scalar
        // type), so every row width must be a multiple of 4 to keep byte offsets aligned.
        if cols_k % 4 != 0 || cols_v % 4 != 0 {
            return Err(VulkanError::UnsupportedOperation(
                "kv_cache_append_q8 requires 4-aligned K/V row widths (packed INT8)".into(),
            ));
        }

        let kc_buf = KoreVulkanBuffer::from_storage(
            k_cache_buf.storage_ref(),
            kc_shape,
            &self.device,
            &self.queue,
        )?;
        let vc_buf = KoreVulkanBuffer::from_storage(
            v_cache_buf.storage_ref(),
            vc_shape,
            &self.device,
            &self.queue,
        )?;
        let kn_buf = KoreVulkanBuffer::from_storage(
            k_new.storage_ref(),
            kn_shape,
            &self.device,
            &self.queue,
        )?;
        let vn_buf = KoreVulkanBuffer::from_storage(
            v_new.storage_ref(),
            vn_shape,
            &self.device,
            &self.queue,
        )?;

        let numel_k = (num_new * cols_k) as u32;
        let numel_v = (num_new * cols_v) as u32;
        if numel_k + numel_v > 0 {
            let uniforms = KvAppendQ8Uniforms {
                numel_k,
                dst_k_offset: (row_offset * cols_k) as u32,
                src_k_offset: 0,
                numel_v,
                dst_v_offset: (row_offset * cols_v) as u32,
                src_v_offset: 0,
            };
            // Dispatch over u32 words (4 bytes each), not bytes.
            let workgroups = numel_k.max(numel_v).div_ceil(4).div_ceil(256).max(1);
            self.record_dispatch(
                &self.kv_cache_append_q8_pipeline,
                bytemuck::bytes_of(&uniforms),
                &[
                    kn_buf.arc_buffer(),
                    vn_buf.arc_buffer(),
                    kc_buf.arc_buffer(),
                    vc_buf.arc_buffer(),
                ],
                (workgroups, 1, 1),
                "kv_cache_append_q8_pass",
            );
        }

        let ks_buf = KoreVulkanBuffer::from_storage(
            k_scales_buf.storage_ref(),
            ks_shape,
            &self.device,
            &self.queue,
        )?;
        let vs_buf = KoreVulkanBuffer::from_storage(
            v_scales_buf.storage_ref(),
            vs_shape,
            &self.device,
            &self.queue,
        )?;
        let kns = KoreVulkanBuffer::from_storage(
            k_scales_new.storage_ref(),
            ks_shape,
            &self.device,
            &self.queue,
        )?;
        let vns = KoreVulkanBuffer::from_storage(
            v_scales_new.storage_ref(),
            vs_shape,
            &self.device,
            &self.queue,
        )?;

        if num_new > 0 {
            let eu = ElementwiseUniforms {
                numel: num_new as u32,
                op_type: 8,
                scalar: 0.0,
                a_numel: num_new as u32,
                b_numel: num_new as u32,
                _pad0: row_offset as u32,
                _pad1: 0,
                _pad2: 0,
            };
            let workgroups = (num_new as u32 + 255) / 256;
            self.record_dispatch(
                &self.elementwise_pipeline,
                bytemuck::bytes_of(&eu),
                &[kns.arc_buffer(), kns.arc_buffer(), ks_buf.arc_buffer()],
                (workgroups, 1, 1),
                "kv_cache_append_q8_k_scales",
            );
            let eu = ElementwiseUniforms {
                numel: num_new as u32,
                op_type: 8,
                scalar: 0.0,
                a_numel: num_new as u32,
                b_numel: num_new as u32,
                _pad0: row_offset as u32,
                _pad1: 0,
                _pad2: 0,
            };
            self.record_dispatch(
                &self.elementwise_pipeline,
                bytemuck::bytes_of(&eu),
                &[vns.arc_buffer(), vns.arc_buffer(), vs_buf.arc_buffer()],
                (workgroups, 1, 1),
                "kv_cache_append_q8_v_scales",
            );
        }

        let total_rows = row_offset + num_new;
        let k_view = KoreVulkanBuffer::new(
            Arc::clone(kc_buf.arc_buffer()),
            DType::U8,
            vec![total_rows, cols_k],
        );
        let v_view = KoreVulkanBuffer::new(
            Arc::clone(vc_buf.arc_buffer()),
            DType::U8,
            vec![total_rows, cols_v],
        );
        Ok((
            self.make_tensor(k_view, &[total_rows, cols_k]),
            self.make_tensor(v_view, &[total_rows, cols_v]),
        ))
    }

    /// Quantize F32 rows to symmetric INT8 **on the GPU** (no host round-trip).
    ///
    /// Returns packed INT8 bytes `[rows, cols]` (4 bytes per u32 word, little-endian)
    /// and per-row scales `[rows]`. This mirrors the CPU recipe used previously:
    /// `scale = absmax(row) / 127`, `q = round(x / scale)` clamped to `[-127, 127]`,
    /// stored offset-binary as `q + 128`. The output feeds `kv_cache_append_q8`.
    pub fn quantize_q8_rows(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let shape = x.shape().dims();
        if shape.len() != 2 {
            return Err(VulkanError::ShapeMismatch {
                op: "quantize_q8_rows".into(),
                lhs: shape.to_vec(),
                rhs: vec![0, 0],
            });
        }
        let rows = shape[0];
        let cols = shape[1];
        if cols % 4 != 0 {
            return Err(VulkanError::UnsupportedOperation(
                "quantize_q8_rows requires 4-aligned row width (packed INT8)".into(),
            ));
        }

        let q_buffer = crate::buffer::alloc_storage_buffer(
            &self.device,
            (rows * cols).max(4),
            "kore_quantize_q8_out",
        );
        let s_buffer = crate::buffer::alloc_storage_buffer(
            &self.device,
            (rows * 4).max(4),
            "kore_quantize_q8_scales",
        );

        if rows > 0 && cols > 0 {
            let in_buf =
                KoreVulkanBuffer::from_storage(x.storage_ref(), shape, &self.device, &self.queue)?;
            let uniforms = QuantQ8Uniforms {
                rows: rows as u32,
                cols: cols as u32,
                _pad0: 0,
                _pad1: 0,
            };
            self.record_dispatch(
                &self.quantize_q8_rows_pipeline,
                bytemuck::bytes_of(&uniforms),
                &[in_buf.arc_buffer(), &q_buffer, &s_buffer],
                (rows as u32, 1, 1),
                "quantize_q8_rows_pass",
            );
        }

        let q_view = KoreVulkanBuffer::new(q_buffer, DType::U8, vec![rows, cols]);
        let s_view = KoreVulkanBuffer::new(s_buffer, DType::F32, vec![rows]);
        let q_storage = kore_core::Storage::from_device(DType::U8, rows * cols, Arc::new(q_view));
        let s_storage = kore_core::Storage::from_device(DType::F32, rows, Arc::new(s_view));
        Ok((
            Tensor::from_storage(q_storage, &[rows, cols]),
            Tensor::from_storage(s_storage, &[rows]),
        ))
    }

    /// Apply normalized Fast Walsh-Hadamard Transform (`fwht`) along the last dimension on GPU.
    pub fn hadamard_transform(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape().dims();
        if shape.is_empty() {
            return Ok(input.clone());
        }
        let cols = *shape.last().unwrap();
        let numel = input.numel();
        let block_size = (1usize << cols.trailing_zeros()).min(256);
        if block_size <= 1 || numel == 0 {
            return Ok(input.clone());
        }
        let total_blocks = numel / block_size;
        let scale = 1.0f32 / (block_size as f32).sqrt();

        let in_buf =
            KoreVulkanBuffer::from_storage(input.storage_ref(), shape, &self.device, &self.queue)?;
        let out_bytes = (numel * std::mem::size_of::<f32>()).max(4);
        let out_buffer =
            crate::buffer::alloc_storage_buffer(&self.device, out_bytes, "kore_hadamard_out");

        let uniforms = HadamardUniforms {
            total_blocks: total_blocks as u32,
            block_size: block_size as u32,
            scale,
            _pad: 0,
        };

        self.record_dispatch(
            &self.hadamard_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[in_buf.arc_buffer(), &out_buffer],
            (total_blocks as u32, 1, 1),
            "hadamard_compute_pass",
        );

        let out_kore_buf = KoreVulkanBuffer::new(out_buffer, DType::F32, shape.to_vec());
        Ok(self.make_tensor(out_kore_buf, shape))
    }

    /// Fused Flash-Decoding Attention (`softmax(Q @ K^T * scale) @ V`) directly over row-major `K, V` (`[seq_k, d]`).
    pub fn fused_attention_decode(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
    ) -> Result<Tensor> {
        let q_shape = q.shape().dims();
        let k_shape = k.shape().dims();
        let v_shape = v.shape().dims();
        if q_shape.len() != 2 || k_shape.len() != 2 || v_shape.len() != 2 {
            return Err(VulkanError::UnsupportedOperation(
                "fused_attention_decode requires 2D Q, K, V tensors".into(),
            ));
        }
        let seq_q = q_shape[0];
        let d = q_shape[1];
        let seq_k = k_shape[0];
        if k_shape[1] != d || v_shape[0] != seq_k || v_shape[1] != d || d > 256 {
            return Err(VulkanError::ShapeMismatch {
                op: "fused_attention_decode".into(),
                lhs: q_shape.to_vec(),
                rhs: k_shape.to_vec(),
            });
        }

        let q_buf =
            KoreVulkanBuffer::from_storage(q.storage_ref(), q_shape, &self.device, &self.queue)?;
        let k_buf =
            KoreVulkanBuffer::from_storage(k.storage_ref(), k_shape, &self.device, &self.queue)?;
        let v_buf =
            KoreVulkanBuffer::from_storage(v.storage_ref(), v_shape, &self.device, &self.queue)?;

        let out_bytes = (seq_q * d * std::mem::size_of::<f32>()).max(4);
        let out_buffer =
            crate::buffer::alloc_storage_buffer(&self.device, out_bytes, "kore_flash_decode_out");

        let uniforms = FlashDecodeUniforms {
            seq_q: seq_q as u32,
            seq_k: seq_k as u32,
            d: d as u32,
            scale,
        };

        self.record_dispatch(
            &self.flash_decode_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[
                q_buf.arc_buffer(),
                k_buf.arc_buffer(),
                v_buf.arc_buffer(),
                &out_buffer,
            ],
            (seq_q as u32, 1, 1),
            "flash_decode_compute_pass",
        );

        let out_kore_buf = KoreVulkanBuffer::new(out_buffer, DType::F32, vec![seq_q, d]);
        Ok(self.make_tensor(out_kore_buf, &[seq_q, d]))
    }

    /// Fused Flash-Decoding Grouped Query Attention (GQA):
    /// Computes softmax(Q @ K^T * scale) @ V directly over resident row-major K and V
    /// for all query heads in parallel in a single GPU dispatch, with zero CPU transfers.
    pub fn fused_attention_decode_gqa(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Tensor> {
        let q_shape = q.shape().dims();
        let k_shape = k.shape().dims();
        let v_shape = v.shape().dims();
        if q_shape.len() != 2 || k_shape.len() != 2 || v_shape.len() != 2 {
            return Err(VulkanError::UnsupportedOperation(
                "fused_attention_decode_gqa requires 2D Q, K, V tensors".into(),
            ));
        }
        let seq_q = q_shape[0];
        let q_dim = q_shape[1];
        let seq_k = k_shape[0];
        let kv_dim = k_shape[1];

        if q_dim != num_heads * head_dim || kv_dim != num_kv_heads * head_dim {
            return Err(VulkanError::ShapeMismatch {
                op: "fused_attention_decode_gqa".into(),
                lhs: q_shape.to_vec(),
                rhs: k_shape.to_vec(),
            });
        }

        let q_buf =
            KoreVulkanBuffer::from_storage(q.storage_ref(), q_shape, &self.device, &self.queue)?;
        let k_buf =
            KoreVulkanBuffer::from_storage(k.storage_ref(), k_shape, &self.device, &self.queue)?;
        let v_buf =
            KoreVulkanBuffer::from_storage(v.storage_ref(), v_shape, &self.device, &self.queue)?;

        let out_bytes = (seq_q * q_dim * std::mem::size_of::<f32>()).max(4);
        let out_buffer = crate::buffer::alloc_storage_buffer(
            &self.device,
            out_bytes,
            "kore_flash_decode_gqa_out",
        );

        let uniforms = FlashDecodeGqaUniforms {
            seq_q: seq_q as u32,
            seq_k: seq_k as u32,
            head_dim: head_dim as u32,
            scale,
            num_heads: num_heads as u32,
            num_kv_heads: num_kv_heads as u32,
            kv_dim: kv_dim as u32,
            q_dim: q_dim as u32,
        };

        self.record_dispatch(
            &self.flash_decode_gqa_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[
                q_buf.arc_buffer(),
                k_buf.arc_buffer(),
                v_buf.arc_buffer(),
                &out_buffer,
            ],
            (num_heads as u32, seq_q as u32, 1),
            "flash_decode_gqa_compute_pass",
        );

        let out_kore_buf = KoreVulkanBuffer::new(out_buffer, DType::F32, vec![seq_q, q_dim]);
        Ok(self.make_tensor(out_kore_buf, &[seq_q, q_dim]))
    }

    /// Fused GQA decode attention (quantized; K/V are INT8 + per-token scales).
    ///
    /// Reads K/V as 8-bit storage arrays and dequantizes per-thread with the
    /// per-token scale buffers — the KV read traffic is 4x lower than F32.
    pub fn fused_attention_decode_gqa_q8(
        &self,
        q: &Tensor,
        k_q8: &Tensor,
        v_q8: &Tensor,
        k_scales: &Tensor,
        v_scales: &Tensor,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        seq_k: usize,
    ) -> Result<Tensor> {
        let q_shape = q.shape().dims();
        let k_shape = k_q8.shape().dims();
        let v_shape = v_q8.shape().dims();
        let ks_shape = k_scales.shape().dims();
        let vs_shape = v_scales.shape().dims();
        if q_shape.len() != 2
            || k_shape.len() != 2
            || v_shape.len() != 2
            || ks_shape.len() != 1
            || vs_shape.len() != 1
        {
            return Err(VulkanError::UnsupportedOperation(
                "fused_attention_decode_gqa_q8 requires 2D Q/K/V + 1D scale buffers".into(),
            ));
        }
        let seq_q = q_shape[0];
        let q_dim = q_shape[1];
        let kv_dim = k_shape[1];
        // `seq_k` may be narrowed to attend an earlier position of a cache that
        // already holds later keys, but never widened past what is resident.
        if seq_k == 0 || seq_k > k_shape[0] || seq_k > ks_shape[0] || seq_k > vs_shape[0] {
            return Err(VulkanError::ShapeMismatch {
                op: "fused_attention_decode_gqa_q8 seq_k".into(),
                lhs: vec![seq_k],
                rhs: k_shape.to_vec(),
            });
        }

        if q_dim != num_heads * head_dim || kv_dim != num_kv_heads * head_dim {
            return Err(VulkanError::ShapeMismatch {
                op: "fused_attention_decode_gqa_q8".into(),
                lhs: q_shape.to_vec(),
                rhs: k_shape.to_vec(),
            });
        }
        if kv_dim % 4 != 0 {
            return Err(VulkanError::UnsupportedOperation(
                "fused_attention_decode_gqa_q8 requires 4-aligned kv_dim (packed INT8)".into(),
            ));
        }

        let q_buf =
            KoreVulkanBuffer::from_storage(q.storage_ref(), q_shape, &self.device, &self.queue)?;
        let k_buf =
            KoreVulkanBuffer::from_storage(k_q8.storage_ref(), k_shape, &self.device, &self.queue)?;
        let v_buf =
            KoreVulkanBuffer::from_storage(v_q8.storage_ref(), v_shape, &self.device, &self.queue)?;
        let ks_buf = KoreVulkanBuffer::from_storage(
            k_scales.storage_ref(),
            ks_shape,
            &self.device,
            &self.queue,
        )?;
        let vs_buf = KoreVulkanBuffer::from_storage(
            v_scales.storage_ref(),
            vs_shape,
            &self.device,
            &self.queue,
        )?;

        let out_bytes = (seq_q * q_dim * std::mem::size_of::<f32>()).max(4);
        let out_buffer = crate::buffer::alloc_storage_buffer(
            &self.device,
            out_bytes,
            "kore_flash_decode_gqa_q8_out",
        );

        let uniforms = Q8GqaUniforms {
            seq_q: seq_q as u32,
            seq_k: seq_k as u32,
            head_dim: head_dim as u32,
            scale,
            num_heads: num_heads as u32,
            num_kv_heads: num_kv_heads as u32,
            kv_dim: kv_dim as u32,
            q_dim: q_dim as u32,
        };

        self.record_dispatch(
            &self.flash_decode_gqa_q8_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[
                q_buf.arc_buffer(),
                k_buf.arc_buffer(),
                v_buf.arc_buffer(),
                ks_buf.arc_buffer(),
                vs_buf.arc_buffer(),
                &out_buffer,
            ],
            (num_heads as u32, seq_q as u32, 1),
            "flash_decode_gqa_q8_compute_pass",
        );

        let out_kore_buf = KoreVulkanBuffer::new(out_buffer, DType::F32, vec![seq_q, q_dim]);
        Ok(self.make_tensor(out_kore_buf, &[seq_q, q_dim]))
    }

    /// Split-KV variant of [`Self::fused_attention_decode_gqa_q8`] for `seq_q == 1`.
    ///
    /// Shards the KV window across `n_splits` workgroups per head
    /// (grid `(num_heads, n_splits, 1)`) and merges the shards in a second,
    /// tiny pass. The single-group kernel has only `num_heads` workgroups to hide
    /// memory latency with, which is why decode attention falls to ~1.4 GB/s at
    /// long context; sharding gives it `num_heads * n_splits`.
    ///
    /// `head_dim` must be a multiple of 32 (the shard kernel's shared `Q` staging
    /// and vectorized V accumulation require it) and `n_splits >= 1`.
    pub fn fused_attention_decode_gqa_q8_split(
        &self,
        q: &Tensor,
        k_q8: &Tensor,
        v_q8: &Tensor,
        k_scales: &Tensor,
        v_scales: &Tensor,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        n_splits: usize,
        seq_k: usize,
    ) -> Result<Tensor> {
        let q_shape = q.shape().dims();
        let k_shape = k_q8.shape().dims();
        let v_shape = v_q8.shape().dims();
        let ks_shape = k_scales.shape().dims();
        let vs_shape = v_scales.shape().dims();
        if q_shape.len() != 2
            || k_shape.len() != 2
            || v_shape.len() != 2
            || ks_shape.len() != 1
            || vs_shape.len() != 1
        {
            return Err(VulkanError::UnsupportedOperation(
                "fused_attention_decode_gqa_q8_split requires 2D Q/K/V + 1D scale buffers".into(),
            ));
        }
        let seq_q = q_shape[0];
        let q_dim = q_shape[1];
        let kv_dim = k_shape[1];
        // See the single-group kernel: `seq_k` narrows the window so a batched
        // caller can reproduce an earlier position's shard partition exactly.
        if seq_k == 0 || seq_k > k_shape[0] {
            return Err(VulkanError::ShapeMismatch {
                op: "fused_attention_decode_gqa_q8_split seq_k".into(),
                lhs: vec![seq_k],
                rhs: k_shape.to_vec(),
            });
        }
        if seq_q != 1 {
            return Err(VulkanError::UnsupportedOperation(format!(
                "fused_attention_decode_gqa_q8_split is decode-only (seq_q == 1), got {seq_q}"
            )));
        }
        if q_dim != num_heads * head_dim || kv_dim != num_kv_heads * head_dim {
            return Err(VulkanError::ShapeMismatch {
                op: "fused_attention_decode_gqa_q8_split".into(),
                lhs: q_shape.to_vec(),
                rhs: k_shape.to_vec(),
            });
        }
        if head_dim == 0 || head_dim % 32 != 0 || head_dim > 256 {
            return Err(VulkanError::UnsupportedOperation(format!(
                "fused_attention_decode_gqa_q8_split needs head_dim % 32 == 0 and head_dim <= 256 \
                 (shared-Q staging), got {head_dim}"
            )));
        }
        if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
            return Err(VulkanError::UnsupportedOperation(format!(
                "num_heads {num_heads} must be a multiple of num_kv_heads {num_kv_heads}"
            )));
        }
        if n_splits == 0 || seq_k == 0 {
            return Err(VulkanError::UnsupportedOperation(
                "fused_attention_decode_gqa_q8_split needs n_splits >= 1 and seq_k >= 1".into(),
            ));
        }
        // A negative `scale` is the caller's signal to apply causal masking (see
        // the single-group kernel); the shard kernel cannot, so refuse rather
        // than silently return unmasked attention.
        if scale < 0.0 {
            return Err(VulkanError::UnsupportedOperation(
                "fused_attention_decode_gqa_q8_split does not support causal masking".into(),
            ));
        }

        let q_buf =
            KoreVulkanBuffer::from_storage(q.storage_ref(), q_shape, &self.device, &self.queue)?;
        let k_buf =
            KoreVulkanBuffer::from_storage(k_q8.storage_ref(), k_shape, &self.device, &self.queue)?;
        let v_buf =
            KoreVulkanBuffer::from_storage(v_q8.storage_ref(), v_shape, &self.device, &self.queue)?;
        let ks_buf = KoreVulkanBuffer::from_storage(
            k_scales.storage_ref(),
            ks_shape,
            &self.device,
            &self.queue,
        )?;
        let vs_buf = KoreVulkanBuffer::from_storage(
            v_scales.storage_ref(),
            vs_shape,
            &self.device,
            &self.queue,
        )?;

        let n_shards = num_heads * n_splits;
        // Both streams are scratch: they are bound directly and never wrapped in
        // a tensor, so they must recycle through `PooledBuffer` or every call
        // destroys and re-creates an `n_shards * head_dim` allocation.
        let ml_buffer = crate::buffer::PooledBuffer::new(crate::buffer::alloc_storage_buffer(
            &self.device,
            n_shards * 2 * std::mem::size_of::<f32>(),
            "kore_flash_decode_gqa_q8_split_ml",
        ));
        let acc_buffer = crate::buffer::PooledBuffer::new(crate::buffer::alloc_storage_buffer(
            &self.device,
            n_shards * head_dim * std::mem::size_of::<f32>(),
            "kore_flash_decode_gqa_q8_split_acc",
        ));
        let out_buffer = crate::buffer::alloc_storage_buffer(
            &self.device,
            (q_dim * std::mem::size_of::<f32>()).max(4),
            "kore_flash_decode_gqa_q8_split_out",
        );

        let su = Q8GqaSplitUniforms {
            seq_q: seq_q as u32,
            seq_k: seq_k as u32,
            head_dim: head_dim as u32,
            scale,
            num_heads: num_heads as u32,
            num_kv_heads: num_kv_heads as u32,
            kv_dim: kv_dim as u32,
            q_dim: q_dim as u32,
            n_splits: n_splits as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        self.record_dispatch(
            &self.flash_decode_gqa_q8_split_pipeline,
            bytemuck::bytes_of(&su),
            &[
                q_buf.arc_buffer(),
                k_buf.arc_buffer(),
                v_buf.arc_buffer(),
                ks_buf.arc_buffer(),
                vs_buf.arc_buffer(),
                ml_buffer.arc(),
                acc_buffer.arc(),
            ],
            (num_heads as u32, n_splits as u32, 1),
            "flash_decode_gqa_q8_split_pass",
        );

        let cu = Q8CombineUniforms {
            head_dim: head_dim as u32,
            n_splits: n_splits as u32,
            num_heads: num_heads as u32,
            _pad0: 0,
        };
        self.record_dispatch(
            &self.flash_decode_gqa_q8_combine_pipeline,
            bytemuck::bytes_of(&cu),
            &[ml_buffer.arc(), acc_buffer.arc(), &out_buffer],
            (num_heads as u32, 1, 1),
            "flash_decode_gqa_q8_combine_pass",
        );

        let out_kore_buf = KoreVulkanBuffer::new(out_buffer, DType::F32, vec![seq_q, q_dim]);
        Ok(self.make_tensor(out_kore_buf, &[seq_q, q_dim]))
    }

    /// Rotary Position Embedding (RoPE) on device.
    pub fn rope(
        &self,
        x: &Tensor,
        pos: usize,
        num_heads: usize,
        head_dim: usize,
        rope_theta: f32,
    ) -> Result<Tensor> {
        let x_shape = x.shape().dims();
        let total_dim = num_heads * head_dim;
        let numel = x.numel();
        if numel % total_dim != 0 {
            return Err(VulkanError::ShapeMismatch {
                op: "rope".into(),
                lhs: x_shape.to_vec(),
                rhs: vec![num_heads, head_dim],
            });
        }
        let half_dim = head_dim / 2;
        let total_pairs = (numel / head_dim) * half_dim;

        let in_buf =
            KoreVulkanBuffer::from_storage(x.storage_ref(), x_shape, &self.device, &self.queue)?;

        let out_bytes = (numel * std::mem::size_of::<f32>()).max(4);
        let out_buffer =
            crate::buffer::alloc_storage_buffer(&self.device, out_bytes, "kore_rope_out");

        let uniforms = RopeUniforms {
            pos: pos as u32,
            num_heads: num_heads as u32,
            head_dim: head_dim as u32,
            half_dim: half_dim as u32,
            rope_theta,
            total_pairs: total_pairs as u32,
            _pad0: 0,
            _pad1: 0,
        };

        let workgroups = ((total_pairs as u32) + 63) / 64;
        self.record_dispatch(
            &self.rope_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[in_buf.arc_buffer(), &out_buffer],
            (workgroups, 1, 1),
            "rope_compute_pass",
        );

        let out_kore_buf = KoreVulkanBuffer::new(out_buffer, DType::F32, x_shape.to_vec());
        Ok(self.make_tensor(out_kore_buf, x_shape))
    }

    /// Parallel GPU argmax reduction along the last dimension, returning only 4 bytes (`u32`) per row.
    pub fn argmax_last_dim(&self, logits: &Tensor) -> Result<Vec<u32>> {
        let shape = logits.shape().dims();
        if shape.is_empty() {
            return Ok(vec![0]);
        }
        let cols = *shape.last().unwrap();
        let rows = logits.numel() / cols;

        let in_buf =
            KoreVulkanBuffer::from_storage(logits.storage_ref(), shape, &self.device, &self.queue)?;
        let out_bytes = (rows * std::mem::size_of::<u32>()).max(4);
        let out_buffer =
            crate::buffer::alloc_storage_buffer(&self.device, out_bytes, "kore_argmax_out");

        let uniforms = ArgmaxUniforms {
            rows: rows as u32,
            cols: cols as u32,
            _pad: [0; 2],
        };

        self.record_dispatch(
            &self.argmax_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[in_buf.arc_buffer(), &out_buffer],
            (rows as u32, 1, 1),
            "argmax_compute_pass",
        );

        let mut batch = self.batch.lock();
        batch.uniform_cursor = 0;
        let pending_encoder = batch.take_encoder(&self.device);
        let out_kore_buf = KoreVulkanBuffer::new(out_buffer, DType::I32, vec![rows]);
        let cpu_storage =
            out_kore_buf.to_storage_with_encoder(&self.device, &self.queue, pending_encoder)?;
        drop(batch);
        let indices: Vec<u32> = bytemuck::cast_slice::<u8, u32>(cpu_storage.as_bytes()).to_vec();
        Ok(indices)
    }

    /// Quantized Q4_K matrix-vector multiplication: y = x @ W^T
    /// x: [M, K] in F32
    /// w_raw: [N * (K/256) * 144] in U8
    /// n: output dimension (number of rows in W)
    /// k: input dimension (number of columns in W)
    pub fn gemv_q4_k(&self, x: &Tensor, w_raw: &Tensor, n: usize, k: usize) -> Result<Tensor> {
        let x_shape = x.shape().dims();
        // A 1-D input is a single row: `m` is 1, not its length. Reading
        // `x_shape[0]` as `m` would turn a `[k]` activation vector into a
        // `[k, k]` output. (`gemv_q2_0` has always done it this way.)
        let m = if x_shape.len() == 2 { x_shape[0] } else { 1 };
        let k_in = x.numel() / m;
        if k_in != k {
            return Err(VulkanError::ShapeMismatch {
                op: "gemv_q4_k".into(),
                lhs: vec![m, k_in],
                rhs: vec![n, k],
            });
        }

        let blocks_per_row = (k / 256) as u32;
        let uniforms = Q4KUniforms {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            blocks_per_row,
        };

        let x_buf =
            KoreVulkanBuffer::from_storage(x.storage_ref(), x_shape, &self.device, &self.queue)?;
        let w_buf = KoreVulkanBuffer::from_storage(
            w_raw.storage_ref(),
            w_raw.shape().dims(),
            &self.device,
            &self.queue,
        )?;

        let out_bytes = (m * n * std::mem::size_of::<f32>()).max(4);
        let y_buffer =
            crate::buffer::alloc_storage_buffer(&self.device, out_bytes, "kore_gemv_q4_k_out");

        let workgroups_x = (n as u32).div_ceil(4).max(1);
        let workgroups_y = (m as u32).div_ceil(4).max(1);
        self.record_dispatch(
            &self.gemv_q4_k_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[x_buf.arc_buffer(), w_buf.arc_buffer(), &y_buffer],
            (workgroups_x, workgroups_y, 1),
            "gemv_q4_k_compute_pass",
        );

        let y_kore_buf = KoreVulkanBuffer::new(y_buffer, DType::F32, vec![m, n]);
        Ok(self.make_tensor(y_kore_buf, &[m, n]))
    }

    /// Quantized Q5_K matrix-vector multiplication: y = x @ W^T
    /// x: [M, K] in F32
    /// w_raw: [N * (K/256) * 176] in U8
    /// n: output dimension (number of rows in W)
    /// k: input dimension (number of columns in W)
    pub fn gemv_q5_k(&self, x: &Tensor, w_raw: &Tensor, n: usize, k: usize) -> Result<Tensor> {
        let x_shape = x.shape().dims();
        // A 1-D input is a single row: `m` is 1, not its length. Reading
        // `x_shape[0]` as `m` would turn a `[k]` activation vector into a
        // `[k, k]` output. (`gemv_q2_0` has always done it this way.)
        let m = if x_shape.len() == 2 { x_shape[0] } else { 1 };
        let k_in = x.numel() / m;
        if k_in != k {
            return Err(VulkanError::ShapeMismatch {
                op: "gemv_q5_k".into(),
                lhs: vec![m, k_in],
                rhs: vec![n, k],
            });
        }

        let blocks_per_row = (k / 256) as u32;
        let uniforms = Q5KUniforms {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            blocks_per_row,
        };

        let x_buf =
            KoreVulkanBuffer::from_storage(x.storage_ref(), x_shape, &self.device, &self.queue)?;
        let w_buf = KoreVulkanBuffer::from_storage(
            w_raw.storage_ref(),
            w_raw.shape().dims(),
            &self.device,
            &self.queue,
        )?;

        let out_bytes = (m * n * std::mem::size_of::<f32>()).max(4);
        let y_buffer =
            crate::buffer::alloc_storage_buffer(&self.device, out_bytes, "kore_gemv_q5_k_out");

        let workgroups_x = (n as u32).div_ceil(4).max(1);
        let workgroups_y = (m as u32).div_ceil(4).max(1);
        self.record_dispatch(
            &self.gemv_q5_k_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[x_buf.arc_buffer(), w_buf.arc_buffer(), &y_buffer],
            (workgroups_x, workgroups_y, 1),
            "gemv_q5_k_compute_pass",
        );

        let y_kore_buf = KoreVulkanBuffer::new(y_buffer, DType::F32, vec![m, n]);
        Ok(self.make_tensor(y_kore_buf, &[m, n]))
    }

    /// Quantized Q6_K matrix-vector multiplication: y = x @ W^T
    /// x: [M, K] in F32
    /// w_raw: [N * (K/256) * 210] in U8
    /// n: output dimension (number of rows in W)
    /// k: input dimension (number of columns in W)
    pub fn gemv_q6_k(&self, x: &Tensor, w_raw: &Tensor, n: usize, k: usize) -> Result<Tensor> {
        let x_shape = x.shape().dims();
        // A 1-D input is a single row: `m` is 1, not its length. Reading
        // `x_shape[0]` as `m` would turn a `[k]` activation vector into a
        // `[k, k]` output. (`gemv_q2_0` has always done it this way.)
        let m = if x_shape.len() == 2 { x_shape[0] } else { 1 };
        let k_in = x.numel() / m;
        if k_in != k {
            return Err(VulkanError::ShapeMismatch {
                op: "gemv_q6_k".into(),
                lhs: vec![m, k_in],
                rhs: vec![n, k],
            });
        }

        let blocks_per_row = (k / 256) as u32;
        let uniforms = Q6KUniforms {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            blocks_per_row,
        };

        let x_buf =
            KoreVulkanBuffer::from_storage(x.storage_ref(), x_shape, &self.device, &self.queue)?;
        let w_buf = KoreVulkanBuffer::from_storage(
            w_raw.storage_ref(),
            w_raw.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        let padded_w = self.q6k_padded(w_buf.arc_buffer(), n, blocks_per_row);

        let out_bytes = (m * n * std::mem::size_of::<f32>()).max(4);
        let y_buffer =
            crate::buffer::alloc_storage_buffer(&self.device, out_bytes, "kore_gemv_q6_k_out");

        let workgroups_x = (n as u32).div_ceil(4).max(1);
        let workgroups_y = (m as u32).div_ceil(4).max(1);
        self.record_dispatch(
            &self.gemv_q6_k_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[x_buf.arc_buffer(), &padded_w, &y_buffer],
            (workgroups_x, workgroups_y, 1),
            "gemv_q6_k_compute_pass",
        );

        let y_kore_buf = KoreVulkanBuffer::new(y_buffer, DType::F32, vec![m, n]);
        Ok(self.make_tensor(y_kore_buf, &[m, n]))
    }

    /// Get or create the 224B-block padded version of a packed Q6_K weight buffer.
    /// Repack is recorded once per source buffer into the pending submission;
    /// the source `Arc` is pinned in the cache so its pointer can't be recycled.
    fn q6k_padded(
        &self,
        w_arc: &Arc<wgpu::Buffer>,
        n: usize,
        blocks_per_row: u32,
    ) -> Arc<wgpu::Buffer> {
        let key = Arc::as_ptr(w_arc) as usize;
        let cached = {
            self.q6k_pad_cache
                .lock()
                .get(&key)
                .map(|(_, p)| Arc::clone(p))
        };
        match cached {
            Some(p) => p,
            None => {
                let padded = crate::buffer::alloc_storage_buffer(
                    &self.device,
                    n * blocks_per_row as usize * 224,
                    "kore_q6k_padded",
                );
                let rp = RepackQ6KUniforms {
                    blocks_per_row,
                    n_rows: n as u32,
                    _pad: [0; 2],
                };
                self.record_dispatch(
                    &self.repack_q6_k_pipeline,
                    bytemuck::bytes_of(&rp),
                    &[w_arc, &padded],
                    ((n as u32).div_ceil(8).max(1), 1, 1),
                    "repack_q6_k_pass",
                );
                self.q6k_pad_cache
                    .lock()
                    .insert(key, (Arc::clone(w_arc), Arc::clone(&padded)));
                padded
            }
        }
    }

    /// Fused Q|K|V projection over a single concatenated Q4_K weight buffer
    /// (`w` rows = nq | nk | nv). One dispatch produces three `[m, n*]` outputs.
    pub fn gemv_q4_k_split3(
        &self,
        x: &Tensor,
        w_raw: &Tensor,
        nq: usize,
        nk: usize,
        nv: usize,
        k: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        self.gemv_split3_dispatch(
            x,
            w_raw,
            nq,
            nk,
            nv,
            k,
            &self.gemv_q4_k_split3_pipeline,
            "q4_k",
        )
    }

    /// Fused Q|K|V projection over a concatenated Q6_K weight buffer (padded
    /// to 224B blocks on first use, like `gemv_q6_k`).
    pub fn gemv_q6_k_split3(
        &self,
        x: &Tensor,
        w_raw: &Tensor,
        nq: usize,
        nk: usize,
        nv: usize,
        k: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        self.gemv_split3_dispatch(
            x,
            w_raw,
            nq,
            nk,
            nv,
            k,
            &self.gemv_q6_k_split3_pipeline,
            "q6_k",
        )
    }

    fn gemv_split3_dispatch(
        &self,
        x: &Tensor,
        w_raw: &Tensor,
        nq: usize,
        nk: usize,
        nv: usize,
        k: usize,
        pipeline: &Arc<wgpu::ComputePipeline>,
        label: &str,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let x_shape = x.shape().dims();
        // A 1-D input is a single row: `m` is 1, not its length. Reading
        // `x_shape[0]` as `m` would turn a `[k]` activation vector into a
        // `[k, k]` output. (`gemv_q2_0` has always done it this way.)
        let m = if x_shape.len() == 2 { x_shape[0] } else { 1 };
        let k_in = x.numel() / m;
        let n = nq + nk + nv;
        if k_in != k {
            return Err(VulkanError::ShapeMismatch {
                op: format!("gemv_{label}_split3"),
                lhs: vec![m, k_in],
                rhs: vec![n, k],
            });
        }

        let blocks_per_row = (k / 256) as u32;
        let uniforms = GemvSplit3Uniforms {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            blocks_per_row,
            nq: nq as u32,
            nk: nk as u32,
            _pad0: 0,
            _pad1: 0,
        };

        let x_buf =
            KoreVulkanBuffer::from_storage(x.storage_ref(), x_shape, &self.device, &self.queue)?;
        let w_buf = KoreVulkanBuffer::from_storage(
            w_raw.storage_ref(),
            w_raw.shape().dims(),
            &self.device,
            &self.queue,
        )?;

        // Q6_K weights go through the padded-block repack; Q4_K bind raw.
        let w_arc = if label == "q6_k" {
            self.q6k_padded(w_buf.arc_buffer(), n, blocks_per_row)
        } else {
            Arc::clone(w_buf.arc_buffer())
        };

        let yq_buffer =
            crate::buffer::alloc_storage_buffer(&self.device, (m * nq * 4).max(4), "kore_split3_q");
        let yk_buffer =
            crate::buffer::alloc_storage_buffer(&self.device, (m * nk * 4).max(4), "kore_split3_k");
        let yv_buffer =
            crate::buffer::alloc_storage_buffer(&self.device, (m * nv * 4).max(4), "kore_split3_v");

        let workgroups_x = (n as u32).div_ceil(4).max(1);
        let workgroups_y = (m as u32).div_ceil(4).max(1);
        self.record_dispatch(
            pipeline,
            bytemuck::bytes_of(&uniforms),
            &[
                x_buf.arc_buffer(),
                &w_arc,
                &yq_buffer,
                &yk_buffer,
                &yv_buffer,
            ],
            (workgroups_x, workgroups_y, 1),
            "gemv_split3_compute_pass",
        );

        Ok((
            self.make_tensor(
                KoreVulkanBuffer::new(yq_buffer, DType::F32, vec![m, nq]),
                &[m, nq],
            ),
            self.make_tensor(
                KoreVulkanBuffer::new(yk_buffer, DType::F32, vec![m, nk]),
                &[m, nk],
            ),
            self.make_tensor(
                KoreVulkanBuffer::new(yv_buffer, DType::F32, vec![m, nv]),
                &[m, nv],
            ),
        ))
    }

    /// Fused RoPE for Q and K in a single dispatch.
    /// q: [M, num_heads_q * head_dim], k: [M, num_kv_heads * head_dim].
    pub fn rope_pair(
        &self,
        q: &Tensor,
        k: &Tensor,
        pos: usize,
        num_heads_q: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_theta: f32,
    ) -> Result<(Tensor, Tensor)> {
        let q_shape = q.shape().dims();
        let k_shape = k.shape().dims();
        let half_dim = head_dim / 2;
        let total_pairs_q = (q.numel() / head_dim) * half_dim;
        let total_pairs_k = (k.numel() / head_dim) * half_dim;
        if q.numel() % (num_heads_q * head_dim) != 0 || k.numel() % (num_kv_heads * head_dim) != 0 {
            return Err(VulkanError::ShapeMismatch {
                op: "rope_pair".into(),
                lhs: q_shape.to_vec(),
                rhs: k_shape.to_vec(),
            });
        }

        let q_buf =
            KoreVulkanBuffer::from_storage(q.storage_ref(), q_shape, &self.device, &self.queue)?;
        let k_buf =
            KoreVulkanBuffer::from_storage(k.storage_ref(), k_shape, &self.device, &self.queue)?;
        let q_out = crate::buffer::alloc_storage_buffer(
            &self.device,
            (q.numel() * 4).max(4),
            "kore_rope_pair_q",
        );
        let k_out = crate::buffer::alloc_storage_buffer(
            &self.device,
            (k.numel() * 4).max(4),
            "kore_rope_pair_k",
        );

        let uniforms = RopePairUniforms {
            pos: pos as u32,
            head_dim: head_dim as u32,
            half_dim: half_dim as u32,
            rope_theta,
            num_heads_q: num_heads_q as u32,
            num_heads_k: num_kv_heads as u32,
            total_pairs_q: total_pairs_q as u32,
            total_pairs_k: total_pairs_k as u32,
        };

        let workgroups = ((total_pairs_q + total_pairs_k) as u32).div_ceil(64).max(1);
        self.record_dispatch(
            &self.rope_pair_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[q_buf.arc_buffer(), k_buf.arc_buffer(), &q_out, &k_out],
            (workgroups, 1, 1),
            "rope_pair_compute_pass",
        );

        Ok((
            self.make_tensor(
                KoreVulkanBuffer::new(q_out, DType::F32, q_shape.to_vec()),
                q_shape,
            ),
            self.make_tensor(
                KoreVulkanBuffer::new(k_out, DType::F32, k_shape.to_vec()),
                k_shape,
            ),
        ))
    }

    /// Fused KV-cache append for K and V in a single dispatch.
    /// Returns views `[row_offset + num_new, cols_*]` over the resident caches.
    pub fn kv_cache_append_pair(
        &self,
        k_cache: &Tensor,
        v_cache: &Tensor,
        k_new: &Tensor,
        v_new: &Tensor,
        row_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let kc_shape = k_cache.shape().dims();
        let vc_shape = v_cache.shape().dims();
        let kn_shape = k_new.shape().dims();
        let vn_shape = v_new.shape().dims();
        if kc_shape.len() != 2
            || vc_shape.len() != 2
            || kn_shape.len() != 2
            || vn_shape.len() != 2
            || kn_shape[1] != kc_shape[1]
            || vn_shape[1] != vc_shape[1]
            || kn_shape[0] != vn_shape[0]
        {
            return Err(VulkanError::ShapeMismatch {
                op: "kv_cache_append_pair".into(),
                lhs: kc_shape.to_vec(),
                rhs: kn_shape.to_vec(),
            });
        }

        let num_new = kn_shape[0];
        let cols_k = kc_shape[1];
        let cols_v = vc_shape[1];

        let kc_buf = KoreVulkanBuffer::from_storage(
            k_cache.storage_ref(),
            kc_shape,
            &self.device,
            &self.queue,
        )?;
        let vc_buf = KoreVulkanBuffer::from_storage(
            v_cache.storage_ref(),
            vc_shape,
            &self.device,
            &self.queue,
        )?;
        let kn_buf = KoreVulkanBuffer::from_storage(
            k_new.storage_ref(),
            kn_shape,
            &self.device,
            &self.queue,
        )?;
        let vn_buf = KoreVulkanBuffer::from_storage(
            v_new.storage_ref(),
            vn_shape,
            &self.device,
            &self.queue,
        )?;

        let numel_k = (num_new * cols_k) as u32;
        let numel_v = (num_new * cols_v) as u32;
        if numel_k + numel_v > 0 {
            let uniforms = KvAppendPairUniforms {
                numel_k,
                numel_v,
                k_dst_off: (row_offset * cols_k) as u32,
                v_dst_off: (row_offset * cols_v) as u32,
            };
            let workgroups = (numel_k + numel_v).div_ceil(256).max(1);
            self.record_dispatch(
                &self.kv_append_pair_pipeline,
                bytemuck::bytes_of(&uniforms),
                &[
                    kn_buf.arc_buffer(),
                    vn_buf.arc_buffer(),
                    kc_buf.arc_buffer(),
                    vc_buf.arc_buffer(),
                ],
                (workgroups, 1, 1),
                "kv_append_pair_pass",
            );
        }

        let total_rows = row_offset + num_new;
        let k_view = KoreVulkanBuffer::new(
            Arc::clone(kc_buf.arc_buffer()),
            DType::F32,
            vec![total_rows, cols_k],
        );
        let v_view = KoreVulkanBuffer::new(
            Arc::clone(vc_buf.arc_buffer()),
            DType::F32,
            vec![total_rows, cols_v],
        );
        Ok((
            self.make_tensor(k_view, &[total_rows, cols_k]),
            self.make_tensor(v_view, &[total_rows, cols_v]),
        ))
    }

    /// Fused residual add + RMSNorm: computes `res = a + b` and
    /// `normed = rms_norm(res) * weight` in one dispatch. Returns `(res, normed)`.
    pub fn add_rms_norm(
        &self,
        a: &Tensor,
        b: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor)> {
        let shape = a.shape().dims();
        if a.numel() != b.numel() {
            return Err(VulkanError::ShapeMismatch {
                op: "add_rms_norm".into(),
                lhs: shape.to_vec(),
                rhs: b.shape().dims().to_vec(),
            });
        }
        let (rows, cols) = if shape.len() == 2 {
            (shape[0], shape[1])
        } else {
            (1, a.numel())
        };
        if weight.numel() != cols {
            return Err(VulkanError::ShapeMismatch {
                op: "add_rms_norm".into(),
                lhs: vec![cols],
                rhs: vec![weight.numel()],
            });
        }

        let a_buf =
            KoreVulkanBuffer::from_storage(a.storage_ref(), shape, &self.device, &self.queue)?;
        let b_buf =
            KoreVulkanBuffer::from_storage(b.storage_ref(), shape, &self.device, &self.queue)?;
        let w_buf = KoreVulkanBuffer::from_storage(
            weight.storage_ref(),
            weight.shape().dims(),
            &self.device,
            &self.queue,
        )?;

        let out_bytes = (a.numel() * 4).max(4);
        let res_buf =
            crate::buffer::alloc_storage_buffer(&self.device, out_bytes, "kore_addres_out");
        let norm_buf =
            crate::buffer::alloc_storage_buffer(&self.device, out_bytes, "kore_addnorm_out");

        let uniforms = AddRmsNormUniforms {
            rows: rows as u32,
            cols: cols as u32,
            eps,
            _pad: 0,
        };
        self.record_dispatch(
            &self.add_rms_norm_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[
                a_buf.arc_buffer(),
                b_buf.arc_buffer(),
                w_buf.arc_buffer(),
                &res_buf,
                &norm_buf,
            ],
            (rows as u32, 1, 1),
            "add_rms_norm_pass",
        );

        Ok((
            self.make_tensor(
                KoreVulkanBuffer::new(res_buf, DType::F32, shape.to_vec()),
                shape,
            ),
            self.make_tensor(
                KoreVulkanBuffer::new(norm_buf, DType::F32, shape.to_vec()),
                shape,
            ),
        ))
    }

    /// Q2_0 quantized GEMV: y[m,n] = x[m,k] @ W^T where W is [n,k] Q2_0-packed
    /// (64 elems/block, 18 bytes: fp16 scale + 16 bytes of 2-bit {-1,0,1,2} codes).
    pub fn gemv_q2_0(&self, x: &Tensor, w_raw: &Tensor, n: usize, k: usize) -> Result<Tensor> {
        if k % 512 != 0 {
            return Err(VulkanError::ShapeMismatch {
                op: "gemv_q2_0".into(),
                lhs: vec![k],
                rhs: vec![512],
            });
        }
        let x_shape = x.shape().dims();
        let m = if x_shape.len() == 2 { x_shape[0] } else { 1 };
        if x.numel() != m * k {
            return Err(VulkanError::ShapeMismatch {
                op: "gemv_q2_0".into(),
                lhs: x_shape.to_vec(),
                rhs: vec![m, k],
            });
        }
        let blocks_per_row = k / 64;
        if w_raw.numel() != n * blocks_per_row * 18 {
            return Err(VulkanError::ShapeMismatch {
                op: "gemv_q2_0".into(),
                lhs: w_raw.shape().dims().to_vec(),
                rhs: vec![n * blocks_per_row * 18],
            });
        }
        if m > 4 {
            return Err(VulkanError::UnsupportedOperation(
                "gemv_q2_0 supports up to 4 rows".into(),
            ));
        }

        let x_buf =
            KoreVulkanBuffer::from_storage(x.storage_ref(), x_shape, &self.device, &self.queue)?;
        let w_buf = KoreVulkanBuffer::from_storage(
            w_raw.storage_ref(),
            w_raw.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        let out =
            crate::buffer::alloc_storage_buffer(&self.device, (m * n * 4).max(4), "kore_q2_out");

        let uniforms = Q2GemvUniforms {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            blocks_per_row: blocks_per_row as u32,
        };
        self.record_dispatch(
            &self.gemv_q2_0_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[x_buf.arc_buffer(), w_buf.arc_buffer(), &out],
            // 4 sub-warps x 4 rows per warp = 16 output rows per workgroup.
            (n.div_ceil(16).max(1) as u32, m.div_ceil(4).max(1) as u32, 1),
            "gemv_q2_0_pass",
        );
        Ok(self.make_tensor(KoreVulkanBuffer::new(out, DType::F32, vec![m, n]), &[m, n]))
    }

    /// Signed block-wise Hadamard transform.
    /// mode 0 (forward): out = WHT(signs ⊙ x) / sqrt(block)
    /// mode 1 (restore): out = signs ⊙ (WHT(x) / sqrt(block))
    pub fn hadamard_signed(
        &self,
        x: &Tensor,
        signs: &Tensor,
        block_size: usize,
        mode: u32,
    ) -> Result<Tensor> {
        let shape = x.shape().dims();
        let numel = x.numel();
        // The transform acts on 512-blocks *within each row*, and `signs` covers one
        // row, so an `[rows, cols]` input applies the same sign pattern to every row.
        // At `rows = 1` (`cols == numel`) this is the check it always was.
        let cols = shape.last().copied().unwrap_or(numel);
        if block_size != 512
            || numel % block_size != 0
            || cols % block_size != 0
            || signs.numel() < cols
        {
            return Err(VulkanError::ShapeMismatch {
                op: "hadamard_signed".into(),
                lhs: vec![numel],
                rhs: vec![signs.numel(), block_size],
            });
        }
        let x_buf = KoreVulkanBuffer::from_storage(
            x.storage_ref(),
            x.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        let s_buf = KoreVulkanBuffer::from_storage(
            signs.storage_ref(),
            signs.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        let out =
            crate::buffer::alloc_storage_buffer(&self.device, (numel * 4).max(4), "kore_hads_out");

        let uniforms = HadamardSignedUniforms {
            total_blocks: (numel / block_size) as u32,
            block_size: block_size as u32,
            scale: (block_size as f32).sqrt().recip(),
            mode,
            cols: cols as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        self.record_dispatch(
            &self.hadamard_signed_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[x_buf.arc_buffer(), s_buf.arc_buffer(), &out],
            ((numel / block_size) as u32, 1, 1),
            "hadamard_signed_pass",
        );
        // Preserve the input's rank: the transform is per row, so an `[rows, cols]`
        // input yields `[rows, cols]` and the downstream matvecs see the same `m`
        // the caller passed. At `rows = 1` this is the old `[numel]`.
        Ok(self.make_tensor(
            KoreVulkanBuffer::new(out, DType::F32, shape.to_vec()),
            shape,
        ))
    }

    /// Fused "pre-op + 512-point signed Hadamard" decode kernel.
    ///
    /// Every decode layer does `norm -> hadamard -> mat-vec` back to back, and in
    /// the strictly serial decode chain each extra dispatch costs a full exposed
    /// dependency latency. This merges the pair into a single kernel.
    ///
    /// `mode`:
    /// * `0` — `add` + `rms_norm` + `hadamard` (residual add folded into the norm)
    /// * `1` — `rms_norm` + `hadamard`
    /// * `2` — `swiglu` + `hadamard`
    /// * `3` — `sigmoid_mul` + `hadamard`
    ///
    /// Returns `(out_res, out_norm, out_had)`:
    /// * mode 0: all three (`a + b`, `rms_norm(a+b) * weight`, `WHT(signs * norm)`)
    /// * mode 1: `out_norm`, `out_had` (`out_res` is uninitialized)
    /// * mode 2/3: `out_had` only
    ///
    /// `weight` is only read for modes 0/1 (pass `signs` for modes 2/3).
    #[allow(clippy::too_many_arguments)]
    pub fn had_pre(
        &self,
        mode: u32,
        had: bool,
        a: &Tensor,
        b: &Tensor,
        weight: &Tensor,
        signs: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let shape = a.shape().dims();
        let numel = a.numel();
        if mode > 3 {
            return Err(VulkanError::UnsupportedOperation(format!(
                "had_pre mode {mode} (expected 0..=3)"
            )));
        }
        if b.numel() != numel || numel == 0 {
            return Err(VulkanError::ShapeMismatch {
                op: "had_pre".into(),
                lhs: shape.to_vec(),
                rhs: b.shape().dims().to_vec(),
            });
        }
        let (rows, cols) = if shape.len() == 2 {
            (shape[0], shape[1])
        } else {
            (1, numel)
        };
        if cols % 512 != 0 {
            return Err(VulkanError::ShapeMismatch {
                op: "had_pre".into(),
                lhs: vec![cols],
                rhs: vec![cols.next_multiple_of(512)],
            });
        }
        if mode <= 1 && weight.numel() != cols {
            return Err(VulkanError::ShapeMismatch {
                op: "had_pre".into(),
                lhs: vec![cols],
                rhs: vec![weight.numel()],
            });
        }
        // `signs` covers one row; the pattern is applied to every row of an
        // `[rows, cols]` input. At `rows = 1` this is `signs.numel() < numel`.
        // Only read when the Hadamard stage runs, so an unfolded model may pass
        // any tensor here.
        if had && signs.numel() < cols {
            return Err(VulkanError::ShapeMismatch {
                op: "had_pre".into(),
                lhs: vec![cols],
                rhs: vec![signs.numel()],
            });
        }

        let a_buf =
            KoreVulkanBuffer::from_storage(a.storage_ref(), shape, &self.device, &self.queue)?;
        let b_buf =
            KoreVulkanBuffer::from_storage(b.storage_ref(), shape, &self.device, &self.queue)?;
        let w_buf = KoreVulkanBuffer::from_storage(
            weight.storage_ref(),
            weight.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        let s_buf = KoreVulkanBuffer::from_storage(
            signs.storage_ref(),
            signs.shape().dims(),
            &self.device,
            &self.queue,
        )?;

        let out_bytes = (numel * 4).max(4);
        let res_buf =
            crate::buffer::alloc_storage_buffer(&self.device, out_bytes, "kore_hadpre_res");
        let norm_buf =
            crate::buffer::alloc_storage_buffer(&self.device, out_bytes, "kore_hadpre_norm");
        let had_buf =
            crate::buffer::alloc_storage_buffer(&self.device, out_bytes, "kore_hadpre_had");

        let blocks = cols / 512;
        let uniforms = HadPreUniforms {
            rows: rows as u32,
            cols: cols as u32,
            blocks: blocks as u32,
            mode,
            eps,
            scale: 512f32.sqrt().recip(),
            had: had as u32,
            _pad1: 0,
        };
        self.record_dispatch(
            &self.had_pre_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[
                a_buf.arc_buffer(),
                b_buf.arc_buffer(),
                w_buf.arc_buffer(),
                s_buf.arc_buffer(),
                &res_buf,
                &norm_buf,
                &had_buf,
            ],
            ((rows * blocks) as u32, 1, 1),
            "had_pre_pass",
        );

        Ok((
            self.make_tensor(
                KoreVulkanBuffer::new(res_buf, DType::F32, shape.to_vec()),
                shape,
            ),
            self.make_tensor(
                KoreVulkanBuffer::new(norm_buf, DType::F32, shape.to_vec()),
                shape,
            ),
            self.make_tensor(
                KoreVulkanBuffer::new(had_buf, DType::F32, shape.to_vec()),
                shape,
            ),
        ))
    }

    /// Gated-DeltaNet depthwise causal conv + SiLU with in-place rolling history.
    /// x [conv_dim], w [conv_dim, conv_k], hist [conv_dim, conv_k-1] mutated.
    pub fn gdn_conv(
        &self,
        x: &Tensor,
        w: &Tensor,
        hist: &Tensor,
        conv_dim: usize,
        conv_k: usize,
    ) -> Result<Tensor> {
        let x_buf = KoreVulkanBuffer::from_storage(
            x.storage_ref(),
            x.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        let w_buf = KoreVulkanBuffer::from_storage(
            w.storage_ref(),
            w.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        let h_buf = KoreVulkanBuffer::from_storage(
            hist.storage_ref(),
            hist.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        let out = crate::buffer::alloc_storage_buffer(
            &self.device,
            (conv_dim * 4).max(4),
            "kore_gdn_conv_out",
        );

        let uniforms = GdnConvUniforms {
            conv_dim: conv_dim as u32,
            conv_k: conv_k as u32,
            _pad0: 0,
            _pad1: 0,
        };
        self.record_dispatch(
            &self.gdn_conv_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[
                x_buf.arc_buffer(),
                w_buf.arc_buffer(),
                h_buf.arc_buffer(),
                &out,
            ],
            ((conv_dim as u32).div_ceil(256).max(1), 1, 1),
            "gdn_conv_pass",
        );
        Ok(self.make_tensor(
            KoreVulkanBuffer::new(out, DType::F32, vec![conv_dim]),
            &[conv_dim],
        ))
    }

    /// Fused alpha/beta gates: beta = sigmoid(W_b @ h),
    /// gate = softplus(W_a @ h + dt_bias) * a. Returns (beta[n_v], gate[n_v]).
    pub fn gdn_gates(
        &self,
        h: &Tensor,
        w_alpha: &Tensor,
        w_beta: &Tensor,
        dt_bias: &Tensor,
        ssm_a: &Tensor,
        hidden: usize,
        n_v: usize,
    ) -> Result<(Tensor, Tensor)> {
        let h_buf = KoreVulkanBuffer::from_storage(
            h.storage_ref(),
            h.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        let wa_buf = KoreVulkanBuffer::from_storage(
            w_alpha.storage_ref(),
            w_alpha.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        let wb_buf = KoreVulkanBuffer::from_storage(
            w_beta.storage_ref(),
            w_beta.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        let dt_buf = KoreVulkanBuffer::from_storage(
            dt_bias.storage_ref(),
            dt_bias.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        let a_buf = KoreVulkanBuffer::from_storage(
            ssm_a.storage_ref(),
            ssm_a.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        let beta_out =
            crate::buffer::alloc_storage_buffer(&self.device, (n_v * 4).max(4), "kore_gdn_beta");
        let gate_out =
            crate::buffer::alloc_storage_buffer(&self.device, (n_v * 4).max(4), "kore_gdn_gate");

        let uniforms = GdnGatesUniforms {
            hidden: hidden as u32,
            n_v: n_v as u32,
            _pad0: 0,
            _pad1: 0,
        };
        self.record_dispatch(
            &self.gdn_gates_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[
                h_buf.arc_buffer(),
                wa_buf.arc_buffer(),
                wb_buf.arc_buffer(),
                dt_buf.arc_buffer(),
                a_buf.arc_buffer(),
                &beta_out,
                &gate_out,
            ],
            (n_v as u32, 1, 1),
            "gdn_gates_pass",
        );
        Ok((
            self.make_tensor(
                KoreVulkanBuffer::new(beta_out, DType::F32, vec![n_v]),
                &[n_v],
            ),
            self.make_tensor(
                KoreVulkanBuffer::new(gate_out, DType::F32, vec![n_v]),
                &[n_v],
            ),
        ))
    }

    /// Fused DeltaNet recurrent step: L2-normed q/k, delta-rule state update
    /// (state mutated in place), output projection, gated RMSNorm with z.
    /// qkv = [q | k | v] conv output, state [n_v*d_v*d_v], out [n_v*d_v].
    pub fn gdn_step(
        &self,
        qkv: &Tensor,
        z: &Tensor,
        state: &Tensor,
        beta: &Tensor,
        gate: &Tensor,
        norm_w: &Tensor,
        n_v: usize,
        n_k: usize,
        d_k: usize,
        d_v: usize,
        eps: f32,
    ) -> Result<Tensor> {
        let qkv_buf = KoreVulkanBuffer::from_storage(
            qkv.storage_ref(),
            qkv.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        let z_buf = KoreVulkanBuffer::from_storage(
            z.storage_ref(),
            z.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        let s_buf = KoreVulkanBuffer::from_storage(
            state.storage_ref(),
            state.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        let b_buf = KoreVulkanBuffer::from_storage(
            beta.storage_ref(),
            beta.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        let g_buf = KoreVulkanBuffer::from_storage(
            gate.storage_ref(),
            gate.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        let nw_buf = KoreVulkanBuffer::from_storage(
            norm_w.storage_ref(),
            norm_w.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        let out = crate::buffer::alloc_storage_buffer(
            &self.device,
            (n_v * d_v * 4).max(4),
            "kore_gdn_step_out",
        );

        let uniforms = GdnStepUniforms {
            n_v: n_v as u32,
            n_k: n_k as u32,
            d_k: d_k as u32,
            d_v: d_v as u32,
            eps,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        self.record_dispatch(
            &self.gdn_step_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[
                qkv_buf.arc_buffer(),
                z_buf.arc_buffer(),
                s_buf.arc_buffer(),
                b_buf.arc_buffer(),
                g_buf.arc_buffer(),
                nw_buf.arc_buffer(),
                &out,
            ],
            (n_v as u32, 1, 1),
            "gdn_step_pass",
        );
        Ok(self.make_tensor(
            KoreVulkanBuffer::new(out, DType::F32, vec![n_v * d_v]),
            &[n_v * d_v],
        ))
    }

    /// Per-head Q/K RMSNorm + gate split + partial NEOX RoPE for Qwen3.5
    /// full-attention layers. qg [n_heads*(2*head_dim)] = [q|gate] per head.
    /// Returns (q_roped, k_roped, gate).
    pub fn qknorm_rope(
        &self,
        qg: &Tensor,
        kk: &Tensor,
        q_norm: &Tensor,
        k_norm: &Tensor,
        n_heads: usize,
        n_kv: usize,
        head_dim: usize,
        rope_dim: usize,
        pos: usize,
        theta: f32,
        eps: f32,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        if head_dim != 256 {
            return Err(VulkanError::UnsupportedOperation(
                "qknorm_rope requires head_dim 256".into(),
            ));
        }
        let qg_buf = KoreVulkanBuffer::from_storage(
            qg.storage_ref(),
            qg.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        let kk_buf = KoreVulkanBuffer::from_storage(
            kk.storage_ref(),
            kk.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        let qn_buf = KoreVulkanBuffer::from_storage(
            q_norm.storage_ref(),
            q_norm.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        let kn_buf = KoreVulkanBuffer::from_storage(
            k_norm.storage_ref(),
            k_norm.shape().dims(),
            &self.device,
            &self.queue,
        )?;
        // `rows` comes from the input shape, so a [1, ...] caller is unchanged
        // and a [m, ...] caller gets per-row positions `pos + row` for free.
        let rows = qg
            .shape()
            .dims()
            .first()
            .copied()
            .filter(|&r| r > 0)
            .unwrap_or(1);
        let q_out = crate::buffer::alloc_storage_buffer(
            &self.device,
            (rows * n_heads * head_dim * 4).max(4),
            "kore_qknr_q",
        );
        let k_out = crate::buffer::alloc_storage_buffer(
            &self.device,
            (rows * n_kv * head_dim * 4).max(4),
            "kore_qknr_k",
        );
        let g_out = crate::buffer::alloc_storage_buffer(
            &self.device,
            (rows * n_heads * head_dim * 4).max(4),
            "kore_qknr_g",
        );

        let uniforms = QkNormRopeUniforms {
            n_heads: n_heads as u32,
            n_kv: n_kv as u32,
            head_dim: head_dim as u32,
            rope_dim: rope_dim as u32,
            pos: pos as u32,
            theta,
            eps,
            rows: rows as u32,
        };
        self.record_dispatch(
            &self.qknorm_rope_pipeline,
            bytemuck::bytes_of(&uniforms),
            &[
                qg_buf.arc_buffer(),
                kk_buf.arc_buffer(),
                qn_buf.arc_buffer(),
                kn_buf.arc_buffer(),
                &q_out,
                &k_out,
                &g_out,
            ],
            ((n_heads + n_kv) as u32, rows as u32, 1),
            "qknorm_rope_pass",
        );
        Ok((
            self.make_tensor(
                KoreVulkanBuffer::new(q_out, DType::F32, vec![rows, n_heads * head_dim]),
                &[rows, n_heads * head_dim],
            ),
            self.make_tensor(
                KoreVulkanBuffer::new(k_out, DType::F32, vec![rows, n_kv * head_dim]),
                &[rows, n_kv * head_dim],
            ),
            self.make_tensor(
                KoreVulkanBuffer::new(g_out, DType::F32, vec![rows, n_heads * head_dim]),
                &[rows, n_heads * head_dim],
            ),
        ))
    }

    /// Register this Vulkan backend instance with `kore_core`'s global backend registry.
    pub fn register(self: Arc<Self>) {
        kore_core::device::register_backend(self);
    }
}

impl kore_core::device::DeviceBackend for VulkanBackend {
    fn device_name(&self) -> &'static str {
        "vulkan"
    }

    fn supports(&self, device: Device) -> bool {
        matches!(device, Device::Vulkan(_))
    }

    fn copy_to_device(
        &self,
        storage: &kore_core::Storage,
        _target: Device,
    ) -> kore_core::Result<kore_core::Storage> {
        let buf =
            KoreVulkanBuffer::from_storage(storage, &[storage.numel()], &self.device, &self.queue)
                .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))?;
        Ok(kore_core::Storage::from_device(
            storage.dtype(),
            storage.numel(),
            Arc::new(buf),
        ))
    }

    /// Upload a host slice straight into a device buffer.
    ///
    /// `copy_to_device` would work, but it forces the caller to hand over an
    /// owned `Vec` first -- a full-size copy and allocation on a path that an
    /// offloading engine walks on every expert miss.
    fn upload_u8(&self, bytes: &[u8], _target: Device) -> kore_core::Result<kore_core::Storage> {
        let buf = KoreVulkanBuffer::from_slice(
            bytes,
            DType::U8,
            &[bytes.len()],
            &self.device,
            &self.queue,
        )
        .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))?;
        Ok(kore_core::Storage::from_device(
            DType::U8,
            bytes.len(),
            Arc::new(buf),
        ))
    }

    fn copy_to_cpu(&self, storage: &kore_core::Storage) -> kore_core::Result<kore_core::Storage> {
        let dev_storage = storage.device_storage().ok_or_else(|| {
            kore_core::KoreError::StorageError("Storage is not a device storage".into())
        })?;
        let kore_buf = dev_storage
            .as_any()
            .downcast_ref::<KoreVulkanBuffer>()
            .ok_or_else(|| {
                kore_core::KoreError::StorageError("DeviceStorage is not a KoreVulkanBuffer".into())
            })?;
        let mut batch = self.batch.lock();
        batch.uniform_cursor = 0;
        let pending_encoder = batch.take_encoder(&self.device);
        let res = kore_buf
            .to_storage_with_encoder(&self.device, &self.queue, pending_encoder)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()));
        drop(batch);
        res
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> kore_core::Result<Tensor> {
        self.matmul(a, b)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> kore_core::Result<Tensor> {
        self.add(a, b)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn sub(&self, a: &Tensor, b: &Tensor) -> kore_core::Result<Tensor> {
        self.elementwise_binary(a, b, 2)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> kore_core::Result<Tensor> {
        self.mul(a, b)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn div(&self, a: &Tensor, b: &Tensor) -> kore_core::Result<Tensor> {
        self.elementwise_binary(a, b, 3)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn add_scalar(&self, a: &Tensor, scalar: f32) -> kore_core::Result<Tensor> {
        self.add_scalar(a, scalar)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn mul_scalar(&self, a: &Tensor, scalar: f32) -> kore_core::Result<Tensor> {
        self.mul_scalar(a, scalar)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> kore_core::Result<Tensor> {
        self.rms_norm(input, weight, eps)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn softmax(&self, input: &Tensor) -> kore_core::Result<Tensor> {
        self.softmax(input)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn contiguous(&self, input: &Tensor) -> kore_core::Result<Tensor> {
        if input.shape().dims().len() == 2 {
            self.transpose_2d(input)
                .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
        } else {
            Err(kore_core::KoreError::StorageError(
                "contiguous only implemented for 2D on Vulkan".into(),
            ))
        }
    }

    fn slice_rows(
        &self,
        input: &Tensor,
        start_row: usize,
        num_rows: usize,
    ) -> kore_core::Result<Tensor> {
        self.slice_rows(input, start_row, num_rows)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn scatter_row(&self, dst: &Tensor, row: usize, src: &Tensor) -> kore_core::Result<()> {
        self.scatter_row(dst, row, src)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn copy_buffer(&self, dst: &Tensor, src: &Tensor) -> kore_core::Result<()> {
        self.copy_buffer(dst, src)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn kv_cache_append(
        &self,
        cache_buf: &Tensor,
        new_rows: &Tensor,
        row_offset: usize,
    ) -> kore_core::Result<Tensor> {
        self.kv_cache_append(cache_buf, new_rows, row_offset)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn hadamard_transform(&self, input: &Tensor) -> kore_core::Result<Tensor> {
        self.hadamard_transform(input)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn quantized_matmul(
        &self,
        a: &Tensor,
        b: &Tensor,
        scales: &Tensor,
        _format: u32,
    ) -> kore_core::Result<Tensor> {
        self.quantized_matmul(a, b, scales)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn fused_attention_decode(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
    ) -> kore_core::Result<Tensor> {
        self.fused_attention_decode(q, k, v, scale)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn fused_attention_decode_gqa(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> kore_core::Result<Tensor> {
        self.fused_attention_decode_gqa(q, k, v, num_heads, num_kv_heads, head_dim, scale)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn rope(
        &self,
        x: &Tensor,
        pos: usize,
        num_heads: usize,
        head_dim: usize,
        rope_theta: f32,
    ) -> kore_core::Result<Tensor> {
        self.rope(x, pos, num_heads, head_dim, rope_theta)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn argmax_last_dim(&self, logits: &Tensor) -> kore_core::Result<Vec<u32>> {
        self.argmax_last_dim(logits)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn silu(&self, a: &Tensor) -> kore_core::Result<Tensor> {
        self.silu(a)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn swiglu(&self, gate: &Tensor, up: &Tensor) -> kore_core::Result<Tensor> {
        self.swiglu(gate, up)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn swiglu_split(
        &self,
        gate_up: &Tensor,
        intermediate_size: usize,
    ) -> kore_core::Result<Tensor> {
        self.swiglu_split(gate_up, intermediate_size)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn gemv_q4_k(
        &self,
        x: &Tensor,
        w_raw: &Tensor,
        n: usize,
        k: usize,
    ) -> kore_core::Result<Tensor> {
        self.gemv_q4_k(x, w_raw, n, k)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn gemv_q6_k(
        &self,
        x: &Tensor,
        w_raw: &Tensor,
        n: usize,
        k: usize,
    ) -> kore_core::Result<Tensor> {
        self.gemv_q6_k(x, w_raw, n, k)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn gemv_q5_k(
        &self,
        x: &Tensor,
        w_raw: &Tensor,
        n: usize,
        k: usize,
    ) -> kore_core::Result<Tensor> {
        self.gemv_q5_k(x, w_raw, n, k)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn gemv_q4_k_split3(
        &self,
        x: &Tensor,
        w_raw: &Tensor,
        nq: usize,
        nk: usize,
        nv: usize,
        k: usize,
    ) -> kore_core::Result<(Tensor, Tensor, Tensor)> {
        self.gemv_q4_k_split3(x, w_raw, nq, nk, nv, k)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn gemv_q6_k_split3(
        &self,
        x: &Tensor,
        w_raw: &Tensor,
        nq: usize,
        nk: usize,
        nv: usize,
        k: usize,
    ) -> kore_core::Result<(Tensor, Tensor, Tensor)> {
        self.gemv_q6_k_split3(x, w_raw, nq, nk, nv, k)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn rope_pair(
        &self,
        q: &Tensor,
        k: &Tensor,
        pos: usize,
        num_heads_q: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_theta: f32,
    ) -> kore_core::Result<(Tensor, Tensor)> {
        self.rope_pair(q, k, pos, num_heads_q, num_kv_heads, head_dim, rope_theta)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn kv_cache_append_pair(
        &self,
        k_cache: &Tensor,
        v_cache: &Tensor,
        k_new: &Tensor,
        v_new: &Tensor,
        row_offset: usize,
    ) -> kore_core::Result<(Tensor, Tensor)> {
        self.kv_cache_append_pair(k_cache, v_cache, k_new, v_new, row_offset)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn add_rms_norm(
        &self,
        a: &Tensor,
        b: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> kore_core::Result<(Tensor, Tensor)> {
        self.add_rms_norm(a, b, weight, eps)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn sigmoid_mul(&self, a: &Tensor, b: &Tensor) -> kore_core::Result<Tensor> {
        self.elementwise_binary(a, b, 10)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn gemv_q2_0(
        &self,
        x: &Tensor,
        w_raw: &Tensor,
        n: usize,
        k: usize,
    ) -> kore_core::Result<Tensor> {
        self.gemv_q2_0(x, w_raw, n, k)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn hadamard_signed(
        &self,
        x: &Tensor,
        signs: &Tensor,
        block_size: usize,
        mode: u32,
    ) -> kore_core::Result<Tensor> {
        self.hadamard_signed(x, signs, block_size, mode)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn had_pre(
        &self,
        mode: u32,
        had: bool,
        a: &Tensor,
        b: &Tensor,
        weight: &Tensor,
        signs: &Tensor,
        eps: f32,
    ) -> kore_core::Result<(Tensor, Tensor, Tensor)> {
        self.had_pre(mode, had, a, b, weight, signs, eps)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn gdn_conv(
        &self,
        x: &Tensor,
        w: &Tensor,
        hist: &Tensor,
        conv_dim: usize,
        conv_k: usize,
    ) -> kore_core::Result<Tensor> {
        self.gdn_conv(x, w, hist, conv_dim, conv_k)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn gdn_gates(
        &self,
        h: &Tensor,
        w_alpha: &Tensor,
        w_beta: &Tensor,
        dt_bias: &Tensor,
        ssm_a: &Tensor,
        hidden: usize,
        n_v: usize,
    ) -> kore_core::Result<(Tensor, Tensor)> {
        self.gdn_gates(h, w_alpha, w_beta, dt_bias, ssm_a, hidden, n_v)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn gdn_step(
        &self,
        qkv: &Tensor,
        z: &Tensor,
        state: &Tensor,
        beta: &Tensor,
        gate: &Tensor,
        norm_w: &Tensor,
        n_v: usize,
        n_k: usize,
        d_k: usize,
        d_v: usize,
        eps: f32,
    ) -> kore_core::Result<Tensor> {
        self.gdn_step(qkv, z, state, beta, gate, norm_w, n_v, n_k, d_k, d_v, eps)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn qknorm_rope(
        &self,
        qg: &Tensor,
        kk: &Tensor,
        q_norm: &Tensor,
        k_norm: &Tensor,
        n_heads: usize,
        n_kv: usize,
        head_dim: usize,
        rope_dim: usize,
        pos: usize,
        theta: f32,
        eps: f32,
    ) -> kore_core::Result<(Tensor, Tensor, Tensor)> {
        self.qknorm_rope(
            qg, kk, q_norm, k_norm, n_heads, n_kv, head_dim, rope_dim, pos, theta, eps,
        )
        .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn kv_cache_append_q8(
        &self,
        k_cache_buf: &Tensor,
        v_cache_buf: &Tensor,
        k_scales_buf: &Tensor,
        v_scales_buf: &Tensor,
        k_new: &Tensor,
        v_new: &Tensor,
        k_scales_new: &Tensor,
        v_scales_new: &Tensor,
        row_offset: usize,
    ) -> kore_core::Result<(Tensor, Tensor)> {
        self.kv_cache_append_q8(
            k_cache_buf,
            v_cache_buf,
            k_scales_buf,
            v_scales_buf,
            k_new,
            v_new,
            k_scales_new,
            v_scales_new,
            row_offset,
        )
        .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn fused_attention_decode_gqa_q8(
        &self,
        q: &Tensor,
        k_q8: &Tensor,
        v_q8: &Tensor,
        k_scales: &Tensor,
        v_scales: &Tensor,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        seq_k: usize,
    ) -> kore_core::Result<Tensor> {
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
        .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    fn quantize_q8_rows(&self, x: &Tensor) -> kore_core::Result<(Tensor, Tensor)> {
        self.quantize_q8_rows(x)
            .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    fn fused_attention_decode_gqa_q8_split(
        &self,
        q: &Tensor,
        k_q8: &Tensor,
        v_q8: &Tensor,
        k_scales: &Tensor,
        v_scales: &Tensor,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        n_splits: usize,
        seq_k: usize,
    ) -> kore_core::Result<Tensor> {
        self.fused_attention_decode_gqa_q8_split(
            q,
            k_q8,
            v_q8,
            k_scales,
            v_scales,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            n_splits,
            seq_k,
        )
        .map_err(|e| kore_core::KoreError::StorageError(e.to_string()))
    }
}

/// Register a new default Vulkan backend with `kore_core`'s global backend registry.
pub fn register_vulkan_backend() -> Result<Arc<VulkanBackend>> {
    let backend = Arc::new(VulkanBackend::new()?);
    kore_core::device::register_backend(backend.clone());
    Ok(backend)
}
