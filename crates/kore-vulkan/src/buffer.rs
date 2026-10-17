//! Tensor Storage ↔ Vulkan/GPU Buffer bridge using wgpu.

use crate::error::{Result, VulkanError};
use kore_core::{DType, Device, Storage};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

struct BufferPool {
    /// Buffers ready for immediate reuse, keyed by buffer byte size.
    free: HashMap<u64, Vec<Arc<wgpu::Buffer>>>,
    /// Buffers dropped since the last `Queue::submit`. They may still be read by
    /// a command buffer that has not finished, so they are parked here and only
    /// returned to `free` once that submission completes.
    pending: Vec<Arc<wgpu::Buffer>>,
}

static STORAGE_BUFFER_POOL: OnceLock<parking_lot::Mutex<BufferPool>> = OnceLock::new();

static ALLOC_HITS: AtomicU64 = AtomicU64::new(0);
static ALLOC_MISSES: AtomicU64 = AtomicU64::new(0);

/// `(reused, freshly created)` storage buffers since process start.
///
/// A decode step is a long chain of small ops, each of which allocates its
/// output; if this shows misses growing ~1:1 with dispatches, the pool is not
/// being refilled fast enough and the token is paying GPU buffer creation on
/// every op.
pub fn buffer_alloc_stats() -> (u64, u64) {
    (
        ALLOC_HITS.load(Ordering::Relaxed),
        ALLOC_MISSES.load(Ordering::Relaxed),
    )
}

fn get_buffer_pool() -> &'static parking_lot::Mutex<BufferPool> {
    STORAGE_BUFFER_POOL.get_or_init(|| {
        parking_lot::Mutex::new(BufferPool {
            free: HashMap::new(),
            pending: Vec::with_capacity(64),
        })
    })
}

/// Allocate or reuse a storage buffer of at least `size_bytes` (aligned to 4 bytes).
pub fn alloc_storage_buffer(
    device: &wgpu::Device,
    size_bytes: usize,
    label: &'static str,
) -> Arc<wgpu::Buffer> {
    let aligned_size = ((size_bytes + 3) & !3).max(4) as u64;
    {
        let mut pool = get_buffer_pool().lock();
        if let Some(list) = pool.free.get_mut(&aligned_size) {
            if let Some(buf) = list.pop() {
                ALLOC_HITS.fetch_add(1, Ordering::Relaxed);
                return buf;
            }
        }
    }
    ALLOC_MISSES.fetch_add(1, Ordering::Relaxed);

    Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: aligned_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }))
}

fn recycle(buffers: Vec<Arc<wgpu::Buffer>>) {
    if buffers.is_empty() {
        return;
    }
    let mut pool = get_buffer_pool().lock();
    for buf in buffers {
        let size = buf.size();
        let list = pool.free.entry(size).or_default();
        if list.len() < 4096 {
            list.push(buf);
        }
    }
}

/// Submit `encoder`, then return every buffer dropped so far to the pool once
/// that submission has completed.
///
/// This is the difference between a decode step that recycles its intermediates
/// and one that calls `create_buffer` for each of the ~500 tensors it makes.
/// `Drop` cannot hand a buffer straight back: the encoder being built still
/// references it. Snapshotting the dropped list at submit time ties the
/// recycling to the completion of the submission that last used them, which is
/// exactly when they become safe to overwrite.
pub fn submit_with_reclaim(queue: &wgpu::Queue, encoder: wgpu::CommandEncoder) {
    let pending = std::mem::take(&mut get_buffer_pool().lock().pending);
    queue.submit(std::iter::once(encoder.finish()));
    if !pending.is_empty() {
        // Fires on a later submit/poll, i.e. roughly one command buffer later.
        queue.on_submitted_work_done(move || recycle(pending));
    }
}

/// Move all dropped buffers into the free pool after GPU synchronization.
pub fn reclaim_in_flight_buffers() {
    let pending = std::mem::take(&mut get_buffer_pool().lock().pending);
    recycle(pending);
}

/// Wraps a GPU buffer with a KORE-compatible interface.
#[derive(Clone, Debug)]
pub struct KoreVulkanBuffer {
    buffer: Arc<wgpu::Buffer>,
    dtype: DType,
    shape: Vec<usize>,
}

impl Drop for KoreVulkanBuffer {
    fn drop(&mut self) {
        if Arc::strong_count(&self.buffer) == 1 {
            let mut pool = get_buffer_pool().lock();
            if pool.pending.len() < 8192 {
                pool.pending.push(Arc::clone(&self.buffer));
            }
        }
    }
}

impl kore_core::storage::DeviceStorage for KoreVulkanBuffer {
    fn nbytes(&self) -> usize {
        let numel: usize = self.shape.iter().product();
        self.dtype.storage_bytes(numel)
    }

    fn device(&self) -> Device {
        Device::Vulkan(0)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl KoreVulkanBuffer {
    /// Create a new KoreVulkanBuffer from existing wgpu::Buffer.
    pub fn new(buffer: Arc<wgpu::Buffer>, dtype: DType, shape: Vec<usize>) -> Self {
        Self {
            buffer,
            dtype,
            shape,
        }
    }

    /// Create from KORE storage (transfers data to GPU).
    pub fn from_storage(
        storage: &Storage,
        shape: &[usize],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Self> {
        let dtype = storage.dtype();
        if let Some(dev_storage) = storage.device_storage() {
            if let Some(vul_buf) = dev_storage.as_any().downcast_ref::<KoreVulkanBuffer>() {
                return Ok(Self {
                    buffer: Arc::clone(&vul_buf.buffer),
                    dtype,
                    shape: shape.to_vec(),
                });
            }
        }
        let nbytes = storage.nbytes();
        let buffer = alloc_storage_buffer(device, nbytes, "kore_vulkan_storage_buffer");

        if storage.device().is_cpu() && nbytes > 0 {
            let raw = storage.as_bytes();
            if raw.len().is_multiple_of(4) {
                queue.write_buffer(&buffer, 0, raw);
            } else {
                let aligned_len = (raw.len() + 3) & !3;
                let mut padded = vec![0u8; aligned_len];
                padded[..raw.len()].copy_from_slice(raw);
                queue.write_buffer(&buffer, 0, &padded);
            }
        }

        Ok(Self {
            buffer,
            dtype,
            shape: shape.to_vec(),
        })
    }

    /// Download data back to CPU storage, optionally flushing a pending CommandEncoder in the same submission.
    pub fn to_storage_with_encoder(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pending_encoder: Option<wgpu::CommandEncoder>,
    ) -> Result<Storage> {
        let numel: usize = self.shape.iter().product();
        let nbytes = self.dtype.storage_bytes(numel);
        let aligned_size = ((nbytes + 3) & !3).max(4) as u64;

        static STAGING_CACHE: OnceLock<parking_lot::Mutex<Option<wgpu::Buffer>>> = OnceLock::new();
        let staging_mu = STAGING_CACHE.get_or_init(|| parking_lot::Mutex::new(None));
        let mut staging_guard = staging_mu.lock();

        let staging_buffer = match staging_guard.take() {
            Some(b) if b.size() >= aligned_size => b,
            _ => device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("kore_vulkan_staging_read"),
                size: aligned_size.max(4 * 1024 * 1024),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        };

        let mut encoder = pending_encoder.unwrap_or_else(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("kore_vulkan_readback_encoder"),
            })
        });

        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, aligned_size);
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..aligned_size);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        reclaim_in_flight_buffers();

        let map_res = receiver
            .recv()
            .map_err(|e| VulkanError::Transfer(format!("Failed to receive map event: {e}")));

        match map_res {
            Ok(Ok(())) => {
                let mapped = buffer_slice.get_mapped_range().map_err(|e| {
                    VulkanError::Transfer(format!("Buffer range map failed: {e:?}"))
                })?;
                let mut data = vec![0u8; nbytes];
                data.copy_from_slice(&mapped[..nbytes]);
                drop(mapped);
                staging_buffer.unmap();
                *staging_guard = Some(staging_buffer);
                Storage::from_bytes(self.dtype, numel, data)
                    .map_err(|e| VulkanError::Transfer(e.to_string()))
            }
            Ok(Err(e)) => {
                staging_buffer.unmap();
                Err(VulkanError::Transfer(format!(
                    "Buffer mapping failed: {e:?}"
                )))
            }
            Err(e) => Err(e),
        }
    }

    /// Download data back to CPU storage.
    pub fn to_storage(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Storage> {
        self.to_storage_with_encoder(device, queue, None)
    }

    /// Access the underlying wgpu buffer.
    pub fn wgpu_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Access the underlying Arc<wgpu::Buffer>.
    pub fn arc_buffer(&self) -> &Arc<wgpu::Buffer> {
        &self.buffer
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Trait for KORE types that can provide Vulkan / GPU buffers.
pub trait ToVulkanBuffer {
    fn to_vulkan(
        &self,
        shape: &[usize],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<KoreVulkanBuffer>;
}

impl ToVulkanBuffer for Storage {
    fn to_vulkan(
        &self,
        shape: &[usize],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<KoreVulkanBuffer> {
        KoreVulkanBuffer::from_storage(self, shape, device, queue)
    }
}
