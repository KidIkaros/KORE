//! Host->device upload throughput for the shapes the expert stream actually uses.
//!
//! The port streams one expert (2.0 MiB packed) at a time into a pre-allocated
//! VRAM slot, six per layer. So the number that matters is not a single 64 MiB
//! copy: it is `write_buffer` into slots that already exist, with one submit per
//! layer. `KORE_UPLOAD_MODE` picks the shape, `KORE_UPLOAD_MB` the copy size.
use std::time::Instant;

use kore_core::device::{Device, DeviceBackend};
use kore_vulkan::buffer::alloc_storage_buffer;

fn main() {
    let vk = kore_vulkan::VulkanBackend::new().expect("vulkan");
    let device = vk.wgpu_device();
    let queue = vk.wgpu_queue();
    let _ = Device::Vulkan(0);

    let mb: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(600);
    let per_group: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    let mode = std::env::var("KORE_UPLOAD_MODE").unwrap_or_else(|_| "group".into());

    let size = mb << 20;
    let bytes = vec![0x5au8; size];

    // A pool of slots, the way a VRAM expert cache is laid out: allocated once,
    // overwritten on every miss.
    let slots: Vec<_> = (0..64)
        .map(|_| alloc_storage_buffer(device, size, "probe_slot"))
        .collect();

    // Warm: first writes build the belt and touch the pages.
    for s in slots.iter().take(8) {
        queue.write_buffer(s, 0, &bytes);
    }
    vk.flush_uploads();
    vk.sync();

    let t = Instant::now();
    let mut written = 0usize;
    match mode.as_str() {
        // One submit per group, then wait: the per-layer pattern.
        "group" => {
            let groups = iters.div_ceil(per_group);
            for g in 0..groups {
                for j in 0..per_group {
                    queue.write_buffer(&slots[(g * per_group + j) % slots.len()], 0, &bytes);
                    written += size;
                }
                vk.flush_uploads();
                vk.sync();
            }
        }
        // One submit per group, no wait until the end: maximum pipelining.
        "pipeline" => {
            let groups = iters.div_ceil(per_group);
            for g in 0..groups {
                for j in 0..per_group {
                    queue.write_buffer(&slots[(g * per_group + j) % slots.len()], 0, &bytes);
                    written += size;
                }
                queue.submit(std::iter::empty());
            }
            vk.sync();
        }
        // One big buffer, one wait at the end.
        "burst" => {
            let big = alloc_storage_buffer(device, size, "probe_big");
            for _ in 0..iters {
                queue.write_buffer(&big, 0, &bytes);
                written += size;
            }
            queue.submit(std::iter::empty());
            vk.sync();
        }
        other => panic!("unknown KORE_UPLOAD_MODE={other}"),
    }
    let dt = t.elapsed().as_secs_f64();
    let gib = written as f64 / (1u64 << 30) as f64;
    eprintln!(
        "{mode:>9}: {mb} MiB x {} copies in groups of {per_group} = {gib:.2} GiB in {dt:.3} s = {:.2} GiB/s",
        iters,
        gib / dt
    );

    // Host-side reference: what a plain memcpy of the same volume gets, so the
    // link can be separated from the staging copy that feeds it.
    let mut dst = vec![0u8; size];
    let t = Instant::now();
    for _ in 0..iters {
        dst.copy_from_slice(&bytes);
    }
    let dt = t.elapsed().as_secs_f64();
    eprintln!(
        "{:>9}: host memcpy {:.2} GiB/s",
        "memcpy",
        (written as f64 / (1u64 << 30) as f64) / dt
    );
    std::hint::black_box(&dst);
}
