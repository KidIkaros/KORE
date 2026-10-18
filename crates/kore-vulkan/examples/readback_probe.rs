//! Two host-side numbers the routed-FFN profile cannot separate.
//!
//! `moe_ffn_gpu` starts its *route* timer before `ht.to(Device::Cpu)`, and that
//! download drains the queue. So `route` contains every GPU dispatch queued
//! before it, and reading it as "readback cost" would be wrong. This probe
//! answers the two questions directly:
//!
//! 1. What does an idle readback cost (poll + map + copy) when there is nothing
//!    queued? That is the floor of the 40-per-token round trip.
//! 2. What does the production upload path sustain, warm and with no compute in
//!    the queue? That separates PCIe from page-cache faulting.
//!
//! Run: `cargo run --release -p kore-vulkan --example readback_probe`

use kore_core::device::DeviceBackend;
use kore_core::{Device, Tensor};
use kore_vulkan::VulkanBackend;
use std::time::Instant;

fn main() {
    let b = VulkanBackend::new().expect("vulkan");
    let dev = Device::Vulkan(0);

    // --- 1. idle readback round trip -------------------------------------
    let x = Tensor::from_f32(&vec![0.5f32; 2048], &[1, 2048])
        .to(dev)
        .expect("upload");
    // Warm the staging buffers and the map path.
    for _ in 0..20 {
        let _ = x.to(Device::Cpu).expect("warm readback");
    }
    let t = Instant::now();
    const N: usize = 400;
    for _ in 0..N {
        let y = x.to(Device::Cpu).expect("readback");
        std::hint::black_box(y.as_f32_slice().map(|s| s[0]));
    }
    let per = t.elapsed().as_secs_f64() * 1e6 / N as f64;
    println!(
        "idle readback (2048 f32): {per:.1} us  ({:.0} round trips/s)",
        1e6 / per
    );

    // --- 2. raw upload throughput ----------------------------------------
    // 340 KB is one POCKET expert matrix. 512 of them is ~170 MB, close to one
    // token's worth of misses, and small enough to stay in RAM across repeats.
    let n_bytes = 340 * 1024;
    let bytes = vec![0x5Au8; n_bytes];
    for &count in &[64usize, 512] {
        let mut keep: Vec<Tensor> = Vec::with_capacity(count.min(64));
        // Hold only a few so the buffer pool does not grow without bound.
        let t = Instant::now();
        for i in 0..count {
            let s = b.upload_u8(&bytes, dev).expect("upload");
            let t = Tensor::from_storage(s, &[n_bytes]);
            if keep.len() < 64 {
                keep.push(t);
            } else {
                keep[i % 64] = t;
            }
        }
        let secs = t.elapsed().as_secs_f64();
        let mb = (count * n_bytes) as f64 / 1e6;
        println!(
            "upload {count} x {n_bytes} B: {:.1} ms  ({:.2} GB/s)",
            secs * 1e3,
            mb / 1000.0 / secs
        );
    }

    // --- 3. expert-shaped GEMV: kernel time vs sync time ------------------
    // One POCKET expert matrix is 512 rows out, 2048 in, Q2_K (84 B / 256 w).
    // 55 of these per layer is the whole routed FFN. Dispatch them back to
    // back and the number is kernel throughput; read back after each and the
    // number is the latency the model actually pays.
    //
    // The row counts are the interesting axis: 512 is one expert, 4096 is the
    // eight a token takes stacked into a single dispatch. Same bytes, same
    // arithmetic -- only the launch count differs.
    let k = 2048usize;
    let x = Tensor::from_f32(&vec![0.25f32; k], &[1, k])
        .to(dev)
        .expect("activation upload");
    for &n in &[512usize, 2048, 4096, 16384] {
        let w_len = n * (k / 256) * 84;
        let w = vec![0x55u8; w_len];
        let wt = Tensor::from_storage(
            kore_core::Storage::from_packed_bytes(kore_core::DType::U8, w_len, w),
            &[n, w_len / n],
        )
        .to(dev)
        .expect("weight upload");
        let reps = if n >= 16384 { 8 } else { 64 };

        std::hint::black_box(b.gemv_q2_k(&x, &wt, n, k).expect("gemv"));
        b.sync();
        let t = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(b.gemv_q2_k(&x, &wt, n, k).expect("gemv"));
        }
        b.sync();
        let piped = t.elapsed().as_secs_f64() * 1e6 / reps as f64;

        let t = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(b.gemv_q2_k(&x, &wt, n, k).expect("gemv"));
            b.sync();
        }
        let serial = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
        println!(
            "gemv q2_k n={n:<6} k={k}: pipelined {piped:7.1} us ({:5.1} GB/s) | \
             sync each {serial:7.1} us",
            (w_len as f64 / 1e9) / (piped / 1e6)
        );
    }
}
