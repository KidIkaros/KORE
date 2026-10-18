//! K-quant GEMVs on the GPU lane (Q2_K, Q3_K).
//!
//! Two properties matter, and they are different in kind:
//!
//! * the kernel must agree with `k_quant`'s specification (which is itself
//!   bit-exact against gguf-py), within f32 reassociation -- a GPU reduction
//!   cannot match a scalar dot bit for bit and is not asked to;
//! * the *same* row must give the *same* bits whether it was computed alone or
//!   as part of a batch. That is the contract lossless speculative decoding
//!   rests on, so it is asserted exactly, not with a tolerance.

use kore_core::{DType, Device, Storage, Tensor};
use kore_kernels::k_quant;
use kore_vulkan::VulkanBackend;

const Q2_K_TYPE_ID: u32 = 10;
const BYTES: usize = 84;
const QK: usize = 256;

/// Deterministic pseudo-random bytes with a finite pair of f16 scales, so a
/// tolerance comparison can never be satisfied or broken by an inf.
fn packed(n: usize, k: usize, seed: u32) -> Vec<u8> {
    let blocks = n * (k / QK);
    let mut raw = vec![0u8; blocks * BYTES];
    let mut state = seed | 1;
    for (b, blk) in raw.chunks_exact_mut(BYTES).enumerate() {
        // d = 1.0, dmin = 0.25 (0x3400), then random scales and payload.
        blk[80] = 0x00;
        blk[81] = 0x3C;
        blk[82] = 0x00;
        blk[83] = 0x34;
        for i in 0..80 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // Vary `b` too: identical blocks per row would hide row-offset bugs.
            blk[i] = ((state >> 16) as u8) ^ (b as u8);
        }
    }
    raw
}

fn rows(m: usize, k: usize) -> Vec<f32> {
    (0..m * k)
        .map(|i| (((i * 37) % 101) as f32 * 0.021 - 1.0) * (1.0 + (i % 5) as f32 * 0.1))
        .collect()
}

#[test]
fn gpu_q2_k_matches_the_specification() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return, // no adapter on this box
    };

    // 8 output rows and two blocks per row, so `blocks_per_row > 1` is covered.
    let (n, k, m) = (8usize, 512usize, 4usize);
    let raw = packed(n, k, 0x51EED);
    let deq = k_quant::dequantize(Q2_K_TYPE_ID, &raw, n * k).expect("spec dequant");

    let len = raw.len();
    let w = Tensor::from_storage(Storage::from_packed_bytes(DType::U8, len, raw), &[len]);
    let x_data = rows(m, k);
    let x = Tensor::from_f32(&x_data, &[m, k]);
    let y = backend.gemv_q2_k(&x, &w, n, k).expect("gemv_q2_k");
    assert_eq!(y.shape().dims(), &[m, n]);
    let y = y.to(Device::Cpu).expect("to cpu");
    let got = y.as_f32_slice().expect("f32");

    for r in 0..m {
        for c in 0..n {
            // f64 reference: the kernel sums in f32 with a different order, so
            // compare relatively rather than absolutely.
            let want: f64 = (0..k)
                .map(|j| f64::from(deq[c * k + j]) * f64::from(x_data[r * k + j]))
                .sum();
            let got = f64::from(got[r * n + c]);
            let tol = 1e-4 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tol,
                "row {r} col {c}: gpu {got} vs spec {want} (diff {:e})",
                (got - want).abs()
            );
        }
    }
}

#[test]
fn gpu_q2_k_row_bits_do_not_depend_on_the_batch() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    let (n, k, m) = (5usize, 256usize, 4usize);
    // One block per row keeps the comparison about the batch axis.
    let raw = packed(n, k, 0xB47C4);
    let len = raw.len();
    let w = Tensor::from_storage(Storage::from_packed_bytes(DType::U8, len, raw), &[len]);

    let x_data = rows(m, k);
    let x = Tensor::from_f32(&x_data, &[m, k]);
    let batched = backend
        .gemv_q2_k(&x, &w, n, k)
        .expect("gemv_q2_k")
        .to(Device::Cpu)
        .expect("to cpu");
    let batched = batched.as_f32_slice().expect("f32").to_vec();

    for r in 0..m {
        let alone = Tensor::from_f32(&x_data[r * k..(r + 1) * k], &[1, k]);
        let y = backend
            .gemv_q2_k(&alone, &w, n, k)
            .expect("gemv_q2_k")
            .to(Device::Cpu)
            .expect("to cpu");
        let y = y.as_f32_slice().expect("f32");
        for c in 0..n {
            assert_eq!(
                y[c].to_bits(),
                batched[r * n + c].to_bits(),
                "row {r} col {c}: m=1 {} vs m={m} {}",
                y[c],
                batched[r * n + c]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Q3_K
//
// Same two properties, plus one of its own: the kernel reads a buffer the host
// padded from 110 to 112 bytes per block, so a test that feeds it the file's own
// packing would pass while the real upload path read garbage. The tests below
// therefore upload exactly what `repack_q3_k` produces and keep the file's
// packing only for the reference.
// ---------------------------------------------------------------------------

const Q3_K_TYPE_ID: u32 = 11;
const Q3_K_BYTES: usize = 110;

fn packed_q3_k(n: usize, k: usize, seed: u32) -> Vec<u8> {
    let blocks = n * (k / QK);
    let mut raw = vec![0u8; blocks * Q3_K_BYTES];
    let mut state = seed | 1;
    for (b, blk) in raw.chunks_exact_mut(Q3_K_BYTES).enumerate() {
        // d = 1.0 in the low half of the last field, so no inf can enter.
        blk[108] = 0x00;
        blk[109] = 0x3C;
        for i in 0..108 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            blk[i] = ((state >> 16) as u8) ^ (b as u8);
        }
    }
    raw
}

fn upload(w: &[u8]) -> Tensor {
    let len = w.len();
    Tensor::from_storage(
        Storage::from_packed_bytes(DType::U8, len, w.to_vec()),
        &[len],
    )
}

#[test]
fn gpu_q3_k_matches_the_specification() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    let (n, k, m) = (8usize, 512usize, 4usize);
    let raw = packed_q3_k(n, k, 0x3CAFE);
    let deq = k_quant::dequantize(Q3_K_TYPE_ID, &raw, n * k).expect("spec dequant");
    let padded = k_quant::repack_q3_k(&raw).expect("repack");

    let w = upload(&padded);
    let x_data = rows(m, k);
    let x = Tensor::from_f32(&x_data, &[m, k]);
    let y = backend.gemv_q3_k(&x, &w, n, k).expect("gemv_q3_k");
    assert_eq!(y.shape().dims(), &[m, n]);
    let y = y.to(Device::Cpu).expect("to cpu");
    let got = y.as_f32_slice().expect("f32");

    for r in 0..m {
        for c in 0..n {
            let want: f64 = (0..k)
                .map(|j| f64::from(deq[c * k + j]) * f64::from(x_data[r * k + j]))
                .sum();
            let got = f64::from(got[r * n + c]);
            let tol = 1e-4 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tol,
                "row {r} col {c}: gpu {got} vs spec {want} (diff {:e})",
                (got - want).abs()
            );
        }
    }
}

#[test]
fn gpu_q3_k_row_bits_do_not_depend_on_the_batch() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    let (n, k, m) = (5usize, 256usize, 4usize);
    let raw = packed_q3_k(n, k, 0x9E3);
    let w = upload(&k_quant::repack_q3_k(&raw).expect("repack"));

    let x_data = rows(m, k);
    let x = Tensor::from_f32(&x_data, &[m, k]);
    let batched = backend
        .gemv_q3_k(&x, &w, n, k)
        .expect("gemv_q3_k")
        .to(Device::Cpu)
        .expect("to cpu");
    let batched = batched.as_f32_slice().expect("f32").to_vec();

    for r in 0..m {
        let alone = Tensor::from_f32(&x_data[r * k..(r + 1) * k], &[1, k]);
        let y = backend
            .gemv_q3_k(&alone, &w, n, k)
            .expect("gemv_q3_k")
            .to(Device::Cpu)
            .expect("to cpu");
        let y = y.as_f32_slice().expect("f32");
        for c in 0..n {
            assert_eq!(
                y[c].to_bits(),
                batched[r * n + c].to_bits(),
                "row {r} col {c}: m=1 {} vs m={m} {}",
                y[c],
                batched[r * n + c]
            );
        }
    }
}
