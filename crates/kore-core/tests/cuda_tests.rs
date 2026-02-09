//! GPU integration tests for Kore CUDA backend.
//! Run with: cargo test -p kore-core --features cuda -- --nocapture

#![cfg(feature = "cuda")]

use kore_core::{Device, Tensor};

fn assert_close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < tol,
            "element {} differs: {} vs {} (tol={})",
            i, x, y, tol
        );
    }
}

// ============================================================================
// Device transfer tests
// ============================================================================

#[test]
fn test_cpu_to_cuda_roundtrip() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let cpu_tensor = Tensor::from_f32(&data, &[2, 3]);
    assert!(cpu_tensor.is_cpu());

    let gpu_tensor = cpu_tensor.cuda(0).expect("Failed to move to GPU");
    assert!(gpu_tensor.is_cuda());
    assert_eq!(gpu_tensor.device(), Device::Cuda(0));
    assert_eq!(gpu_tensor.shape().dims(), &[2, 3]);

    let back = gpu_tensor.cpu().expect("Failed to move back to CPU");
    assert!(back.is_cpu());
    assert_eq!(back.as_f32_slice().unwrap(), &data);
}

#[test]
fn test_cuda_noop_transfer() {
    let t = Tensor::from_f32(&[1.0, 2.0], &[2]);
    let gpu = t.cuda(0).unwrap();
    let gpu2 = gpu.cuda(0).unwrap(); // should be no-op
    assert!(gpu2.is_cuda());
}

// ============================================================================
// Element-wise binary ops
// ============================================================================

#[test]
fn test_cuda_add() {
    let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4]).cuda(0).unwrap();
    let b = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0], &[4]).cuda(0).unwrap();
    let c = a.add(&b).unwrap();
    assert!(c.is_cuda());
    let result = c.cpu().unwrap();
    assert_eq!(result.as_f32_slice().unwrap(), &[6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn test_cuda_sub() {
    let a = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0], &[4]).cuda(0).unwrap();
    let b = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4]).cuda(0).unwrap();
    let c = a.sub(&b).unwrap();
    let result = c.cpu().unwrap();
    assert_eq!(result.as_f32_slice().unwrap(), &[4.0, 4.0, 4.0, 4.0]);
}

#[test]
fn test_cuda_mul() {
    let a = Tensor::from_f32(&[2.0, 3.0, 4.0, 5.0], &[4]).cuda(0).unwrap();
    let b = Tensor::from_f32(&[3.0, 4.0, 5.0, 6.0], &[4]).cuda(0).unwrap();
    let c = a.mul(&b).unwrap();
    let result = c.cpu().unwrap();
    assert_eq!(result.as_f32_slice().unwrap(), &[6.0, 12.0, 20.0, 30.0]);
}

#[test]
fn test_cuda_div() {
    let a = Tensor::from_f32(&[10.0, 20.0, 30.0, 40.0], &[4]).cuda(0).unwrap();
    let b = Tensor::from_f32(&[2.0, 4.0, 5.0, 8.0], &[4]).cuda(0).unwrap();
    let c = a.div(&b).unwrap();
    let result = c.cpu().unwrap();
    assert_eq!(result.as_f32_slice().unwrap(), &[5.0, 5.0, 6.0, 5.0]);
}

// ============================================================================
// Unary ops
// ============================================================================

#[test]
fn test_cuda_neg() {
    let a = Tensor::from_f32(&[1.0, -2.0, 3.0, -4.0], &[4]).cuda(0).unwrap();
    let c = a.neg().unwrap();
    let result = c.cpu().unwrap();
    assert_eq!(result.as_f32_slice().unwrap(), &[-1.0, 2.0, -3.0, 4.0]);
}

#[test]
fn test_cuda_abs() {
    let a = Tensor::from_f32(&[-1.0, 2.0, -3.0, 4.0], &[4]).cuda(0).unwrap();
    let c = a.abs().unwrap();
    let result = c.cpu().unwrap();
    assert_eq!(result.as_f32_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_cuda_exp_log() {
    let a = Tensor::from_f32(&[0.0, 1.0, 2.0, 3.0], &[4]).cuda(0).unwrap();
    let b = a.exp().unwrap();
    let c = b.log().unwrap();
    let result = c.cpu().unwrap();
    let data = result.as_f32_slice().unwrap();
    assert_close(data, &[0.0, 1.0, 2.0, 3.0], 1e-5);
}

#[test]
fn test_cuda_sqrt() {
    let a = Tensor::from_f32(&[1.0, 4.0, 9.0, 16.0], &[4]).cuda(0).unwrap();
    let c = a.sqrt().unwrap();
    let result = c.cpu().unwrap();
    assert_close(result.as_f32_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0], 1e-6);
}

// ============================================================================
// Scalar ops
// ============================================================================

#[test]
fn test_cuda_add_scalar() {
    let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4]).cuda(0).unwrap();
    let c = a.add_scalar(10.0).unwrap();
    let result = c.cpu().unwrap();
    assert_eq!(result.as_f32_slice().unwrap(), &[11.0, 12.0, 13.0, 14.0]);
}

#[test]
fn test_cuda_mul_scalar() {
    let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4]).cuda(0).unwrap();
    let c = a.mul_scalar(3.0).unwrap();
    let result = c.cpu().unwrap();
    assert_eq!(result.as_f32_slice().unwrap(), &[3.0, 6.0, 9.0, 12.0]);
}

// ============================================================================
// Matrix multiplication
// ============================================================================

#[test]
fn test_cuda_matmul_small() {
    // [2,3] @ [3,2] → [2,2]
    let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).cuda(0).unwrap();
    let b = Tensor::from_f32(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).cuda(0).unwrap();
    let c = a.matmul(&b).unwrap();
    assert!(c.is_cuda());
    assert_eq!(c.shape().dims(), &[2, 2]);
    let result = c.cpu().unwrap();
    assert_eq!(result.as_f32_slice().unwrap(), &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_cuda_matmul_identity() {
    // A @ I = A
    let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).cuda(0).unwrap();
    let eye = Tensor::from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).cuda(0).unwrap();
    let c = a.matmul(&eye).unwrap();
    let result = c.cpu().unwrap();
    assert_eq!(result.as_f32_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_cuda_matmul_medium() {
    // 64x64 @ 64x64 — exercises the tiled2x2 kernel
    let n = 64;
    let a_data: Vec<f32> = (0..n * n).map(|i| (i % 7) as f32 * 0.1).collect();
    let b_data: Vec<f32> = (0..n * n).map(|i| (i % 11) as f32 * 0.1).collect();

    let a_cpu = Tensor::from_f32(&a_data, &[n, n]);
    let b_cpu = Tensor::from_f32(&b_data, &[n, n]);
    let c_cpu = a_cpu.matmul(&b_cpu).unwrap();

    let a_gpu = Tensor::from_f32(&a_data, &[n, n]).cuda(0).unwrap();
    let b_gpu = Tensor::from_f32(&b_data, &[n, n]).cuda(0).unwrap();
    let c_gpu = a_gpu.matmul(&b_gpu).unwrap().cpu().unwrap();

    assert_close(
        c_cpu.as_f32_slice().unwrap(),
        c_gpu.as_f32_slice().unwrap(),
        1e-3,
    );
}

#[test]
fn test_cuda_matmul_large() {
    // 256x128 @ 128x256 — larger tiled2x2 test
    let m = 256;
    let k = 128;
    let n = 256;
    let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();

    let a_cpu = Tensor::from_f32(&a_data, &[m, k]);
    let b_cpu = Tensor::from_f32(&b_data, &[k, n]);
    let c_cpu = a_cpu.matmul(&b_cpu).unwrap();

    let a_gpu = Tensor::from_f32(&a_data, &[m, k]).cuda(0).unwrap();
    let b_gpu = Tensor::from_f32(&b_data, &[k, n]).cuda(0).unwrap();
    let c_gpu = a_gpu.matmul(&b_gpu).unwrap().cpu().unwrap();

    assert_close(
        c_cpu.as_f32_slice().unwrap(),
        c_gpu.as_f32_slice().unwrap(),
        1e-2,
    );
}

// ============================================================================
// Chained GPU operations (no intermediate CPU transfers)
// ============================================================================

#[test]
fn test_cuda_chained_ops() {
    let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4]).cuda(0).unwrap();
    let b = Tensor::from_f32(&[2.0, 2.0, 2.0, 2.0], &[4]).cuda(0).unwrap();

    // (a + b) * a - b = (3*6*5*8) - (2*2*2*2) = (3,8,15,24) - ... wait
    // (a + b) = [3, 4, 5, 6]
    // (a + b) * a = [3, 8, 15, 24]
    // result - b = [1, 6, 13, 22]
    let c = a.add(&b).unwrap().mul(&a).unwrap().sub(&b).unwrap();
    assert!(c.is_cuda()); // all ops stayed on GPU
    let result = c.cpu().unwrap();
    assert_eq!(result.as_f32_slice().unwrap(), &[1.0, 6.0, 13.0, 22.0]);
}
