//! K-quant correctness, checked against two independent oracles.
//!
//! The fixture bytes and their expected dequantized outputs live in
//! `tests/data/kquant/`, produced by `tests/data/kquant/gen.py` using
//! `gguf.quants` -- llama.cpp's own reference dequantizers. The fixtures cover
//! two real GGUFs' payloads (`Q4_K_M` mixes Q4_K/Q5_K/Q6_K) plus pseudo-random
//! blocks, which are the only cases that exercise every nibble, every high bit
//! and every scale/min packing branch.
//!
//! Three claims are checked:
//!
//! 1. `kore_kernels::k_quant` reproduces the reference bit-for-bit.
//! 2. `oxillama_quant`'s kernels reproduce it too (a second, independent
//!    implementation, and the one the qwen35 CPU lane dispatches through).
//! 3. A fused GEMV over the same bytes equals a dequant-then-dot-product of the
//!    same block, which is the property that actually matters for inference.

use kore_kernels::k_quant;
use oxillama_gguf::GgufTensorType;
use oxillama_quant::{global_dispatcher, QuantKernel, QuantTensor};

const FIXTURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/kquant");

/// Every fixture case: (name, GGML type id).
const CASES: &[(&str, u32)] = &[
    ("real_q4_k", 12),
    ("real2_q4_k", 12),
    ("rand_q4_k", 12),
    ("zero_q4_k", 12),
    ("real_q5_k", 13),
    ("rand_q5_k", 13),
    ("zero_q5_k", 13),
    ("real_q6_k", 14),
    ("real2_q6_k", 14),
    ("rand_q6_k", 14),
    ("zero_q6_k", 14),
];

fn read_fixture(name: &str, ext: &str) -> Vec<u8> {
    let path = format!("{FIXTURE_DIR}/{name}.{ext}");
    std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"))
}

fn expected_f32(name: &str) -> Vec<f32> {
    read_fixture(name, "f32")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Compare two f32 slices bit-for-bit and report the first divergence usefully.
fn assert_bit_exact(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    let mut bad = 0usize;
    let mut first = None;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g.to_bits() != w.to_bits() {
            if first.is_none() {
                first = Some((i, *g, *w));
            }
            bad += 1;
        }
    }
    if let Some((i, g, w)) = first {
        panic!(
            "{label}: {bad}/{} elements differ; first at {i}: got {g:?} ({:#010x}) want {w:?} ({:#010x})",
            got.len(),
            g.to_bits(),
            w.to_bits()
        );
    }
}

fn kernel_for(type_id: u32) -> std::sync::Arc<dyn QuantKernel> {
    let ty = GgufTensorType::from_u32(type_id).expect("known gguf type");
    global_dispatcher()
        .get_kernel(ty)
        .unwrap_or_else(|e| panic!("no oxillama kernel for type {type_id}: {e}"))
}

#[test]
fn dequant_matches_llama_cpp_reference() {
    for &(name, type_id) in CASES {
        let raw = read_fixture(name, "raw");
        let want = expected_f32(name);
        let got = k_quant::dequantize(type_id, &raw, want.len())
            .unwrap_or_else(|e| panic!("{name}: dequantize failed: {e}"));
        assert_bit_exact(name, &got, &want);
    }
}

#[test]
fn oxillama_kernel_matches_the_same_reference() {
    // Guards the assumption that the third-party kernels the CPU lane uses agree
    // with the layout we document and test against.
    for &(name, type_id) in CASES {
        let raw = read_fixture(name, "raw");
        let want = expected_f32(name);
        let kernel = kernel_for(type_id);
        let block_size = k_quant::QK_K;
        let block_bytes = k_quant::block_bytes(type_id).unwrap();
        let mut got = vec![0.0f32; want.len()];
        for b in 0..want.len() / block_size {
            let blk = &raw[b * block_bytes..(b + 1) * block_bytes];
            kernel
                .dequant_block(blk, &mut got[b * block_size..(b + 1) * block_size])
                .unwrap_or_else(|e| panic!("{name}: oxillama dequant_block failed: {e}"));
        }
        assert_bit_exact(&format!("{name} (oxillama)"), &got, &want);
    }
}

#[test]
fn fused_gemv_equals_dequant_then_dot() {
    // Each fixture is a whole number of 256-weight blocks, so it doubles as a
    // [1, K] weight matrix. A fused GEMV must agree with explicitly
    // dequantizing and dot-producting.
    for &(name, type_id) in CASES {
        let raw = read_fixture(name, "raw");
        let deq = k_quant::dequantize(type_id, &raw, raw.len() / k_quant::block_bytes(type_id).unwrap() * k_quant::QK_K).unwrap();
        let k = deq.len();
        let x: Vec<f32> = (0..k)
            .map(|i| ((i as f32) * 0.017 - 1.0).sin() * (1.0 + (i % 7) as f32 * 0.1))
            .collect();
        let want: f32 = deq.iter().zip(&x).map(|(w, a)| w * a).sum();

        let ty = GgufTensorType::from_u32(type_id).unwrap();
        let qt = QuantTensor::new(raw.clone(), vec![1, k], ty);
        let kernel = kernel_for(type_id);
        let mut got = vec![0.0f32; 1];
        kernel
            .gemv(&qt, &x, &mut got)
            .unwrap_or_else(|e| panic!("{name}: gemv failed: {e}"));

        let tol = 1e-4 * want.abs().max(1.0);
        assert!(
            (got[0] - want).abs() <= tol,
            "{name}: gemv {} vs dequant-dot {} (tol {tol:e})",
            got[0],
            want
        );
    }
}
