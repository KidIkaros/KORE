//! Per-op dispatch census for one decode step.
//!
//! A decode step is a strict dependency chain, so its cost is roughly
//! `count x exposed latency` per op. Counting the dispatches by label is what
//! turns "the small ops are about a quarter of the token" into a target list,
//! without GPU timestamp queries (this adapter does not expose them).
//!
//! Requires `KORE_OP_COUNT=1` (the census is compiled in but off until asked
//! for, since the counter is off the hot path otherwise).
//!
//! usage: KORE_OP_COUNT=1 cargo run --release -p kore-cli --example op_counts -- [gguf] [tokenizer] [n_decode] [max_ctx]
use kore_nn::qwen35::{Qwen35Gpu, Qwen35Model};
use std::path::Path;

fn main() {
    let gguf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/bonsai2-2b/Ternary-Bonsai-2-2B-Q2_0.gguf".into());
    let tok_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "models/bonsai2-2b/tokenizer.json".into());
    let n_decode: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let max_ctx: usize = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);

    if std::env::var("KORE_OP_COUNT")
        .map(|v| v == "0")
        .unwrap_or(true)
    {
        eprintln!("note: set KORE_OP_COUNT=1 or the census stays empty");
    }
    // The tokenizer is unused here, but the argument is kept so the call shape
    // matches `bonsai` (which needs it) and the same command line works.
    let _ = &tok_path;

    kore_vulkan::register_vulkan_backend().expect("vulkan backend");
    let model = Qwen35Model::from_gguf(Path::new(&gguf)).expect("load model");
    let gpu = Qwen35Gpu::from_cpu(&model, kore_core::Device::Vulkan(0)).expect("gpu materialize");
    let mut states = gpu.new_state(max_ctx).expect("gpu state");

    // Warm up: pipeline creation, buffer growth, and the first-touch decode
    // shapes are all one-off costs that would otherwise land in the census.
    let mut next = 1u32;
    for i in 0..2 {
        next = gpu
            .forward(&model, next, i, &mut states)
            .expect("warmup fwd");
    }

    kore_vulkan::reset_op_counts();
    let (hits0, miss0) = kore_vulkan::buffer_alloc_stats();
    let t0 = std::time::Instant::now();
    for i in 0..n_decode {
        next = gpu
            .forward(&model, next, 2 + i, &mut states)
            .expect("decode fwd");
    }
    let elapsed = t0.elapsed();
    let (hits1, miss1) = kore_vulkan::buffer_alloc_stats();

    let counts = kore_vulkan::op_counts();
    let total: u64 = counts.iter().map(|(_, c)| c).sum();
    let per = |c: u64| c as f64 / n_decode as f64;
    println!(
        "decode step census at ctx {}: {} dispatches over {n_decode} steps = {:.1}/step",
        2 + n_decode,
        total,
        per(total)
    );
    for (label, c) in &counts {
        println!("  {:>7.2}/step  {label}", per(*c));
    }
    let (dh, dm) = (hits1 - hits0, miss1 - miss0);
    println!(
        "buffer allocs/step: {:.1} reused, {:.1} create_buffer  ({} total drawn)",
        dh as f64 / n_decode as f64,
        dm as f64 / n_decode as f64,
        (dh + dm) as f64 / n_decode as f64
    );
    println!(
        "decode wall time: {:.2} ms/step ({:.1} tok/s, no sync between steps)",
        elapsed.as_secs_f64() * 1e3 / n_decode as f64,
        n_decode as f64 / elapsed.as_secs_f64()
    );
}
