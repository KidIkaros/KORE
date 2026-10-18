//! Context-sweep benchmark harness for the Qwen35/Bonsai GPU decode path.
//!
//! Usage:
//!   cargo run --release -p kore-cli --example bench_context -- \
//!       [gguf] [tokenizer] [prefill_ctx] [n_decode] [prompt]
//!
//! Prefills `prefill_ctx` tokens (the prompt cycled to that length), then times
//! `n_decode` greedy steps at that context. Reports prefill tok/s, decode tok/s,
//! peak VRAM (nvidia-smi) and free system RAM (/proc/meminfo).
//!
//! The point of the harness is the *decode at long context* number: decode cost is
//! O(seq_k) KV reads, so tok/s falls as context grows. Compare before/after the Q8
//! KV path at a fixed context.

use kore_nn::qwen35::{Qwen35Gpu, Qwen35Model};
use std::path::Path;
use std::time::Instant;

fn nvidia_mem_mib() -> Option<(u64, u64)> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let mut parts = s.trim().split(',').map(|p| p.trim().parse::<u64>());
    match (parts.next(), parts.next()) {
        (Some(Ok(u)), Some(Ok(t))) => Some((u, t)),
        _ => None,
    }
}

fn mem_available_mib() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.trim().split_whitespace().next()?.parse().ok()?;
            return Some(kb / 1024);
        }
    }
    None
}

fn main() {
    let gguf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/bonsai2-2b/Ternary-Bonsai-2-2B-Q2_0.gguf".into());
    let tok_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "models/bonsai2-2b/tokenizer.json".into());
    let prefill_ctx: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let n_decode: usize = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let prompt = std::env::args()
        .nth(5)
        .unwrap_or_else(|| "The capital of France is".into());

    let (vram0, vram_total) = nvidia_mem_mib().unwrap_or((0, 0));
    let ram0 = mem_available_mib().unwrap_or(0);

    let model = Qwen35Model::from_gguf(Path::new(&gguf)).expect("load model");
    let tok = tokenizers::Tokenizer::from_file(&tok_path).expect("tokenizer");

    let base = tok
        .encode(prompt.as_str(), true)
        .expect("encode")
        .get_ids()
        .to_vec();
    assert!(!base.is_empty(), "empty prompt encoding");

    // Build a prefill sequence of exactly `prefill_ctx` tokens by cycling the prompt.
    let prefill: Vec<u32> = (0..prefill_ctx).map(|i| base[i % base.len()]).collect();
    let max_ctx = prefill_ctx + n_decode + 8;

    kore_vulkan::register_vulkan_backend().expect("vulkan backend");
    let gpu = Qwen35Gpu::from_cpu(&model, kore_core::Device::Vulkan(0)).expect("gpu materialize");
    let mut states = gpu.new_state(max_ctx).expect("gpu state");

    // ---- prefill ----
    let t_pre = Instant::now();
    let mut next = 0u32;
    for (i, &t) in prefill.iter().enumerate() {
        next = gpu.forward(&model, t, i, &mut states).expect("prefill fwd");
    }
    let pre_s = t_pre.elapsed().as_secs_f64();

    // ---- decode at context `prefill_ctx` ----
    let t_dec = Instant::now();
    let mut out = Vec::with_capacity(n_decode);
    for k in 0..n_decode {
        let id = next;
        out.push(id);
        if id == model.cfg.eos {
            break;
        }
        next = gpu
            .forward(&model, id, prefill_ctx + k, &mut states)
            .expect("decode fwd");
    }
    let dec_s = t_dec.elapsed().as_secs_f64();

    let (vram1, _) = nvidia_mem_mib().unwrap_or((vram0, vram_total));
    let ram1 = mem_available_mib().unwrap_or(ram0);

    let pre_tps = prefill_ctx as f64 / pre_s;
    let dec_tps = out.len() as f64 / dec_s;

    println!("cfg: {:?}", model.cfg);
    println!("prefill_ctx={prefill_ctx} n_decode={n_decode} max_ctx={max_ctx}");
    println!(
        "prefill: {} tok in {:.3}s = {:.2} tok/s",
        prefill_ctx, pre_s, pre_tps
    );
    println!(
        "decode : {} tok in {:.3}s = {:.2} tok/s (at ctx {})",
        out.len(),
        dec_s,
        dec_tps,
        prefill_ctx
    );
    println!("tokens : {out:?}");
    println!(
        "vram   : {} -> {} MiB (total {} MiB)",
        vram0, vram1, vram_total
    );
    println!("ram    : free {} -> {} MiB", ram0, ram1);
}
