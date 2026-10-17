//! Like `bench_context`, but thermally honest: it cools the die between the
//! prefill and the timed decode loop.
//!
//! Usage:
//!   cargo run --release -p kore-cli --example bench_context_cool -- \
//!       [gguf] [tokenizer] [prefill_ctx] [n_decode] [prompt]
//!
//! Why this exists: `bench_context` measures decode *immediately* after the
//! prefill, so the decode window is sampled at the hottest, most-throttled point
//! of the run. The prefill duration grows with the context (a 4096-token prefill
//! is ~80 s of full GPU load, an 8192-token one ~250 s), which means
//! `bench_context`'s decode number conflates *context* with *heat*: the ctx 5
//! number is taken on a cold die and the ctx 8192 number on a heat-soaked one.
//! That is exactly the comparison this file exists to remove.
//!
//! Env:
//!   KORE_COOL_C      target die temperature before decoding (default 55)
//!   KORE_COOL_MAX_S  give up waiting after this many seconds (default 900)
//!
//! The KV cache and the weights stay resident in VRAM across the pause, so the
//! post-pause decode is a valid measurement of the same context.

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

fn gpu_temp_c() -> Option<u32> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

fn gpu_clocks() -> String {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.max.sm",
            "--format=csv,noheader",
        ])
        .output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        Err(_) => "n/a".into(),
    }
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
    let cool_c: u32 = std::env::var("KORE_COOL_C")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(55);
    let cool_max_s: u64 = std::env::var("KORE_COOL_MAX_S")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(900);

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

    let prefill: Vec<u32> = (0..prefill_ctx).map(|i| base[i % base.len()]).collect();
    // + a 32-token margin: the KV cache must be able to hold prefill + warmup +
    // n_decode (a smaller max_ctx would trip the state's bounds check).
    let max_ctx = prefill_ctx + n_decode + 64;

    kore_vulkan::register_vulkan_backend().expect("vulkan backend");
    let gpu = Qwen35Gpu::from_cpu(&model, kore_core::Device::Vulkan(0)).expect("gpu materialize");
    let mut states = gpu.new_state(max_ctx).expect("gpu state");

    // ---- prefill (discarded for timing: its only job is to reach the context) ----
    let t_pre = Instant::now();
    let mut next = 0u32;
    for (i, &t) in prefill.iter().enumerate() {
        next = gpu.forward(&model, t, i, &mut states).expect("prefill fwd");
    }
    let pre_s = t_pre.elapsed().as_secs_f64();

    // ---- cool the die (VRAM state survives) ----
    let hot = gpu_temp_c();
    let t_cool = Instant::now();
    let mut cooled_to = hot;
    loop {
        let t = match gpu_temp_c() {
            Some(t) => t,
            None => break,
        };
        cooled_to = Some(t);
        if t <= cool_c || t_cool.elapsed().as_secs() >= cool_max_s {
            break;
        }
        std::thread::sleep(std::time::Duration::from_secs(3));
    }
    let cool_s = t_cool.elapsed().as_secs_f64();

    // ---- warm the SM clock (untimed) ----
    // A cooled die idles at 300 MHz; the first few forward passes are spent
    // ramping to boost. Timing those would measure the ramp, not the model (the
    // first attempt at this averaged 81 ms/token for the first four against
    // 15.5 ms/token for the last four).
    let warmup: usize = std::env::var("KORE_WARMUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    for k in 0..warmup {
        next = gpu
            .forward(&model, next, prefill_ctx + k, &mut states)
            .expect("warmup fwd");
    }
    let decode_ctx = prefill_ctx + warmup;
    let temp_before = gpu_clocks();

    // ---- decode at context `decode_ctx`, on a cool, clock-ramped die ----
    let t_dec = Instant::now();
    let mut out = Vec::with_capacity(n_decode);
    let mut per_tok_ms = Vec::with_capacity(n_decode);
    for k in 0..n_decode {
        let id = next;
        out.push(id);
        if id == model.cfg.eos {
            break;
        }
        let t0 = Instant::now();
        next = gpu
            .forward(&model, id, decode_ctx + k, &mut states)
            .expect("decode fwd");
        per_tok_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let dec_s = t_dec.elapsed().as_secs_f64();
    let temp_after = gpu_clocks();

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
        "cool   : {} C -> {} C in {:.0}s (target {cool_c} C)",
        hot.map(|t| t.to_string()).unwrap_or_else(|| "?".into()),
        cooled_to
            .map(|t| t.to_string())
            .unwrap_or_else(|| "?".into()),
        cool_s
    );
    println!("gpu@dec_start: {temp_before}");
    println!(
        "decode : {} tok in {:.3}s = {:.2} tok/s (at ctx {})",
        out.len(),
        dec_s,
        dec_tps,
        decode_ctx
    );
    println!("gpu@dec_end  : {temp_after}");
    // Per-token ms tells us whether the window itself drifts (throttle creep).
    if !per_tok_ms.is_empty() {
        let head = &per_tok_ms[..per_tok_ms.len().min(4)];
        let tail = &per_tok_ms[per_tok_ms.len().saturating_sub(4)..];
        let first = head.iter().sum::<f64>() / head.len() as f64;
        let last = tail.iter().sum::<f64>() / tail.len() as f64;
        let mut sorted = per_tok_ms.clone();
        sorted.sort_by(f64::total_cmp);
        let median = sorted[sorted.len() / 2];
        println!(
            "per-tok ms : first4 {first:.2}  last4 {last:.2}  median {median:.2}  min {:.2}",
            sorted[0]
        );
    }
    println!("tokens : {out:?}");
    println!(
        "vram   : {} -> {} MiB (total {} MiB)",
        vram0, vram1, vram_total
    );
    println!("ram    : free {} -> {} MiB", ram0, ram1);
}
