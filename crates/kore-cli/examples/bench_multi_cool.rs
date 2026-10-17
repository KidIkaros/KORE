//! Cool-die, interleaved A/B of the multi-column forward at a fixed context.
//!
//! `bench_multi` walks a single state from context 0, so each round is tens of
//! seconds of continuous full load: by the time it reports, the die is heat
//! saturated and the numbers are 3-4x the cool-die ones. This harness does what
//! `bench_context_cool` does -- prefill, then wait for the die to come back to
//! the target temperature and for the SM clock to ramp -- and then alternates
//! the batch widths in short chunks, each width owning its own state and KV
//! cache, both prefilled to the same context.
//!
//! Alternating inside one cool window is the only valid comparison on this box:
//! the clock drifts continuously, so a sweep measures the drift. The order is
//! flipped every round so the pair is not biased by which one runs first.
//!
//! Usage: `bench_multi_cool <gguf> <tokenizer> <prefill_ctx> <chunk> <rounds> [prompt] [widths]`
//!   widths default to `1,2,4`.
//!
//! Env: `KORE_COOL_C` (default 55), `KORE_COOL_MAX_S` (default 900),
//!      `KORE_WARMUP` (default 8).

use kore_nn::qwen35::{GpuState, Qwen35Gpu, Qwen35Model};
use std::path::Path;
use std::time::Instant;

fn gpu_temp_c() -> Option<u32> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"])
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

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
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
        .unwrap_or(1024);
    let chunk: usize = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let rounds: usize = std::env::args()
        .nth(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    let prompt = std::env::args()
        .nth(6)
        .unwrap_or_else(|| "The capital of France is".into());
    let widths: Vec<usize> = std::env::args()
        .nth(7)
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 4]);
    let cool_c: u32 = std::env::var("KORE_COOL_C")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(55);
    let cool_max_s: u64 = std::env::var("KORE_COOL_MAX_S")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(900);
    let warmup: usize = std::env::var("KORE_WARMUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);

    for &m in &widths {
        assert!(chunk % m == 0, "chunk {chunk} must be a multiple of width {m}");
    }

    let model = Qwen35Model::from_gguf(Path::new(&gguf)).expect("load model");
    let tok = tokenizers::Tokenizer::from_file(&tok_path).expect("tokenizer");
    let base = tok
        .encode(prompt.as_str(), true)
        .expect("encode")
        .get_ids()
        .to_vec();
    assert!(!base.is_empty(), "empty prompt");

    // The position range every width will cover, identical for each state:
    // the untimed clock ramp plus the timed rounds.
    let span = warmup + chunk * rounds;
    let max_ctx = prefill_ctx + span + 64;
    let prefill: Vec<u32> = (0..prefill_ctx)
        .map(|i| base[i % base.len()])
        .collect();
    let stream: Vec<u32> = (0..span).map(|i| base[i % base.len()]).collect();

    kore_vulkan::register_vulkan_backend().expect("vulkan backend");
    let gpu = Qwen35Gpu::from_cpu(&model, kore_core::Device::Vulkan(0)).expect("gpu materialize");

    // One state per width, every one prefilled to the same context so the pair
    // always sits at the same `seq_k`.
    let t_pre = Instant::now();
    let mut states: Vec<Vec<GpuState>> = Vec::with_capacity(widths.len());
    for _ in &widths {
        let mut st = gpu.new_state(max_ctx).expect("gpu state");
        for (i, &t) in prefill.iter().enumerate() {
            gpu.forward(&model, t, i, &mut st).expect("prefill fwd");
        }
        states.push(st);
    }
    let pre_s = t_pre.elapsed().as_secs_f64();

    // ---- cool the die (the prefilled KV survives in VRAM) ----
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

    // ---- warm the SM clock (untimed); a cooled die idles at ~300 MHz ----
    // One position at a time for every state, so they all stay at the same
    // context and hold identical content once the ramp is over.
    let mut pos = 0usize;
    let warmup = warmup.min(span);
    for _ in 0..warmup {
        for st in states.iter_mut() {
            gpu.forward(&model, stream[pos], prefill_ctx + pos, st)
                .expect("warmup fwd");
        }
        pos += 1;
    }
    let decode_ctx = prefill_ctx + pos;
    let clocks_before = gpu_clocks();

    // ---- timed rounds: each width advances `chunk` positions, on a cool die ----
    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(rounds); widths.len()];
    for r in 0..rounds {
        let mut order: Vec<usize> = (0..widths.len()).collect();
        if r % 2 == 1 {
            order.reverse();
        }
        for &i in &order {
            let m = widths[i];
            let t = Instant::now();
            let mut done = 0usize;
            while done < chunk {
                let at = pos + done;
                let c = m.min(chunk - done);
                gpu.forward_batch(&model, &stream[at..at + c], prefill_ctx + at, &mut states[i])
                    .expect("forward_batch");
                done += c;
            }
            samples[i].push(t.elapsed().as_secs_f64() * 1e3 / chunk as f64);
        }
        pos += chunk;
    }
    let clocks_after = gpu_clocks();

    println!("prefill_ctx={prefill_ctx} chunk={chunk} rounds={rounds} span={span}");
    println!("prefill: {} states x {} tok in {:.3}s", widths.len(), prefill_ctx, pre_s);
    println!(
        "cool   : {} C -> {} C in {:.0}s (target {cool_c} C)",
        hot.map(|t| t.to_string()).unwrap_or_else(|| "?".into()),
        cooled_to.map(|t| t.to_string()).unwrap_or_else(|| "?".into()),
        cool_s
    );
    println!("gpu@round0: {clocks_before}");
    println!("gpu@end   : {clocks_after}");
    println!("decode at ctx ~{decode_ctx}..{}", prefill_ctx + pos);
    let base_med = median(samples[0].clone());
    for (i, &m) in widths.iter().enumerate() {
        let med = median(samples[i].clone());
        println!(
            "m={m:<2}  {med:8.3} ms/token   {:7.1} tok/s   {:5.3}x vs m={}   all={:?}",
            1000.0 / med,
            base_med / med,
            widths[0],
            samples[i]
                .iter()
                .map(|v| (v * 100.0).round() / 100.0)
                .collect::<Vec<_>>()
        );
    }
}
