//! Interleaved A/B of the multi-column forward's per-token cost.
//!
//! This box's clocks drift between runs, so a sweep is not a measurement: only
//! alternating configurations inside one run is. Every round runs each `m` back
//! to back on a freshly reset state, over the identical token stream and
//! positions, and the report is the median of the rounds.
//!
//! `forward_batch` returns tokens to the host (argmax readback), so each timed
//! region naturally ends in a device sync.
//!
//! Usage: `bench_multi <gguf> <tokenizer> <prompt> <n_tokens> <rounds>`

use kore_nn::qwen35::{GpuState, Qwen35Gpu, Qwen35Model};
use std::path::Path;
use std::time::Instant;

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
    let prompt = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "The capital of France is".into());
    let n_tokens: usize = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let rounds: usize = std::env::args()
        .nth(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let widths: Vec<usize> = std::env::args()
        .nth(6)
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 4]);

    let model = Qwen35Model::from_gguf(Path::new(&gguf)).expect("load model");
    let tok = tokenizers::Tokenizer::from_file(&tok_path).expect("tokenizer");
    let ids = tok
        .encode(prompt.as_str(), true)
        .expect("encode")
        .get_ids()
        .to_vec();
    assert!(!ids.is_empty(), "empty prompt");

    kore_vulkan::register_vulkan_backend().expect("vulkan backend");
    let gpu = Qwen35Gpu::from_cpu(&model, kore_core::Device::Vulkan(0)).expect("gpu materialize");

    let stream: Vec<u32> = (0..n_tokens).map(|i| ids[i % ids.len()]).collect();
    let max_ctx = n_tokens + 16;

    let run = |m: usize, st: &mut Vec<GpuState>| -> f64 {
        let t = Instant::now();
        let mut p = 0usize;
        while p < n_tokens {
            let c = m.min(n_tokens - p);
            gpu.forward_batch(&model, &stream[p..p + c], p, st)
                .expect("forward_batch");
            p += c;
        }
        t.elapsed().as_secs_f64() * 1e3
    };

    // Warm up every width once (JIT/clock ramp), so the first timed round is
    // not paying for it.
    for &m in &widths {
        let mut st = gpu.new_state(max_ctx).expect("state");
        let _ = run(m, &mut st);
    }

    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(rounds); widths.len()];
    for _ in 0..rounds {
        for (i, &m) in widths.iter().enumerate() {
            let mut st = gpu.new_state(max_ctx).expect("state");
            let ms = run(m, &mut st);
            samples[i].push(ms);
        }
    }

    println!("n_tokens={n_tokens} rounds={rounds}");
    let base = median(samples[0].clone());
    for (i, &m) in widths.iter().enumerate() {
        let med = median(samples[i].clone());
        let ms_per_tok = med / n_tokens as f64;
        println!(
            "m={m:<2}  {med:8.2} ms   {ms_per_tok:6.3} ms/token   {:6.1} tok/s   {:5.3}x vs m={}",
            1000.0 / ms_per_tok,
            base / med,
            widths[0]
        );
    }
}
