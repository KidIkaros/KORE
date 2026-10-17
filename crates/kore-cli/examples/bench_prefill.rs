//! Interleaved A/B of prompt consumption: token-at-a-time vs chunked `prefill`.
//!
//! `prefill` runs the multi-column pass over the prompt in chunks of the GEMV's
//! four-column width, so each weight read is amortised over four positions, and
//! it skips the vocabulary projection on every chunk but the last. It is
//! bit-identical to feeding `forward` one token at a time (`verify_m`), so the
//! only question this harness answers is what the chunking is worth.
//!
//! Both arms run from a fresh state, but the chunked arm loads the GPU harder per
//! unit of wall time, so a sweep would still charge thermal drift to whichever
//! arm went last. The arms therefore alternate in ABBA order within each round
//! and each round reports both, so drift is visible instead of being attributed
//! to the change.
//!
//! Each round also probes both states with the same token at the same position
//! and requires the tokens -- and the two arms' last prefill tokens -- to match.
//! A timing comparison between two states that disagree is not a comparison of
//! the same work.
//!
//! Usage: `bench_prefill <gguf> <tokenizer> [prefill_ctx] [rounds] [prompt]`

use kore_nn::qwen35::{Qwen35Gpu, Qwen35Model};
use std::path::Path;
use std::time::Instant;

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

/// One arm of one leg: consume the whole stream from a fresh state, then take
/// one extra single-token step to force the state to be read back. Returns
/// `(seconds, last prefill token, probe token)`.
fn arm(
    gpu: &Qwen35Gpu,
    model: &Qwen35Model,
    stream: &[u32],
    max_ctx: usize,
    chunked: bool,
) -> (f64, u32, u32) {
    let t0 = Instant::now();
    let mut st = gpu.new_state(max_ctx).expect("gpu state");
    let last = if chunked {
        gpu.prefill(model, stream, 0, &mut st).expect("prefill")
    } else {
        let mut next = 0u32;
        for (i, &t) in stream.iter().enumerate() {
            next = gpu.forward(model, t, i, &mut st).expect("prefill fwd");
        }
        next
    };
    let probe = gpu
        .forward(model, 42, stream.len(), &mut st)
        .expect("probe fwd");
    (t0.elapsed().as_secs_f64(), last, probe)
}

fn main() {
    let gguf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/bonsai2-2b/Ternary-Bonsai-2-2B-Q2_0.gguf".into());
    let tok_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "models/bonsai2-2b/tokenizer.json".into());
    let ctx: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let rounds: usize = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let prompt = std::env::args()
        .nth(5)
        .unwrap_or_else(|| "The capital of France is".into());

    let model = Qwen35Model::from_gguf(Path::new(&gguf)).expect("load model");
    let tok = tokenizers::Tokenizer::from_file(&tok_path).expect("tokenizer");
    let seed = tok
        .encode(prompt.as_str(), true)
        .expect("encode")
        .get_ids()
        .to_vec();
    assert!(!seed.is_empty(), "prompt must encode to at least one token");

    // A prompt of the requested length, built by repeating the seed so both arms
    // see the same deterministic stream (content does not affect cost).
    let stream: Vec<u32> = (0..ctx).map(|i| seed[i % seed.len()]).collect();
    let max_ctx = ctx + 8;

    kore_vulkan::register_vulkan_backend().expect("vulkan backend");
    let gpu = Qwen35Gpu::from_cpu(&model, kore_core::Device::Vulkan(0)).expect("gpu materialize");

    let mut seq_s: Vec<f64> = Vec::new();
    let mut chu_s: Vec<f64> = Vec::new();

    println!("prefill_ctx={ctx} rounds={rounds}  (A = `forward` per token, B = chunked `prefill`)");
    println!("gpu@start: {}", gpu_clocks());

    for r in 0..rounds {
        // ABBA within the round: each arm runs once in each half of the window,
        // so a monotone clock drift cancels out of the A/B.
        let mut a_times = Vec::new();
        let mut b_times = Vec::new();
        let mut a_last = 0u32;
        let mut b_last = 0u32;
        let mut a_probe = 0u32;
        let mut b_probe = 0u32;
        let order: [bool; 4] = if r % 2 == 0 {
            [true, false, false, true]
        } else {
            [false, true, true, false]
        };
        for chunked in order {
            let (dt, last, probe) = arm(&gpu, &model, &stream, max_ctx, chunked);
            if chunked {
                b_times.push(dt);
                b_last = last;
                b_probe = probe;
            } else {
                a_times.push(dt);
                a_last = last;
                a_probe = probe;
            }
        }
        if a_last != b_last {
            eprintln!("FAIL: chunked last token {b_last} != sequential {a_last}");
            std::process::exit(1);
        }
        if a_probe != b_probe {
            eprintln!("FAIL: state probe diverged: sequential {a_probe} vs chunked {b_probe}");
            std::process::exit(1);
        }
        seq_s.extend(&a_times);
        chu_s.extend(&b_times);
        println!(
            "round {r}: A {:.3},{:.3} s   B {:.3},{:.3} s   (last {a_last}, probe {a_probe})",
            a_times[0], a_times[1], b_times[0], b_times[1]
        );
    }

    let a = median(seq_s);
    let b = median(chu_s);
    println!("gpu@end: {}", gpu_clocks());
    println!("sequential  {a:.3} s  {:.1} tok/s", ctx as f64 / a);
    println!(
        "chunked     {b:.3} s  {:.1} tok/s   {:.3}x vs sequential",
        ctx as f64 / b,
        a / b
    );
}
