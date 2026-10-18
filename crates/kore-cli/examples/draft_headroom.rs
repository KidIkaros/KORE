//! Does n-gram (prompt-lookup) drafting have anything to draft *on*?
//!
//! Speculative decoding only pays if the drafts a verify batch is fed are mostly
//! right, and that depends on the prompt, not on the kernel. But it can be
//! reconstructed exactly rather than guessed: with greedy sampling there is no
//! distribution to sample from, so a draft is accepted **iff it equals the
//! model's own greedy token at that position**. Which means: decode plainly,
//! keep the token stream, then replay a drafter over it offline, scoring every
//! draft against the token the model actually went on to choose.
//!
//! That gives the acceptance distribution `P(j)` for free -- before any of the
//! engine work the feature would need. Combine it with an in-run, interleaved
//! measurement of what one `m`-column pass costs, and the projected tok/s of the
//! finished loop falls straight out:
//!
//!   fired step:  one `m = 1 + drafts` pass, emits `1 + j` tokens
//!   missed step: one `m = 1` pass, emits 1 token
//!
//! Everything here is a *walk* over the recorded stream, so the projection
//! reproduces the real loop's decisions exactly: a fired step skips the
//! positions it just committed, which is where the money comes from.
//!
//! Usage: `draft_headroom <gguf> <tokenizer> <n_new> <prompt> [rounds]`

use kore_nn::qwen35::{GpuState, Qwen35Gpu, Qwen35Model};
use kore_nn::spec::longest_match;
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

/// The shipped matcher, over the whole recorded history (no `search_back`
/// window): this replay scores the drafter that actually ships, so the
/// acceptance numbers transfer to `bench_draft`.
fn matched(hist: &[u32], t_next: u32, k: usize, max_len: usize) -> (usize, Vec<u32>) {
    longest_match(hist, t_next, k, max_len, usize::MAX)
}

struct Trigger {
    name: &'static str,
    min_match: usize,
    min_drafts: usize,
}

/// One walk of the recorded stream under one trigger.
struct Walk {
    steps: usize,
    fires: usize,
    drafts: usize,
    accepted: usize,
    emitted: usize,
    hist: [usize; 5], // index 0..=3 by accepted-prefix length, 4 = missed
    ms: f64,
}

fn walk(all: &[u32], start: usize, k: usize, max_len: usize, t: &Trigger, step_ms: &[f64; 5]) -> Walk {
    let mut w = Walk {
        steps: 0,
        fires: 0,
        drafts: 0,
        accepted: 0,
        emitted: 0,
        hist: [0; 5],
        ms: 0.0,
    };
    let mut p = start;
    // Only score positions where the whole draft window is known, so the tail
    // cannot silently bank a partial match as a full one.
    while p + 1 + k < all.len() {
        w.steps += 1;
        let t_next = all[p];
        let (l, drafts) = matched(&all[..p], t_next, k, max_len);
        if l >= t.min_match && drafts.len() >= t.min_drafts {
            let m = 1 + drafts.len();
            let j = drafts
                .iter()
                .zip(&all[p + 1..])
                .take_while(|(d, c)| d == c)
                .count();
            w.fires += 1;
            w.drafts += drafts.len();
            w.accepted += j;
            w.hist[j] += 1;
            w.ms += step_ms[m];
            w.emitted += 1 + j;
            p += 1 + j;
        } else {
            w.hist[4] += 1;
            w.ms += step_ms[1];
            w.emitted += 1;
            p += 1;
        }
    }
    w
}

fn main() {
    let gguf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/bonsai2-2b/Ternary-Bonsai-2-2B-Q2_0.gguf".into());
    let tok_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "models/bonsai2-2b/tokenizer.json".into());
    let n_new: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(192);
    let prompt = std::env::args()
        .nth(4)
        .unwrap_or_else(|| "The capital of France is".into());
    let rounds: usize = std::env::args()
        .nth(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

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

    // ---- 1. a plain greedy decode: this is the ground truth the drafter is
    // scored against, and the token stream the timing walk uses.
    let n_prefill = ids.len();
    let max_ctx = (n_prefill + n_new).max(256 + 32 * 4) + 64;
    let t_decode = Instant::now();
    let all: Vec<u32> = {
        let mut st = gpu.new_state(max_ctx).expect("state");
        let mut next = gpu.prefill(&model, &ids, 0, &mut st).expect("prefill");
        let mut v = ids.clone();
        for k in 0..n_new {
            if next == model.cfg.eos {
                break;
            }
            v.push(next);
            next = gpu.forward(&model, next, n_prefill + k, &mut st)
                .expect("forward");
        }
        v
    };
    let decode_ms = t_decode.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "recorded {} tokens ({} new) at {:.2} ms/token (m=1, hot) in {:.1}s",
        all.len(),
        all.len() - n_prefill,
        decode_ms / (all.len() - n_prefill).max(1) as f64,
        decode_ms / 1e3
    );
    println!("=== generated ===");
    println!(
        "{}",
        tok.decode(&all[n_prefill..], true).unwrap_or_default()
    );

    // ---- 2. per-m call cost, interleaved in one run at a realistic context.
    // The pass cost is flat in context past the attention-sharding threshold, so
    // a cyclic replay of the recorded stream is enough to hold the shapes still
    // without needing a stream longer than what was generated.
    let ctx0 = 256usize;
    let n_calls = 32usize;
    let mut timing: Vec<u32> = Vec::new();
    while timing.len() < ctx0 + n_calls * 4 + 8 {
        timing.extend_from_slice(&all);
    }
    let mut step_ms = [0.0f64; 5];
    {
        let widths = [1usize, 2, 3, 4];
        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); widths.len()];
        let run = |m: usize, st: &mut Vec<GpuState>, pos: &mut usize| -> f64 {
            let t = Instant::now();
            for _ in 0..n_calls {
                if m == 1 {
                    gpu.forward(&model, timing[*pos], *pos, st).expect("forward");
                    *pos += 1;
                } else {
                    gpu.forward_batch(&model, &timing[*pos..*pos + m], *pos, st)
                        .expect("forward_batch");
                    *pos += m;
                }
            }
            t.elapsed().as_secs_f64() * 1e3 / n_calls as f64
        };
        // warm every width (clock ramp) before any timing
        for &m in &widths {
            let mut st = gpu.new_state(max_ctx).expect("state");
            gpu.prefill(&model, &timing[..ctx0], 0, &mut st).expect("prefill");
            let mut pos = ctx0;
            let _ = run(m, &mut st, &mut pos);
        }
        for _ in 0..rounds {
            for (i, &m) in widths.iter().enumerate() {
                let mut st = gpu.new_state(max_ctx).expect("state");
                gpu.prefill(&model, &timing[..ctx0], 0, &mut st).expect("prefill");
                let mut pos = ctx0;
                samples[i].push(run(m, &mut st, &mut pos));
            }
        }
        println!("=== one pass at ctx {} (median of {rounds}, hot) ===", ctx0);
        for (i, &m) in widths.iter().enumerate() {
            let med = median(samples[i].clone());
            step_ms[m] = med;
            println!(
                "  m={m}  {med:7.3} ms/pass   {:6.3} ms/token (fully accepted)",
                med / (m as f64)
            );
        }
    }

    // ---- 3. replay the drafter over the recorded stream.
    // `k` is how many drafts a fired step feeds: the pass costs `step_ms[1+k]`,
    // so a *smaller* k has a much lower break-even when matches are short --
    // m = 2 recovers 1.38x if a single draft lands, and only needs to land 45%
    // of the time. The two knobs trade off, so sweep both.
    let max_len = 16usize;
    let triggers = [
        Trigger { name: "n>=1", min_match: 1, min_drafts: 1 },
        Trigger { name: "n>=2", min_match: 2, min_drafts: 1 },
        Trigger { name: "n>=3", min_match: 3, min_drafts: 1 },
        Trigger { name: "n>=4", min_match: 4, min_drafts: 1 },
        Trigger { name: "n>=6", min_match: 6, min_drafts: 1 },
        Trigger { name: "n>=8", min_match: 8, min_drafts: 1 },
    ];
    let base = step_ms[1];
    println!(
        "=== drafter replay over {} recorded positions (max_len={max_len}) ===",
        all.len().saturating_sub(n_prefill)
    );
    println!(
        "  {:>2} {:<6} {:>6} {:>6} {:>6}  {:<20} {:>9} {:>8}",
        "k", "trig", "fire%", "acc%", "P(j>=2)", "P(j)=0/1/2/3/miss", "ms/token", "vs base"
    );
    let mut best = (0.0f64, 0usize, String::new());
    for k in [1usize, 2, 3] {
        for t in &triggers {
            if t.min_drafts > k {
                continue;
            }
            let agg = walk(&all, n_prefill, k, max_len, t, &step_ms);
            let p = |j: usize| agg.hist[j] as f64 / agg.fires.max(1) as f64;
            let ms_tok = agg.ms / agg.emitted.max(1) as f64;
            let ratio = base / ms_tok;
            if ratio > best.0 {
                best = (ratio, k, t.name.to_string());
            }
            println!(
                "  {:>2} {:<6} {:>5.1}% {:>5.1}% {:>5.0}%  {}/{}/{}/{}/{}          {:>8.3} {:>7.2}x",
                k,
                t.name,
                100.0 * agg.fires as f64 / agg.steps.max(1) as f64,
                100.0 * agg.accepted as f64 / agg.drafts.max(1) as f64,
                100.0 * (p(2) + p(3)),
                agg.hist[0],
                agg.hist[1],
                agg.hist[2],
                agg.hist[3],
                agg.hist[4],
                ms_tok,
                ratio,
            );
        }
    }
    println!(
        "  best: k={} {} -> {:.2}x  (base = {base:.3} ms/token plain decode; a draft is accepted iff it equals the model's own greedy token)",
        best.1, best.2, best.0
    );
}
