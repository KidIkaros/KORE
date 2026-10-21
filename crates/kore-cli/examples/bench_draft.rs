//! End-to-end A/B: plain greedy decode vs prompt-lookup speculative decode.
//!
//! The ratio is the interesting number, but the *gate* is identity: greedy
//! speculative decoding is only admissible if it returns the same tokens plain
//! greedy decoding returns, element for element. A verify pass that accepted a
//! draft on a batched argmax that differed from the sequential one -- or a
//! rewind that restored the recurrent state anywhere other than exactly where
//! `keep` columns would have left it -- shows up here as a divergence, before
//! any timing is reported.
//!
//! Every policy is an arm in the *same* run, rotated round by round, so the
//! ratios are all measured against the same plain-greedy baseline under the
//! same clock (this box has no GPU timers, so only within-run comparisons are
//! worth anything).
//!
//! Usage: `bench_draft <gguf> <tokenizer> <prompt> <n_new> <rounds> [policies]`
//! `policies` is a comma separated list of `k:min_match[:guard]`; default
//! `3:4`, which `draft_headroom` measures as the optimum. `3:4,2:4,1:4`
//! compares k, and `3:1:1,3:1:0` compares the payoff guard on a policy whose
//! trigger is deliberately too weak.

use kore_nn::qwen35::Qwen35Gpu;
use kore_nn::qwen35::Qwen35Model;
use kore_nn::spec::{DraftPolicy, SpecStats};
use std::path::Path;
use std::time::Instant;

fn arg<T: std::str::FromStr>(i: usize, default: T) -> T {
    std::env::args()
        .nth(i)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
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

fn parse_policy(s: &str) -> DraftPolicy {
    let mut it = s.split(':');
    let k = it.next().and_then(|x| x.parse().ok()).unwrap_or(3);
    let min_match = it.next().and_then(|x| x.parse().ok()).unwrap_or(4);
    // Optional third field: the payoff guard, so a run can carry guarded and
    // unguarded arms of the same policy and compare them under one clock.
    let guard = match it.next() {
        Some(g) => g != "0" && g != "off",
        None => true,
    };
    // Optional fourth field: `max_match`. The cap is the ceiling on the
    // confidence signal as well as on the search, so a schedule experiment has
    // to be able to raise it.
    let max_match = it.next().and_then(|x| x.parse().ok()).unwrap_or(16);
    DraftPolicy {
        k,
        min_match,
        guard,
        max_match,
        ..Default::default()
    }
    .clamped()
}

fn main() {
    let gguf: String = arg(
        1,
        "models/bonsai2-2b/Ternary-Bonsai-2-2B-Q2_0.gguf".to_string(),
    );
    let tok_path: String = arg(2, "models/bonsai2-2b/tokenizer.json".to_string());
    let prompt: String = arg(3, "The capital of France is".to_string());
    let n_new: usize = arg(4, 192);
    let rounds: usize = arg(5, 3).max(1);
    let spec_arg: String = arg(6, "3:4".to_string());
    let policies: Vec<DraftPolicy> = spec_arg.split(',').map(parse_policy).collect();

    let model = Qwen35Model::from_gguf(Path::new(&gguf)).expect("load model");
    let tok = tokenizers::Tokenizer::from_file(&tok_path).expect("tokenizer");
    let ids = tok
        .encode(prompt.as_str(), true)
        .expect("encode")
        .get_ids()
        .to_vec();
    let max_ctx = ids.len() + n_new + 64;

    kore_vulkan::register_vulkan_backend().expect("vulkan backend");
    let gpu = Qwen35Gpu::from_cpu(&model, kore_core::Device::Vulkan(0)).expect("gpu materialize");

    println!(
        "prompt {} tokens, {} new, {} rounds: {}",
        ids.len(),
        n_new,
        rounds,
        policies
            .iter()
            .map(|p| format!(
                "k={}/n>={}{}",
                p.k,
                p.min_match,
                if p.guard { "" } else { "/noguard" }
            ))
            .collect::<Vec<_>>()
            .join(" ")
    );

    // Arm 0 is plain greedy; arm 1 + i is `policies[i]`.
    let n_arms = 1 + policies.len();
    let mut ms: Vec<Vec<f64>> = vec![Vec::new(); n_arms];
    let mut tokens: Vec<Option<Vec<u32>>> = vec![None; n_arms];
    let mut stats: Vec<SpecStats> = vec![SpecStats::default(); n_arms];

    // Warm every arm before timing any of them. Pipelines are compiled lazily on
    // first use, so without this the arm that runs first in round 0 pays a
    // one-time cost for the whole run -- and when `rounds < n_arms` the rotation
    // never gives that arm a turn behind a warm GPU. It showed up as a policy
    // that never fired (0% fire) timing 1.20x *faster* than plain greedy, which
    // cannot happen: at 0% fire the spec path does strictly more work per token
    // than greedy does. Full `n_new` rather than a short pass, so the same
    // long-context shard counts and the same `Head::Last` width are compiled too.
    for arm in 0..n_arms {
        if arm == 0 {
            gpu.generate_greedy(&model, &ids, n_new, max_ctx)
                .expect("warm-up greedy");
        } else {
            gpu.generate_spec(&model, &ids, n_new, max_ctx, policies[arm - 1])
                .expect("warm-up spec");
        }
    }

    for round in 0..rounds {
        // Rotate which arm runs first, so no arm is systematically the one that
        // pays for the clock it warmed up.
        let first = round % n_arms;
        for a in 0..n_arms {
            let arm = (first + a) % n_arms;
            let t = Instant::now();
            if arm == 0 {
                let toks = gpu
                    .generate_greedy(&model, &ids, n_new, max_ctx)
                    .expect("generate_greedy");
                tokens[0] = Some(toks);
            } else {
                let (toks, st) = gpu
                    .generate_spec(&model, &ids, n_new, max_ctx, policies[arm - 1])
                    .expect("generate_spec");
                tokens[arm] = Some(toks);
                stats[arm] = st;
            }
            ms[arm].push(t.elapsed().as_secs_f64() * 1e3);
        }
    }

    // ---- the gate: every arm must be the same token stream as plain greedy ----
    let plain = tokens[0].clone().expect("plain arm");
    for arm in 1..n_arms {
        let got = tokens[arm].clone().expect("spec arm");
        if got != plain {
            let at = plain
                .iter()
                .zip(got.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(plain.len().min(got.len()));
            eprintln!(
                "FAIL: policy k={} n>={} diverged from greedy at token {at}: {:?} vs {:?}",
                policies[arm - 1].k,
                policies[arm - 1].min_match,
                plain.get(at),
                got.get(at)
            );
            std::process::exit(1);
        }
    }

    let p = median(ms[0].clone());
    println!("=== greedy decode (median of {rounds}) ===");
    println!(
        "  ms/token {:7.3}   {:6.1} tok/s",
        p / n_new as f64,
        1e3 / (p / n_new as f64)
    );
    println!(
        "  {:>10} {:>9} {:>9} {:>7} {:>8} {:>7}  {}",
        "policy", "ms/token", "tok/s", "ratio", "fire%", "acc%", "P(j)=0/1/2/3"
    );
    for arm in 1..n_arms {
        let s = median(ms[arm].clone());
        let st = &stats[arm];
        println!(
            "  {:>10} {:>9.3} {:>9.1} {:>7.3}x {:>7.1}% {:>6.1}%  {}/{}/{}/{}",
            format!(
                "k={} n>={}{}",
                policies[arm - 1].k,
                policies[arm - 1].min_match,
                if policies[arm - 1].guard {
                    ""
                } else {
                    " noguard"
                }
            ),
            s / n_new as f64,
            1e3 / (s / n_new as f64),
            p / s,
            100.0 * st.fire_rate(),
            100.0 * st.acceptance(),
            st.hist[0],
            st.hist[1],
            st.hist[2],
            st.hist[3]
        );
    }
    println!("  {} tokens identical to greedy in every arm", plain.len());
}
