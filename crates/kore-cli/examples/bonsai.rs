//! Bonsai-2 (qwen35/ternary) end-to-end smoke test: load GGUF + tokenizer,
//! greedy-decode a short prompt, print text.
//!
//! arg5 = mode: `gpu` uses the Vulkan fused path, `spec` additionally drafts
//! with prompt lookup (byte-identical to `gpu`, see `bench_draft`), anything
//! else is the CPU reference path.
//! arg6 = `max_ctx`, or the draft policy when it is not a number.
//! arg7 = the draft policy, `k:min_match[:guard]`, defaulting to the shipped one.
use kore_nn::qwen35::{Qwen35Gpu, Qwen35Model};
use kore_nn::spec::DraftPolicy;
use std::path::Path;
use std::time::Instant;

fn parse_policy(s: &str) -> DraftPolicy {
    let mut it = s.split(':');
    let k = it.next().and_then(|x| x.parse().ok()).unwrap_or(3);
    let min_match = it.next().and_then(|x| x.parse().ok()).unwrap_or(4);
    // Third field, if present, is the payoff guard: `0` or `off` disables it.
    let guard = match it.next() {
        Some(g) => g != "0" && g != "off",
        None => true,
    };
    DraftPolicy {
        k,
        min_match,
        guard,
        ..Default::default()
    }
    .clamped()
}

fn main() {
    let gguf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/bonsai2-2b/Ternary-Bonsai-2-2B-Q2_0.gguf".into());
    let tok_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "models/bonsai2-2b/tokenizer.json".into());
    let prompt = std::env::args().nth(3).unwrap_or_else(|| "The capital of France is".into());
    let n_new: usize = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(20);
    let mode = std::env::args().nth(5).unwrap_or_default();
    let use_gpu = mode == "gpu" || mode == "spec";

    let t0 = Instant::now();
    let model = Qwen35Model::from_gguf(Path::new(&gguf)).expect("load model");
    eprintln!("model loaded in {:?}: {:?}", t0.elapsed(), model.cfg);

    let tok = tokenizers::Tokenizer::from_file(&tok_path).expect("tokenizer");
    let ids = tok
        .encode(prompt.as_str(), true)
        .expect("encode")
        .get_ids()
        .to_vec();
    eprintln!("prompt tokens: {ids:?}");
    let max_ctx_arg = std::env::args().nth(6);
    let max_ctx: usize = max_ctx_arg
        .as_deref()
        .and_then(|s| s.parse().ok())
        .unwrap_or(ids.len() + n_new + 8);
    // arg6 doubles as the policy when it is not a number, so `... <prompt> <n>
    // spec 2:8` works without having to spell out `max_ctx` first.
    let policy_arg = if max_ctx_arg.as_deref().and_then(|s| s.parse::<usize>().ok()).is_some() {
        std::env::args().nth(7)
    } else {
        max_ctx_arg
    };

    if use_gpu {
        kore_vulkan::register_vulkan_backend().expect("vulkan backend");
        let t_gpu = Instant::now();
        let gpu = Qwen35Gpu::from_cpu(&model, kore_core::Device::Vulkan(0)).expect("gpu materialize");
        eprintln!("weights uploaded in {:?}", t_gpu.elapsed());

        let t1 = Instant::now();
        let out = if mode == "spec" {
            let policy = match policy_arg {
                Some(s) => parse_policy(&s),
                None => DraftPolicy::default(),
            };
            let (out, stats) = gpu
                .generate_spec(&model, &ids, n_new, max_ctx, policy)
                .expect("gpu spec generate");
            eprintln!(
                "spec: k={} min_match={} | {}/{} steps fired ({:.1}%), drafts accepted {}/{} ({:.1}%), P(j)={}/{}/{}/{}",
                policy.k,
                policy.min_match,
                stats.fired,
                stats.fired + stats.plain,
                100.0 * stats.fire_rate(),
                stats.accepted,
                stats.drafts,
                100.0 * stats.acceptance(),
                stats.hist[0],
                stats.hist[1],
                stats.hist[2],
                stats.hist[3]
            );
            out
        } else {
            gpu.generate_greedy(&model, &ids, n_new, max_ctx).expect("gpu generate")
        };
        let dt = t1.elapsed();
        eprintln!("GPU generated {} tokens in {:?} ({:.2} tok/s)", out.len(), dt, out.len() as f64 / dt.as_secs_f64());
        eprintln!("token ids: {out:?}");
        println!("{}", tok.decode(&out, true).expect("decode"));
        return;
    }

    // top-10 probe after prompt (CPU path)
    {
        let (mut ds, mut as_) = model.new_state();
        let mut logits = Vec::new();
        for (i, &t) in ids.iter().enumerate() {
            logits = model.forward(t, i, &mut ds, &mut as_).expect("fwd");
        }
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        eprintln!("top-10 after prompt:");
        for &i in idx.iter().take(10) {
            let s = tok.decode(&[i as u32], false).unwrap_or_else(|_| "?".into());
            eprintln!("  {:>7} {:>9.3} {:?}", i, logits[i], s);
        }
    }

    let t1 = Instant::now();
    let out = model.generate_greedy(&ids, n_new, model.cfg.eos).expect("generate");
    let dt = t1.elapsed();
    eprintln!("generated {} tokens in {:?} ({:.2} tok/s)", out.len(), dt, out.len() as f64 / dt.as_secs_f64());
    eprintln!("token ids: {out:?}");
    let text = tok.decode(&out, true).expect("decode");
    println!("{text}");
}
