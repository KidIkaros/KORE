//! Flint (40 B-A2.5 B MoE) smoke test on the host path: load the GGUF and the
//! HF tokenizer, greedily decode a prompt, print the text and the rates.
//!
//! `flint <gguf> <tokenizer.json> <prompt> <n_new> [max_ctx]`
//!
//! `KORE_CHAT=1` wraps the prompt in the model's `<|user|> ... <|assistant|>`
//! turn markers (the GGUF's chat template) instead of feeding it bare.
//! `KORE_TOKENS=1` prints the prompt ids, so a run can be reproduced without
//! the tokenizer on a box that does not have it. `KORE_LOGITS=1` prints the
//! top of the distribution at the first generated position, which is what tells
//! a flat, broken forward pass apart from a model that simply disagrees.
use kore_nn::flint::FlintModel;
use std::path::Path;
use std::time::Instant;

fn main() {
    let gguf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/flint/flint-Q4_K_M.gguf".into());
    let tok_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "models/flint/tokenizer.json".into());
    let prompt = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "The capital of France is".into());
    let n_new: usize = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let max_ctx_arg: usize = std::env::args()
        .nth(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let t0 = Instant::now();
    let model = FlintModel::from_gguf(Path::new(&gguf)).expect("load model");
    eprintln!("model loaded in {:?}", t0.elapsed());
    eprintln!(
        "config: hidden {} layers {} ({} dense) heads {}/{} hd {} ffn {} moe {} shexp {} \
         experts {}/{} window {} rope_off {} sliding {}",
        model.cfg.hidden,
        model.cfg.n_layers,
        model.cfg.n_dense,
        model.cfg.n_heads,
        model.cfg.n_kv,
        model.cfg.head_dim,
        model.cfg.ffn,
        model.cfg.moe_ffn,
        model.cfg.shexp_ffn,
        model.cfg.n_experts_used,
        model.cfg.n_experts,
        model.cfg.sliding_window,
        (0..model.cfg.n_layers).filter(|&i| !model.cfg.rope.get(i).copied().unwrap_or(true)).count(),
        (0..model.cfg.n_layers).filter(|&i| model.cfg.sliding.get(i).copied().unwrap_or(false)).count(),
    );

    if std::env::var("KORE_BENCH").ok().as_deref() == Some("1") {
        let reps: usize = std::env::var("KORE_BENCH_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(40);
        model.bench(reps);
        return;
    }
    let tok = tokenizers::Tokenizer::from_file(&tok_path).expect("tokenizer");
    let text = if std::env::var("KORE_CHAT").ok().as_deref() == Some("1") {
        format!("<|user|>{prompt}<|assistant|>")
    } else {
        prompt.clone()
    };
    let ids = tok
        .encode(text.as_str(), true)
        .expect("encode")
        .get_ids()
        .to_vec();
    let ctx = if max_ctx_arg > 1 {
        max_ctx_arg
    } else {
        ids.len() + n_new + 8
    };
    if std::env::var("KORE_TOKENS").ok().as_deref() == Some("1") {
        eprintln!("prompt tokens: {ids:?}");
    }
    eprintln!("prompt: {} tokens, max_ctx {ctx}", ids.len());

    if let Ok(path) = std::env::var("KORE_ROUTE_PROFILE") {
        let ctx = ids.len() + n_new + 8;
        let prof = model.profile_routes(&ids, n_new, ctx).expect("profile routes");
        prof.summary();
        if !path.is_empty() {
            prof.save(Path::new(&path)).expect("save profile");
            eprintln!("wrote {path}");
        }
        return;
    }
    if let Ok(path) = std::env::var("KORE_ROUTE_TEST") {
        // Held-out check: the hot set comes from a saved profile (built on some
        // other prompt), and coverage is measured on this prompt's routes.
        let ctx = ids.len() + n_new + 8;
        let train = kore_nn::flint::RouteProfile::load(Path::new(&path)).expect("load profile");
        let test = model.profile_routes(&ids, n_new, ctx).expect("profile routes");
        eprintln!(
            "=== held-out hot set: {path} ({} tokens) applied to this prompt ({} tokens) ===",
            train.tokens, test.tokens
        );
        if test.expert_bytes == 0 || test.moe_layers == 0 {
            eprintln!("no expert_bytes/moe_layers in the test profile");
            return;
        }
        eprintln!("pin       experts   of bank   same-prompt   held-out");
        for gib in [2u64, 4, 6, 8, 10, 12, 14] {
            let n = (gib << 30) as usize / test.expert_bytes as usize;
            if n == 0 || n >= test.counts.len() {
                continue;
            }
            let hot = train.hot_set_n(n);
            eprintln!(
                "  {gib:2} GiB  {n:7}  {:5.1}%   {:9.1}%   {:7.1}%",
                100.0 * n as f64 / test.counts.len() as f64,
                train.coverage(&hot),
                test.coverage(&hot)
            );
        }
        return;
    }

    // Prefill and decode are timed apart: the first token carries the whole
    // prompt, so folding it into the decode rate would understate the steady
    // state on a long prompt.
    let mut st = model.new_state(ctx);
    let t_pre = Instant::now();
    let mut logits = Vec::new();
    for (i, &id) in ids.iter().enumerate() {
        logits = model.forward(id, i, &mut st).expect("forward");
    }
    let pre = t_pre.elapsed().as_secs_f64();
    let mut next = FlintModel::argmax(&logits);

    if std::env::var("KORE_LOGITS").ok().as_deref() == Some("1") {
        let mut top: Vec<(u32, f32)> = logits.iter().map(|&v| (0u32, v)).collect();
        for (i, e) in top.iter_mut().enumerate() {
            e.0 = i as u32;
        }
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let head: Vec<String> = top
            .iter()
            .take(5)
            .map(|(i, v)| format!("#{i} {v:.3} '{}'", tok.decode(&[*i], false).unwrap_or_default()))
            .collect();
        eprintln!("top-5: {}", head.join(" | "));
    }

    let t_dec = Instant::now();
    FlintModel::profile_reset();
    let mut out = Vec::with_capacity(n_new);
    for k in 0..n_new {
        out.push(next);
        if model.cfg.eos.contains(&next) || k + 1 == n_new {
            break;
        }
        logits = model
            .forward(next, ids.len() + k, &mut st)
            .expect("forward");
        next = FlintModel::argmax(&logits);
    }
    let dec = t_dec.elapsed().as_secs_f64();

    let text = tok.decode(&out, true).unwrap_or_else(|_| format!("{out:?}"));
    println!("{text}");
    // One token came out of the prompt's last forward, so the steady-state rate
    // is (emitted - 1) over the decode window.
    let steps = out.len().saturating_sub(1);
    FlintModel::profile_dump(steps);
    if std::env::var("KORE_IDS").ok().as_deref() == Some("1") {
        println!("ids: {out:?}");
    }
    eprintln!(
        "prefill {} tokens in {:.2} s = {:.2} tok/s | decode {} steps in {:.2} s = {:.2} tok/s",
        ids.len(),
        pre,
        ids.len() as f64 / pre.max(1e-9),
        steps,
        dec,
        steps as f64 / dec.max(1e-9),
    );
}
