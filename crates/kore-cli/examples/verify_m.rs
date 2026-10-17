//! Multi-column forward equivalence check.
//!
//! `forward_batch` over a chunk of `m` tokens must produce exactly the tokens a
//! sequence of `m` single-token `forward` steps would, and leave the recurrent
//! state in exactly the same place. That is the property a spec-decode verify
//! pass depends on: it can accept a draft on the batched argmax only if the
//! batched argmax *is* the sequential one.
//!
//! The check drives both paths over the same token stream:
//!
//!   1. sequential: feed every token one at a time, recording the token each
//!      position predicts;
//!   2. batched: replay the same stream in chunks of `m` and require the chunk
//!      results to match the recorded per-position predictions, element for
//!      element;
//!   3. state probe: one extra single-token step from each path at the same
//!      position must agree, so matching outputs cannot come from two states that
//!      happen to be "differently wrong".
//!
//! Usage: `verify_m <gguf> <tokenizer> <prompt> <n_new> <m>`
//! `m = 1` checks `forward_batch` against `forward` directly.

use kore_nn::qwen35::{Qwen35Gpu, Qwen35Model};
use std::path::Path;

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
    let n_new: usize = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let m: usize = std::env::args()
        .nth(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    assert!(m >= 1, "m must be >= 1");

    let model = Qwen35Model::from_gguf(Path::new(&gguf)).expect("load model");
    let tok = tokenizers::Tokenizer::from_file(&tok_path).expect("tokenizer");
    let ids = tok
        .encode(prompt.as_str(), true)
        .expect("encode")
        .get_ids()
        .to_vec();
    let plen = ids.len();
    let max_ctx = plen + n_new + 8 + m;

    kore_vulkan::register_vulkan_backend().expect("vulkan backend");
    let gpu = Qwen35Gpu::from_cpu(&model, kore_core::Device::Vulkan(0)).expect("gpu materialize");

    // ---- 1. sequential reference, recording what each position predicts ----
    // `ref_out[pos]` is the token predicted after consuming position `pos`, so
    // `ref_out.len() == stream.len()`.
    let mut st_ref = gpu.new_state(max_ctx).expect("state");
    let mut ref_out = Vec::with_capacity(plen + n_new);
    let mut next = 0u32;
    for (i, &t) in ids.iter().enumerate() {
        next = gpu.forward(&model, t, i, &mut st_ref).expect("fwd");
        ref_out.push(next);
    }
    // Continue greedily, pushing each predicted token back in as the next input.
    for k in 0..n_new {
        next = gpu.forward(&model, next, plen + k, &mut st_ref).expect("fwd");
        ref_out.push(next);
    }
    // The stream both paths consume: prompt followed by the greedy continuation.
    let mut stream = ids.clone();
    stream.extend_from_slice(&ref_out[plen - 1..plen - 1 + n_new]);
    let l = stream.len();
    assert_eq!(ref_out.len(), l);

    // ---- 2. batched path over the same stream ----
    let mut st_b = gpu.new_state(max_ctx).expect("state");
    let mut mismatch: Option<(usize, u32, u32)> = None;
    let mut p = 0usize;
    while p < l {
        let chunk = m.min(l - p);
        let outs = gpu
            .forward_batch(&model, &stream[p..p + chunk], p, &mut st_b)
            .expect("forward_batch");
        assert_eq!(outs.len(), chunk, "chunk@{p}: wrong output length");
        for r in 0..chunk {
            if outs[r] != ref_out[p + r] {
                mismatch = Some((p + r, outs[r], ref_out[p + r]));
            }
        }
        p += chunk;
    }

    // ---- 3. state probe: one more step from each, same token, same position ----
    let probe = 42u32;
    let a = gpu
        .forward(&model, probe, l, &mut st_ref)
        .expect("probe sequential");
    let b = gpu
        .forward_batch(&model, &[probe], l, &mut st_b)
        .expect("probe batched")[0];

    if let Some((pos, got, want)) = mismatch {
        eprintln!("FAIL m={m}: position {pos}: batched argmax {got} != sequential {want}");
        std::process::exit(1);
    }
    if a != b {
        eprintln!("FAIL m={m}: state probe diverged: sequential {a} vs batched {b}");
        std::process::exit(1);
    }

    println!(
        "OK m={m}: {l} positions bit-identical to sequential decode, state probe agrees ({a})"
    );
}
