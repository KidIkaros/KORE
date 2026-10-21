//! How much would prompt-lookup drafting have bought on a token stream that
//! already exists?
//!
//! The batch path for the MoE is multi-hour work, so this answers the only
//! question that can cancel it cheaply: does the model's own output contain
//! the repetition a lookup drafter feeds on? It replays an existing greedy
//! stream through the *real* [`DraftPolicy`] and counts what would have been
//! accepted, then projects throughput under a cost model for the `m`-column
//! pass.
//!
//! Usage: `spec_replay <ids-file-or-list> <k:min_match[:max_match]> [c1_ms] [alpha]`
//!
//! `ids` is either a path to a file containing token ids or the list itself;
//! it must be the prompt followed by the generated tokens (the `bonsai`
//! harness prints both). `c1_ms` is a plain step, `alpha` the marginal cost of
//! one more column as a fraction of a plain step: `C(m) = c1 * (1 + alpha*(m-1))`.
//! The projection is reported per alpha, because the real `C(m)` is exactly
//! what the batch path has not measured yet.

use kore_nn::spec::DraftPolicy;

fn arg<T: std::str::FromStr>(i: usize, default: T) -> T {
    std::env::args()
        .nth(i)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn load_ids(spec: &str) -> Vec<u32> {
    let text = match std::fs::read_to_string(spec) {
        Ok(t) => t,
        Err(_) => spec.to_string(),
    };
    text.split(|c: char| c == ',' || c.is_whitespace() || c == '[' || c == ']')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("token id"))
        .collect()
}

fn parse_policy(s: &str) -> DraftPolicy {
    let mut it = s.split(':');
    let k = it.next().and_then(|x| x.parse().ok()).unwrap_or(6);
    let min_match = it.next().and_then(|x| x.parse().ok()).unwrap_or(6);
    let max_match = it.next().and_then(|x| x.parse().ok()).unwrap_or(16);
    DraftPolicy {
        k,
        min_match,
        max_match,
        ..Default::default()
    }
    .clamped()
}

fn main() {
    let ids_arg = std::env::args().nth(1).expect("ids file or list");
    let ids = load_ids(&ids_arg);
    let policy = parse_policy(&std::env::args().nth(2).unwrap_or_else(|| "6:6".into()));
    let c1_ms: f64 = arg(3, 16.1);
    // Prompt length: pass it as the fifth argument (the stream file has no
    // marker). Defaults to 0, i.e. replay the whole stream.
    let prompt_len: usize = arg(4, 0);
    if ids.len() < 2 {
        panic!("need at least two tokens");
    }

    // Replay: `hist` is everything before the current position, `next` is the
    // token the model chose there (the stream itself). Drafts are checked
    // against what actually follows.
    let mut fired = 0usize;
    let mut plain = 0usize;
    let mut drafts_total = 0usize;
    let mut accepted_total = 0usize;
    let mut hist_acc = [0usize; 8];
    let mut slot_prop = [0usize; 8];
    let mut slot_acc = [0usize; 8];
    // Per-fire records for the projections: (m, j, cut).
    let mut fires: Vec<(usize, usize, bool)> = Vec::new();

    let mut i = prompt_len;
    while i + 1 < ids.len() {
        let drafts = policy.drafts(&ids[..i], ids[i]);
        if drafts.is_empty() {
            plain += 1;
            i += 1;
            continue;
        }
        let mut j = 0usize;
        while j < drafts.len() && i + 1 + j < ids.len() && drafts[j] == ids[i + 1 + j] {
            j += 1;
        }
        fired += 1;
        drafts_total += drafts.len();
        accepted_total += j;
        hist_acc[j.min(7)] += 1;
        for (s, _) in drafts.iter().enumerate().take(8) {
            slot_prop[s] += 1;
            if s < j {
                slot_acc[s] += 1;
            }
        }
        fires.push((drafts.len(), j, j < drafts.len()));
        i += 1 + j;
    }

    let generated = ids.len().saturating_sub(prompt_len);
    let steps = fired + plain;
    println!(
        "stream: {} ids (prompt {prompt_len}, generated {generated}), policy k={} min={} max={}",
        ids.len(),
        policy.k,
        policy.min_match,
        policy.max_match
    );
    println!(
        "fires {fired}/{steps} ({:.1}%), drafts {drafts_total}, accepted {accepted_total} ({:.1}%)",
        100.0 * fired as f64 / steps.max(1) as f64,
        100.0 * accepted_total as f64 / drafts_total.max(1) as f64
    );
    print!("accepted-prefix hist:");
    for (j, n) in hist_acc.iter().enumerate() {
        if *n > 0 {
            print!(" j={j}:{n}");
        }
    }
    println!("\nslot acceptance:");
    for s in 0..8 {
        if slot_prop[s] > 0 {
            println!(
                "  slot {}: {:5.1}% ({}/{})",
                s + 1,
                100.0 * slot_acc[s] as f64 / slot_prop[s] as f64,
                slot_acc[s],
                slot_prop[s]
            );
        }
    }

    // Projections. A fire costs `C(m) + snapshot`; a cut adds the rewind. The
    // snapshot/rewind numbers are the dense ones (1.15 ms/fire, 0.2 ms/cut);
    // they are small next to the step and are held fixed across alphas.
    const SNAPSHOT_MS: f64 = 1.15;
    const REWIND_MS: f64 = 0.20;
    println!(
        "\nprojection (c1 = {c1_ms:.2} ms, snapshot {SNAPSHOT_MS:.2}, rewind {REWIND_MS:.2}):"
    );
    println!(
        "  {:>5} {:>12} {:>12} {:>9} {:>9}",
        "alpha", "unguarded", "oracle", "plain", "ratio"
    );
    for alpha in [0.2f64, 0.3, 0.4, 0.5] {
        let c = |m: usize| c1_ms * (1.0 + alpha * (m as f64 - 1.0));
        // Unguarded: every match fires.
        let mut t_ung = plain as f64 * c1_ms;
        let mut t_ora = plain as f64 * c1_ms;
        for &(k, j, cut) in &fires {
            let m = 1 + k;
            let cost = c(m) + SNAPSHOT_MS + if cut { REWIND_MS } else { 0.0 };
            t_ung += cost;
            // Oracle scheduling: fire only when this fire beats plain decoding
            // of the same tokens (1 + j tokens at c1 each). An upper bound on
            // any real payoff guard.
            if (1 + j) as f64 * c1_ms > cost {
                t_ora += cost;
            } else {
                t_ora += (1 + j) as f64 * c1_ms;
            }
        }
        // Plain greedy decodes every generated token at `c1`.
        let plain_tps = 1e3 / c1_ms;
        let ung = generated as f64 / t_ung * 1e3;
        println!(
            "  {:>5.2} {:>10.1} t/s {:>10.1} t/s {:>8.1} t/s {:>8.2}x",
            alpha,
            ung,
            generated as f64 / t_ora * 1e3,
            plain_tps,
            ung / plain_tps
        );
    }
}
