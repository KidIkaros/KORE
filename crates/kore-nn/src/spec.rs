//! Prompt-lookup (n-gram) speculative decoding.
//!
//! A verify pass costs one batched forward and returns one token per column, so
//! the question is only whether the tokens fed to it are the ones the model
//! would itself have chosen. Get that right and `m` positions cost one pass
//! instead of `m`.
//!
//! The cheapest drafter that can answer that needs no weights and no second
//! model: the text itself. Look for the current context earlier in the text and
//! propose whatever followed it there. On a prompt that gets echoed -- a
//! rewrite, a summary, an extraction, a code edit, or a model that has fallen
//! into a repetition -- the continuation is often exactly right, and a 4-gram of
//! context turns out to be a strong enough signal that ~94% of the drafts it
//! produces are accepted (see `draft_headroom` for the measurement).
//!
//! Two properties make this safe to ship:
//!
//! * **Greedy acceptance is exact.** A draft is accepted iff it equals the
//!   argmax the batched pass produced, and a verify pass is bit-identical to
//!   sequential decoding column by column ([`Qwen35Gpu::forward_verify`]).
//!   So a wrong draft can only waste a pass; it can never change the output.
//! * **The trigger is the thing that decides whether it pays.** A fired step at
//!   `m = 4` costs ~22 ms against ~9 ms for a plain step, so it needs the
//!   *average* accepted prefix to clear ~1.7 tokens. Measuring the acceptance
//!   distribution (`draft_headroom`) is what picks `min_match`; firing on any
//!   match at all is *worse* than not drafting.

use std::collections::VecDeque;
use std::time::Instant;

use kore_core::{KoreError, Result};

use crate::qwen35::{Qwen35Gpu, Qwen35Model, MAX_BATCH_COLS};

#[allow(dead_code)]
fn err(msg: &str) -> KoreError {
    KoreError::StorageError(format!("spec: {msg}"))
}

/// How hard to look for a match, and how strong it has to be before a step
/// spends a batched pass on it.
#[derive(Clone, Copy, Debug)]
pub struct DraftPolicy {
    /// Draft tokens a fired step proposes. A fired step is one
    /// `m = 1 + k` pass, so this is capped by the GEMV's column width.
    pub k: usize,
    /// Shortest match worth acting on. The context searched for is
    /// `min_match` tokens long, so a hit *is* a match of at least this length.
    pub min_match: usize,
    /// Longest match sought. Beyond `min_match` a longer match is a stronger
    /// signal, but it is also rarer; the two are one knob in the measurement.
    pub max_match: usize,
    /// How far back to search, in tokens. Bounds the per-step cost of the scan
    /// on a very long context; past this the odds of a still-valid continuation
    /// are not worth the microseconds.
    pub search_back: usize,
    /// Whether to keep checking that drafting is paying for itself (see
    /// [`PayoffGuard`]). On by default: it cannot change the output, only which
    /// steps are batched, and it is the thing that keeps a policy tuned for one
    /// device from being a net loss on another.
    pub guard: bool,
}

impl Default for DraftPolicy {
    /// `k = 6`, `min_match = 6`: the best measured operating point on this box.
    ///
    /// With the GEMV widened to eight columns, the useful draft depth is set by
    /// how far a match usually runs, not by the kernel. Measured on Bonsai-2B
    /// (192 tokens, interleaved rounds): on repetitive text `6:6` reads 1.57x
    /// greedy against 1.54x for the old `3:4`, and on prose 1.37x against 1.35x.
    /// Lowering `min_match` to 1 buys more on repetition (1.62x) but loses on
    /// prose (1.18x), so the conservative trigger is the shipped default. `k`
    /// is capped below.
    fn default() -> Self {
        Self {
            k: 6,
            min_match: 6,
            max_match: 16,
            search_back: 16384,
            guard: true,
        }
    }
}

impl DraftPolicy {
    /// Clamp `k` to what one pass can carry and `min_match`/`max_match` into a
    /// sane order, so a caller-supplied policy cannot ask for an illegal batch.
    pub fn clamped(self) -> Self {
        let k = self.k.min(MAX_BATCH_COLS - 1);
        let min_match = self.min_match.max(1).min(self.max_match.max(1));
        Self {
            k,
            min_match,
            max_match: self.max_match.max(min_match),
            search_back: self.search_back.max(1),
            guard: self.guard,
        }
    }

    /// What to feed the verify pass, or an empty vector to take a plain step.
    pub fn drafts(&self, hist: &[u32], t_next: u32) -> Vec<u32> {
        let (len, drafts) = longest_match(hist, t_next, self.k, self.max_match, self.search_back);
        if len >= self.min_match {
            drafts
        } else {
            Vec::new()
        }
    }
}

/// Longest suffix of `hist ++ [t_next]`, of length at most `max_match`, that
/// also occurs earlier in `hist`. Returns `(length, continuation)`: the
/// continuation is what followed that earlier occurrence, capped at `k` tokens,
/// and is empty when that occurrence had nothing after it (so there is nothing
/// to draft).
///
/// Matching *including* `t_next` is what aligns the continuation with the batch.
/// The model's own next token is already known to be right, so the useful
/// guesses start one position later -- and a context that *ends* with `t_next`
/// is one whose continuation is a guess for exactly that position.
///
/// The scan is a single reverse pass over candidate match *ends*, with a
/// one-comparison fast path for the (overwhelmingly common) end that is not
/// `t_next`; only the ends that are pay the backwards extension.
pub fn longest_match(
    hist: &[u32],
    t_next: u32,
    k: usize,
    max_match: usize,
    search_back: usize,
) -> (usize, Vec<u32>) {
    let p = hist.len();
    if p == 0 || k == 0 || max_match == 0 {
        return (0, Vec::new());
    }
    let lo = p.saturating_sub(search_back);
    let mut best_len = 0usize;
    let mut best_end = lo;
    for e in (lo..p).rev() {
        if hist[e] != t_next {
            continue;
        }
        // The match covers `[e - len + 1, e]` and lines up with the context's
        // tail `[p - len + 1, p]`, where the last context token is `t_next`.
        let mut len = 1usize;
        while len < max_match && len <= e && len <= p && hist[e - len] == hist[p - len] {
            len += 1;
        }
        if len > best_len {
            best_len = len;
            best_end = e;
        }
    }
    if best_len == 0 {
        return (0, Vec::new());
    }
    let start = best_end + 1;
    let end = (start + k).min(p);
    if end <= start {
        return (best_len, Vec::new());
    }
    (best_len, hist[start..end].to_vec())
}

/// Counters from a speculative run.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpecStats {
    /// Steps that ran a batched verify pass.
    pub fired: usize,
    /// Steps that ran a plain single-token pass.
    pub plain: usize,
    /// Draft tokens fed to verify passes.
    pub drafts: usize,
    /// Draft tokens the model agreed with.
    pub accepted: usize,
    /// Accepted-prefix histogram over fired steps: `hist[j]` counts the steps
    /// that accepted exactly `j` of their drafts.
    pub hist: [usize; MAX_BATCH_COLS],
}

impl SpecStats {
    /// Accepted drafts as a share of proposed ones.
    pub fn acceptance(&self) -> f64 {
        if self.drafts == 0 {
            0.0
        } else {
            self.accepted as f64 / self.drafts as f64
        }
    }

    /// Share of steps that spent a batched pass.
    pub fn fire_rate(&self) -> f64 {
        let steps = self.fired + self.plain;
        if steps == 0 {
            0.0
        } else {
            self.fired as f64 / steps as f64
        }
    }
}

/// Keeps a spec run honest about whether drafting is still paying for itself.
///
/// A fired step over `m` columns costs `C(m)`, a plain step `C(1)`, and a fired
/// step emits `1 + j` tokens against a plain step's 1. So a fire is worth taking
/// iff `1 + j̄ > C(m)/C(1)` — on this box about **1.33 accepted drafts at
/// `m = 4`**, which the measured trigger clears with room to spare (94–100% of
/// drafts accepted, against the ~75% rejection rate that would break even).
///
/// But `C(m)/C(1)` is a property of the *device*, not of the trigger. A backend
/// whose batched pass is relatively more expensive — or a device where it is
/// relatively cheaper, and the trigger is therefore too timid — can turn the
/// same policy into the wrong answer. So the guard does not assume the ratio.
/// It keeps the passes' own **tokens per millisecond** and pauses drafting when
/// a fired step stops beating a plain one.
///
/// That comparison is the objective itself, and it is what makes the guard safe
/// to leave on: it is not a proxy for whether drafting pays, it *is* whether
/// drafting pays, measured on the machine currently running rather than on a
/// cost model tuned for this laptop.
///
/// Two more properties matter:
///
/// * **It cannot change the output.** It only chooses which steps run batched,
///   and a batched step that a draft does not survive is rewound; the token
///   stream is the plain-greedy one either way. `bench_draft`'s identity gate
///   therefore still applies verbatim.
/// * **It cannot lock out.** A paused guard re-probes every
///   [`GUARD_REPROBE`] plain steps, so a workload that turns repetitive halfway
///   through is picked up again. A guard that never re-probed would be the more
///   dangerous design — it would silently keep paying the tax on a run that had
///   started to win.
struct PayoffGuard {
    /// `(tokens, ms)` for the most recent plain steps: the bar to clear.
    plain: VecDeque<(f64, f64)>,
    /// `(tokens, ms)` for the most recent fires, including the rewind a partial
    /// acceptance costs — that copy is part of what a fire actually spends.
    fires: VecDeque<(f64, f64)>,
    /// Plain steps taken since the last fire, so a paused guard re-probes.
    since_fire: usize,
}

/// Plain steps remembered. Long enough to average out the noise in a
/// millisecond-scale measurement, short enough to track a growing context.
const GUARD_PLAIN_WINDOW: usize = 16;
/// Fires remembered. Shorter than the plain window because fires are rarer and
/// the thing being detected is a persistent bias, not a single bad draw.
const GUARD_FIRE_WINDOW: usize = 6;
/// How many fires to observe before the guard is willing to have an opinion.
const GUARD_MIN_FIRES: usize = 3;
/// Plain steps a paused guard waits before spending one fire to re-check.
const GUARD_REPROBE: usize = 16;
/// A fire must beat a plain step by this much. Below 1.0 so that measurement
/// noise cannot pause drafting over a difference that is not real.
const GUARD_TOLERANCE: f64 = 0.98;

impl PayoffGuard {
    fn new() -> Self {
        Self {
            plain: VecDeque::with_capacity(GUARD_PLAIN_WINDOW),
            fires: VecDeque::with_capacity(GUARD_FIRE_WINDOW),
            since_fire: 0,
        }
    }

    fn record_plain(&mut self, emitted: usize, cost_ms: f64) {
        push_windowed(&mut self.plain, GUARD_PLAIN_WINDOW, emitted, cost_ms);
        self.since_fire += 1;
    }

    fn record_fire(&mut self, emitted: usize, cost_ms: f64) {
        push_windowed(&mut self.fires, GUARD_FIRE_WINDOW, emitted, cost_ms);
        self.since_fire = 0;
    }

    /// Whether the next step is allowed to draft. Optimistic until there is
    /// enough evidence either way, and it re-probes rather than locking out, so
    /// the answer is never "no forever".
    fn may_fire(&self) -> bool {
        if self.fires.len() < GUARD_MIN_FIRES || self.plain.is_empty() {
            return true;
        }
        if self.since_fire >= GUARD_REPROBE {
            return true;
        }
        let fire_tokens: f64 = self.fires.iter().map(|(e, _)| e).sum();
        let fire_ms: f64 = self.fires.iter().map(|(_, c)| c).sum();
        let plain_tokens: f64 = self.plain.iter().map(|(e, _)| e).sum();
        let plain_ms: f64 = self.plain.iter().map(|(_, c)| c).sum();
        // Same units on both sides -- tokens per millisecond -- so this is a
        // statement about throughput and not about cost ratios. Cross-multiplied
        // to keep it to integers and one division.
        fire_tokens * plain_ms >= GUARD_TOLERANCE * plain_tokens * fire_ms
    }
}

fn push_windowed(w: &mut VecDeque<(f64, f64)>, cap: usize, emitted: usize, cost_ms: f64) {
    if w.len() == cap {
        w.pop_front();
    }
    w.push_back((emitted as f64, cost_ms));
}

impl Qwen35Gpu {
    /// Greedy decode with prompt-lookup drafting.
    ///
    /// Byte-identical to [`Qwen35Gpu::generate_greedy`] by construction: any
    /// column the batched argmax did not reproduce is rewound before the next
    /// step, so the tokens returned are exactly the ones plain greedy decoding
    /// would have returned. The only thing a wrong draft can cost is a wasted
    /// pass.
    ///
    /// Returns the tokens and the [`SpecStats`] that say what the drafting
    /// actually bought on this prompt.
    pub fn generate_spec(
        &self,
        cpu: &Qwen35Model,
        tokens: &[u32],
        n_new: usize,
        max_ctx: usize,
        policy: DraftPolicy,
    ) -> Result<(Vec<u32>, SpecStats)> {
        let policy = policy.clamped();
        let eos = cpu.cfg.eos;
        let mut states = self.new_state(max_ctx)?;
        let mut arena = self.new_rewind()?;
        let mut stats = SpecStats::default();
        let mut guard = PayoffGuard::new();
        let mut out: Vec<u32> = Vec::with_capacity(n_new);
        let mut hist: Vec<u32> = tokens.to_vec();
        let mut pos = tokens.len();
        // Split the fired-step cost into the pass and the truncation copy, so a
        // profiling run can say which one the loop is paying for.
        let profile = std::env::var_os("KORE_SPEC_PROFILE").is_some();
        let mut verify_ms = 0.0f64;
        let mut rewind_ms = 0.0f64;
        let mut rewinds = 0usize;
        let mut next = if tokens.is_empty() {
            0
        } else {
            self.prefill(cpu, tokens, 0, &mut states)?
        };

        while out.len() < n_new {
            if next == eos {
                break;
            }
            let drafts = if policy.guard && !guard.may_fire() {
                Vec::new()
            } else {
                policy.drafts(&hist, next)
            };
            let m = 1 + drafts.len();
            if drafts.is_empty() || pos + m > max_ctx {
                out.push(next);
                hist.push(next);
                let t = Instant::now();
                next = self.forward(cpu, next, pos, &mut states)?;
                guard.record_plain(1, t.elapsed().as_secs_f64() * 1e3);
                pos += 1;
                stats.plain += 1;
                continue;
            }

            // Column 0 is the token the model already chose; the drafts follow
            // it. `outs[r]` is the prediction after consuming the batch up to
            // column `r`, i.e. the prediction for position `pos + r + 1`.
            let mut batch = Vec::with_capacity(m);
            batch.push(next);
            batch.extend_from_slice(&drafts);
            let t = Instant::now();
            let outs = self.forward_verify(cpu, &batch, pos, &mut states, &mut arena)?;
            if profile {
                verify_ms += t.elapsed().as_secs_f64() * 1e3;
            }

            // Longest prefix of drafts the model reproduced.
            let j = drafts
                .iter()
                .zip(outs.iter())
                .take_while(|(d, o)| *d == *o)
                .count();
            if j < drafts.len() {
                let tr = Instant::now();
                self.rewind(&mut states, &arena, 1 + j)?;
                if profile {
                    rewind_ms += tr.elapsed().as_secs_f64() * 1e3;
                    rewinds += 1;
                }
            }
            // Timed through the rewind: a fire that gets cut short really does
            // spend that copy, and the guard must see it.
            let fire_ms = t.elapsed().as_secs_f64() * 1e3;
            stats.fired += 1;
            stats.drafts += drafts.len();
            stats.accepted += j;
            stats.hist[j] += 1;

            let committed = &batch[..1 + j];
            match committed.iter().position(|&t| t == eos) {
                Some(stop) => {
                    guard.record_fire(stop, fire_ms);
                    out.extend_from_slice(&committed[..stop]);
                    hist.extend_from_slice(&committed[..stop]);
                    break;
                }
                None => {
                    guard.record_fire(committed.len(), fire_ms);
                    out.extend_from_slice(committed);
                    hist.extend_from_slice(committed);
                }
            }
            next = outs[j];
            pos += 1 + j;
        }
        if profile {
            eprintln!(
                "spec profile: verify {:.1} ms over {} fires ({:.2} ms/fire), rewind {:.1} ms over {} cuts ({:.2} ms/cut)",
                verify_ms,
                stats.fired,
                verify_ms / stats.fired.max(1) as f64,
                rewind_ms,
                rewinds,
                rewind_ms / rewinds.max(1) as f64,
            );
        }
        out.truncate(n_new);
        Ok((out, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The context searched for ends at the *last* history token and includes
    /// `t_next`; the continuation is whatever followed the earlier occurrence.
    #[test]
    fn finds_the_earlier_occurrence_and_what_followed_it() {
        //         \____ the earlier occurrence ____/
        let hist = vec![1, 2, 3, 4, 5, 6, 1, 2, 3, 4];
        // The context is `1 2 3 4 5` (hist's tail plus `t_next`), which last
        // occurred at 0 -- so `6 1 2` is the draft.
        let (len, drafts) = longest_match(&hist, 5, 3, 16, usize::MAX);
        assert_eq!(len, 5);
        assert_eq!(drafts, vec![6, 1, 2]);
    }

    #[test]
    fn drafts_are_capped_at_k_but_the_match_length_is_not() {
        let hist = vec![1, 2, 3, 4, 5, 6, 1, 2, 3, 4];
        let (len, drafts) = longest_match(&hist, 5, 1, 16, usize::MAX);
        assert_eq!(len, 5, "the reported match length ignores k");
        assert_eq!(drafts, vec![6]);
    }

    /// A match counts even when nothing followed it, but then there is nothing
    /// to draft -- returning `len` (not 0) is what lets the trigger see how
    /// strong the signal was.
    #[test]
    fn a_match_at_the_very_end_has_nothing_to_draft() {
        let hist = vec![1, 2, 3, 4];
        let (len, drafts) = longest_match(&hist, 4, 3, 16, usize::MAX);
        assert_eq!(len, 1);
        assert!(drafts.is_empty());
    }

    #[test]
    fn no_earlier_occurrence_means_no_match() {
        let hist = vec![1, 2, 3];
        assert_eq!(longest_match(&hist, 9, 3, 16, usize::MAX), (0, Vec::new()));
        assert_eq!(longest_match(&[], 9, 3, 16, usize::MAX), (0, Vec::new()));
    }

    /// The longest match wins, and among equal lengths the most recent one --
    /// the closest continuation is the one most likely to still hold.
    #[test]
    fn longest_wins_ties_go_to_the_most_recent() {
        // The context is `9 8 7` (hist's tail plus `t_next`). It occurs twice
        // earlier: at 0, followed by `1 2 9`, and at 5, followed by `3 9 8`.
        // Both matches are length 3, so recency decides.
        let hist = vec![9, 8, 7, 1, 2, 9, 8, 7, 3, 9, 8];
        let (len, drafts) = longest_match(&hist, 7, 3, 16, usize::MAX);
        assert_eq!(len, 3);
        assert_eq!(drafts, vec![3, 9, 8], "the most recent occurrence wins");
    }

    /// A longer match beats a more recent shorter one: the signal is what
    /// matters, not the proximity.
    #[test]
    fn a_longer_earlier_match_beats_a_shorter_recent_one() {
        // The context is `9 8 7 6`. A 4-gram match sits at 0 (followed by
        // `5 1 2`); a more recent but only 2-gram match sits at 7.
        let hist = vec![9, 8, 7, 6, 5, 1, 2, 7, 6, 9, 8, 7];
        let (len, drafts) = longest_match(&hist, 6, 3, 16, usize::MAX);
        assert_eq!(len, 4, "the 4-gram wins over the nearer 2-gram");
        assert_eq!(drafts, vec![5, 1, 2]);
    }

    /// `search_back` bounds the scan; a match that only exists past the window
    /// is deliberately not found.
    #[test]
    fn search_back_bounds_how_far_the_scan_looks() {
        let hist = vec![1, 2, 3, 4, 5, 6, 1, 2, 3, 4];
        assert_eq!(longest_match(&hist, 5, 3, 16, 2), (0, Vec::new()));
        assert_eq!(longest_match(&hist, 5, 3, 16, 8).0, 5);
    }

    #[test]
    fn the_trigger_needs_a_match_at_least_min_match_long() {
        let hist = vec![1, 2, 3, 4, 5, 6, 1, 2, 3, 4];
        let fire = |min_match| {
            DraftPolicy {
                k: 3,
                min_match,
                ..Default::default()
            }
            .clamped()
            .drafts(&hist, 5)
        };
        assert_eq!(fire(5), vec![6, 1, 2], "a 5-gram match fires");
        assert_eq!(fire(6), Vec::<u32>::new(), "a 6-gram match does not exist");
    }

    #[test]
    fn the_default_policy_already_fits_the_widest_pass() {
        let p = DraftPolicy::default();
        let c = p.clamped();
        assert_eq!(c.k, p.k, "default k must fit the verify width");
        assert_eq!(c.min_match, p.min_match);
        assert!(p.k + 1 <= MAX_BATCH_COLS);
    }

    #[test]
    fn k_is_clamped_to_the_gemvs_column_width() {
        let p = DraftPolicy {
            k: 99,
            ..Default::default()
        }
        .clamped();
        assert_eq!(p.k, MAX_BATCH_COLS - 1);
    }

    /// The guard is optimistic until it has seen enough fires to have an
    /// opinion, and it never pauses on a run it has no baseline for.
    #[test]
    fn the_guard_lets_drafting_through_until_it_has_evidence() {
        let mut g = PayoffGuard::new();
        assert!(g.may_fire(), "nothing measured yet");
        g.record_plain(1, 10.0);
        assert!(g.may_fire(), "one plain step is not evidence about fires");
        g.record_fire(1, 23.3);
        g.record_fire(1, 23.3);
        assert!(g.may_fire(), "still under GUARD_MIN_FIRES");
    }

    /// The test is throughput, not a cost ratio: a plain step makes 1 token in
    /// 10 ms, a 3-draft fire that lands none makes 1 in 23.3 ms. 0.043 tokens/ms
    /// against 0.100 is not close.
    #[test]
    fn a_fire_that_is_slower_than_a_plain_step_pauses_drafting() {
        let mut g = PayoffGuard::new();
        for _ in 0..GUARD_PLAIN_WINDOW {
            g.record_plain(1, 10.0);
        }
        for _ in 0..GUARD_MIN_FIRES {
            g.record_fire(1, 23.3);
        }
        assert!(!g.may_fire(), "1 token per 23.3 ms loses to 1 per 10 ms");
    }

    /// The same cost, three drafts accepted: 4 tokens per 23.3 ms is 0.172
    /// tokens/ms, comfortably ahead of the plain step's 0.100.
    #[test]
    fn a_fire_that_is_faster_than_a_plain_step_keeps_drafting() {
        let mut g = PayoffGuard::new();
        for _ in 0..GUARD_PLAIN_WINDOW {
            g.record_plain(1, 10.0);
        }
        for _ in 0..GUARD_MIN_FIRES {
            g.record_fire(4, 23.3);
        }
        assert!(g.may_fire());
    }

    /// A paused guard spends one fire every `GUARD_REPROBE` plain steps to
    /// re-check, so a workload that turns repetitive halfway through is picked
    /// up again with no external signal.
    #[test]
    fn a_paused_guard_re_probes_instead_of_locking_out() {
        let mut g = PayoffGuard::new();
        for _ in 0..GUARD_PLAIN_WINDOW {
            g.record_plain(1, 10.0);
        }
        for _ in 0..GUARD_MIN_FIRES {
            g.record_fire(1, 23.3);
        }
        assert!(!g.may_fire());
        for _ in 0..GUARD_REPROBE {
            g.record_plain(1, 10.0);
        }
        assert!(g.may_fire(), "the next fire is the re-probe");
    }
}
