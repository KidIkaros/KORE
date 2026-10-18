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
}

impl Default for DraftPolicy {
    /// `k = 3`, `min_match = 4`: the best measured operating point on this box
    /// (1.31x on a coherent story continuation, 94% of drafts accepted, 29% of
    /// steps fired). `k` alone is capped below.
    fn default() -> Self {
        Self {
            k: MAX_BATCH_COLS - 1,
            min_match: 4,
            max_match: 16,
            search_back: 16384,
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
        let mut out: Vec<u32> = Vec::with_capacity(n_new);
        let mut hist: Vec<u32> = tokens.to_vec();
        let mut pos = tokens.len();
        let mut next = if tokens.is_empty() {
            0
        } else {
            self.prefill(cpu, tokens, 0, &mut states)?
        };

        while out.len() < n_new {
            if next == eos {
                break;
            }
            let drafts = policy.drafts(&hist, next);
            let m = 1 + drafts.len();
            if drafts.is_empty() || pos + m > max_ctx {
                out.push(next);
                hist.push(next);
                next = self.forward(cpu, next, pos, &mut states)?;
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
            let outs = self.forward_verify(cpu, &batch, pos, &mut states, &mut arena)?;

            // Longest prefix of drafts the model reproduced.
            let j = drafts
                .iter()
                .zip(outs.iter())
                .take_while(|(d, o)| *d == *o)
                .count();
            if j < drafts.len() {
                self.rewind(&mut states, &arena, 1 + j)?;
            }
            stats.fired += 1;
            stats.drafts += drafts.len();
            stats.accepted += j;
            stats.hist[j] += 1;

            let committed = &batch[..1 + j];
            match committed.iter().position(|&t| t == eos) {
                Some(stop) => {
                    out.extend_from_slice(&committed[..stop]);
                    hist.extend_from_slice(&committed[..stop]);
                    break;
                }
                None => {
                    out.extend_from_slice(committed);
                    hist.extend_from_slice(committed);
                }
            }
            next = outs[j];
            pos += 1 + j;
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
    fn k_is_clamped_to_the_gemvs_column_width() {
        let p = DraftPolicy {
            k: 99,
            ..Default::default()
        }
        .clamped();
        assert_eq!(p.k, MAX_BATCH_COLS - 1);
    }
}
