# DeepSeek sparse attention: what shipped, and whether it helps spec decode here

Research note, 2026-10-20. Question: DeepSeek's recent sparse-attention work (DSA →
CSA/HCA → CSA2) — does it help KORE, in particular the MoE speculative-decoding
milestone? Short answer: **no, and the reason is measurable, not ideological.**
The one idea from the same body of work that *does* fit is DSpark's
confidence-scheduled verification, which is free to try on the lookup drafter.

## 1. What was published

| When | Where | Mechanism | Their claim |
|---|---|---|---|
| 2025-09 | DeepSeek-V3.2-Exp, *DeepSeek Sparse Attention* (DSA) | Lightning indexer scores every preceding token with `I[t,s] = Σ_j w[t,j] · ReLU(q[t,j]·k[s])` over a few FP8 heads; the query attends only to the top-k selected KV entries (k = 2048). Core attention `O(Lk)`; the indexer itself is still `O(L²)`, just cheap. | Long-context cost roughly halves; quality on par with V3.1-Terminus. Short **prefill** runs masked dense MHA "to simulate DSA", because sparse is *slower* there. |
| 2026-04 | DeepSeek-V4 (arXiv 2606.19348) | Hybrid **CSA + HCA**. CSA compresses every `m` KV entries into one, then applies DSA top-k over the compressed entries; HCA compresses `m' ≫ m` and attends densely. | 1M context; at 1M, V4-Pro needs 27 % of V3.2's per-token FLOPs and 10 % of the KV cache. |
| 2026-09 | DeepSeek-V4.1-Flash, *Pushing the Limits of KV Cache Compression* | **CSA2**: cross-layer reuse (each layer is Full, Reindex or Reuse — Reuse borrows another layer's KV *and* its top-k indices), hierarchical sparse indexer (top-512, later layers search a candidate pool built by the first Full layer), FP4 main KV, SWA bounded replay. Ships **DSpark** spec decoding. | 890 bytes/token of global KV; ¼ the runtime KV of V4-Flash; decode FLOPs grow only 25 % from 4K to 1M. |

All three are **pretraining** changes. V4.1 trains sparse attention from scratch at
64K with no dense warm-up on 45T tokens; V3.2's DSA needed 944B tokens of continued
pretraining (1k warm-up steps distilling the indexer to the head-summed attention
distribution, then 15k sparse steps). A GGUF contains no indexer and KORE cannot
train one, so the architecture is out of reach by construction.

## 2. What it would be worth on this box, measured

The target is Qwen3.6-35B-A3B: 40 layers, full attention every 4th layer (10
layers), **2 KV heads**, head_dim 256. That is ~1.1 KB of Q8 KV per token per full
layer, i.e. **~11 KB/token**. Decode steps of the device-routed MoE
(`KORE_STEP_PROFILE=1`, prompts of 5 / 2 000 / 8 000 tokens):

| context | step | note |
|---|---|---|
| 96 | 16.14 ms | baseline |
| 2 000 | 16.36 ms | +0.22 ms (arm at 2194 MHz) |
| 8 000 | 16.76 ms | +0.62 ms (arm at 1640 MHz) |

Skipping `flash_decode_gqa_q8` entirely at 2K saves 0.39 ms — measured with the
skip arm running at a **32 % higher clock**, so that is an upper bound, not a
floor. The 8K skip arm was discarded: it was heat-soaked (72 °C) and came out
*slower* than the reference, and skipping attention also changes the expert
routing, so its delta means nothing.

The clock evidence is what makes the table usable across differing clocks: in
the interleaved rounds, 1182 MHz read 17.87 ms/step and 1571 MHz read 17.58 ms
(−25 % clock for +1.6 % time). The step is **bandwidth-bound**, so a FLOP
reduction like sparse attention has very little leverage here.

Sanity check from bandwidth: 11 KB × 8K = 87 MB of KV per token, ≈0.27 ms at the
card's 322 GB/s. At 32K it is ~350 MB (~1.1 ms); at 128K ~1.4 GB (~4.3 ms). So
attention only becomes a term worth optimizing somewhere past **64K**, and an
indexer pass over the same context would be paid on top of it.

## 3. Does it compose with speculative decoding?

Mechanically yes, economically no:

1. **Verification is already KV-amortized.** The `m+1` query rows of one target
   pass share the prefix: K/V is read once per step, not once per query. DSA/CSA
   select top-k *per query token*, so the draft rows get `m` different selections
   and the single coalesced KV read becomes `m` gathers. Per-query sparsity
   fragments exactly the sharing that makes verification cheap.
2. **The indexer is `O(L)` per query.** With `m` draft queries it is paid `m`
   times per target step, then amortized by the accepted tokens — roughly neutral
   per accepted token, never a multiplier.
3. **It attacks the wrong term.** Spec decode amortizes the *weight* reads of the
   target pass; on this box those are 8.0 of 16.1 ms (`gemv_q8_0` 3.0,
   `gemv_q2_k` 2.9, `gemv_q4_k` 2.1) and are read once per pass no matter how long
   the context is. Sparse attention reduces bytes per pass only after tens of
   thousands of tokens.

So they are orthogonal: spec decode reduces the *number* of passes, sparse
attention reduces *bytes per pass at long context*. Here the first is worth
1.3–1.7× and the second is under 4 % at the contexts we actually run.

## 4. What to take instead

- **DSpark's confidence-scheduled verification** (arXiv 2607.05147, shipped in
  V4.1) — **tried and closed**. The lookup drafter's only confidence signal is
  the match length, so `KORE_SPEC_PROFILE` now prints acceptance conditioned on
  it. On the dense 2B, with the cap at 16 and again at 64, the length is bimodal:
  either it saturates at the cap (looping output: 96–100 % acceptance, 1.64×) or
  it sits just above `min_match` (fibonacci prompt: 50–75 % on slot 1 decaying
  to 25–50 % by slot 6, 0.995× overall). Truncating `k` in the first regime
  throws away good drafts; in the second the whole fire loses, which the payoff
  guard already detects with the objective signal (tokens/ms), not a proxy. A
  useful schedule needs the *per-token* probabilities of a trained drafter,
  which this one does not have. Instrumentation kept, schedule not built.
- **Semi-autoregressive / block drafting** (DSpark, DFlash): releases drafters
  for Qwen3-4B/8B/14B and Gemma-4-12B — not for this target — and training one
  is a separate project. Parked.
- **DeepSelect top-k kernels**: 256-expert routing in one workgroup is a
  different problem from token selection over 100K+ entries; nothing to take.
- **CSA2 cross-layer reuse** (Reuse mode skips indexer Q entirely): a good idea
  for a future model that ships with sparsity, not for a frozen GGUF.

## 5. Verdict

Do not build sparse attention for KORE. Revisit if (a) the target checkpoint
ships with an indexer, or (b) ≥64K contexts become routine, where the measured
baseline is ~5 ms/token of KV read. Until then the MoE speculative-decoding
milestone stands; the confidence schedule it started with came back negative
(§4), so the milestone is now only the batch path.
