# DeepInfer ideas, scaled down to one RX 9060 XT

DeepInfer targets SM100/Ascend clusters running DeepSeek-V4.1-Flash on vLLM. None
of that stack runs here, and copying it wholesale would be a rewrite, not a
port. What does transfer is the *reasoning* behind five of its mechanisms. This
note maps each one onto KORE as it exists today, says whether it is already
present, worth building, or out of reach, and records the evidence.

The box: RX 9060 XT, 16 GB, Vulkan/radv, wgpu. Dense baseline
`Ternary-Bonsai-2-2B-Q2_0`; MoE baseline `Qwen3.6-35B-A3B-Q2_K_L`
(`qwen35moe`, 41 layers, 256 experts, 8 used).

## Summary

| DeepInfer mechanism | KORE equivalent | Verdict |
|---|---|---|
| Batch-invariant kernels | Q2_0 / Q3_K GEMVs are bit-identical for any `m` | **Already present**; keep as a test invariant |
| Confidence-scheduled verification (DSpark) | n-gram drafter + `DraftPolicy` payoff guard | **Covered**; adaptive `k` has no headroom at `k=3`, `MAX_BATCH_COLS=4` |
| Expert hot-set pinning (EPLB) | `KORE_MOE_RESIDENT=N` (first *N* experts) | **Worth building for 8 GB**; no gain on 16 GB |
| Zero-overhead / async scheduling | 48-op batches, one queue per step | **No cheap win**; the step is a serial dependency chain |
| Megakernel / persistent kernel | none | **Out of reach in wgpu**; this is the remaining MoE wall |

## 1. Batch-invariant kernels

DeepInfer's point: a decode step at `m=1` and the same token inside a verify
batch at `m=4` must produce the same logits, or speculative acceptance is
measuring kernel scheduling rather than the model.

KORE already guarantees this at the kernel level. The Q2_0 shader states it
directly: *"an output row is bit-identical whatever `m` is"* — each output row
reduces over the same lanes in the same order, and `m` only selects which rows
run. The Q2_K/Q3_K kernels follow the same rule (one workgroup per four rows,
per-row reduction in `sdata`).

**Do:** keep this as an explicit test, not a comment. A cheap invariant is
`forward_verify([t])` logits == `forward_verify([t, x, y, z])[0]` on a fixed
prompt. It already underpins the spec numbers: 100 % draft acceptance on the
repeat prompt (`P(j)=0/0/0/15`) is only meaningful if the verify pass is
batch-invariant.

## 2. Confidence-scheduled verification

DeepInfer's DSpark schedules how many draft tokens to verify from the drafter's
confidence. KORE's drafter is not a model: it is prompt-lookup n-gram matching
(`kore-nn/src/spec.rs`), and the confidence signal is the match length. The
existing `DraftPolicy` already has the equivalent guard: it fires only when a
match of at least `min_match` exists, and the payoff guard skips steps whose
recent acceptance would not repay the wider verify pass.

Measured on the dense 2B (`k=3, min_match=4`, 64 tokens, 5-token prompt):

- fires 71.4 % of steps, 45/45 drafts accepted (100 %), all at `j=3`;
- **189 tok/s vs 118 tok/s greedy = 1.60×** in the current (lower-clock) state;
- the no-fire floor on prose is ~0.975–0.989× (§4 of the handoff), i.e. the
  guard costs 1–2.5 % when there is nothing to draft.

**Verdict:** covered. Adaptive `k` would need `k` to exceed the current
`MAX_BATCH_COLS - 1 = 3` and a confidence estimate finer than match length.
Both are properties of the drafter, not of scheduling. The real version of this
idea is a small draft model, which is a different project.

## 3. Expert hot-set pinning (EPLB)

DeepInfer's EPLB rebalances experts across devices from observed routing
frequency. Scaled to one GPU, the analogue is **partial residency**: keep the
hottest experts resident and stream the tail.

KORE's `KORE_MOE_RESIDENT=N` uploads the **first N experts of every layer**, in
file order. That is the wrong set: expert popularity is not uniform, and file
order is unrelated to it. Ranking by observed routing frequency and pinning the
top N would be strictly better for the same VRAM.

**Measured constraint:** on the 16 GB box it does not matter. With the whole
35B-A3B bank resident (`KORE_MOE_RESIDENT=all`, peak 16,156 MiB of 17,095 after
the `flush_uploads` fix), streaming and residency are identical: 16.56 vs 16.51
tok/s warm. The win would only appear on the 8 GB card the project came from.

**Do (8 GB only):** run a calibration corpus with a per-expert counter, rank
experts per layer, upload the top N, stream the rest. The counter is a few lines
next to `moe_topk`; the upload set is one sort at load. Do not build it for the
16 GB box.

## 4. Zero-overhead / asynchronous scheduling

DeepInfer overlaps host scheduling with device work. KORE's version of that is
the 48-op batch in `record_dispatch` and the one-submit-per-step loop. The
handoff already measured the ceiling: raising the batch from 48 to 512 moves
280 ms → 69 ms in the submit region but the queue drain rises by the same
210 ms — **the sum is flat**. The decode step is a serial chain
(`x → norm → qkv → … → logits`), so there is nothing to overlap at batch 1.

The one place overlap did pay was removing the host from the MoE loop: the
device-routed path is **1.44×** the host-routed path (11.8 → 17.1 tok/s,
interleaved, n=64). That is the same idea — stop round-tripping per layer — not
a scheduler trick.

**Verdict:** no cheap win left here without speculative execution, which already
has its own section.

## 5. Megakernel / persistent kernel

DeepInfer fuses an entire layer (or several) into one persistent kernel. This is
what the remaining MoE gap requires. The current device path is **1143
dispatches/step, 54.5 ms** (no-sync census), against llama.cpp's **80.25 tok/s**
(12.5 ms/step) on the same file and card. Cutting 280 launches (fused `down`)
bought only ~2 ms, i.e. ~6.5 µs per removed dispatch: the step is **latency-bound
on small dependent ops**, not launch-count-bound.

Getting from 17 to 80 tok/s means collapsing each MoE layer into one or two
dispatches: route → gate/up → SwiGLU → down → weighted sum, with the expert
matvecs reading the resident bank in place (`MUL_MAT_ID`-style, no gather). That
is a real kernel project. It is not portable from DeepInfer's CUDA/SM100 code,
but the design is: one workgroup per selected expert, weights addressed by
`idx`, no host and no intermediate stage buffers.

**Verdict:** out of reach as a port; the correct next milestone if the MoE gap
is the goal.

## What was actually adopted

- **Device-side routing** (`KORE_MOE_DEVICE_ROUTE=1`): router matvec, softmax,
  top-k, and gather all on device; one round trip per layer removed. 1.44×.
- **Expert-strided `down` GEMV** (`gemv_q{2,3}_k_id`): all eight experts in one
  dispatch, reading the gathered bank by expert stride, then one `sum_rows`.
  1423 → 1143 dispatches/step.
- **`flush_uploads`** for the bulk expert upload: GTT peak fell from ~16 GB to
  ~1.5 GB by submitting the staging belt instead of letting it pile up.

The batch-invariance rule is retained as the acceptance gate: every MoE
correctness check compares `max|d|` against the host path (2.2e-8 on this box)
and the token ids against the oracle, per batch shape.
