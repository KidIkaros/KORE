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
| Confidence-scheduled verification (DSpark) | n-gram drafter + `DraftPolicy` payoff guard | **Covered**; default is `k=6, min_match=6` at `MAX_BATCH_COLS=8` |
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

Measured on the dense 2B (`k=6, min_match=6`, 200 tokens, 2 rounds
interleaved with greedy on the same prompt):

- repetitive text: **173 tok/s vs 109.5 tok/s greedy = 1.58×** (the old
  `k=3, min_match=4` point reads 167 tok/s, 1.53×);
- prose: 147.5 vs 106.7 tok/s = **1.38×** (old point: 145.8, 1.36×);
- the no-fire floor on prose is ~0.975–0.989× (§4 of the handoff), i.e. the
  guard costs 1–2.5 % when there is nothing to draft.

`MAX_BATCH_COLS` is now 8, so `k` can reach 7; `gemv_q2_0`'s second row group
made an 8-column verify pass bit-identical. `draft_headroom` sweeps the whole
grid (`m=1..8`, `k=1..7`) and picks the operating point: the fully-accepted cost
per token falls from 3.88 ms at `m=4` to 3.51 ms at `m=7`, and the measured
optimum is `k=6`. `k=7` loses because the eighth draft almost never lands.

**Verdict:** covered. The remaining gap is a confidence estimate finer than
match length — the `P(j)` histogram already in `SpecStats` is the natural
signal. The real version of this idea is a small draft model, which is a
different project.

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

DeepInfer fuses an entire layer (or several) into one persistent kernel. The
current device path is **823 dispatches/step, 14.9 ms** (no-sync census) and
**57–58 tok/s** in the real loop, against llama.cpp's **80.25 tok/s**
(12.5 ms/step) on the same file and card.

The ablation changed the diagnosis. Dispatch count is not the wall: the step
was ~75 % two kernels, `moe_topk` (a single-lane softmax + selection, 21.8 ms)
and the transposed F32 router matvec (~15 ms). Parallelising the first and
reading the router in its native row-major layout moved the device lane from
1.44× to **3.8×** the host without fusing a single layer. The remaining 17 ms is
spread thin: `gdn_step` 4.9, attention 6.1 (Q4_K + Q8_0), MoE Q2_K 3.4,
`had_pre` 2.9. The cheap fusions have been taken (`swiglu_rows` folds the
routing weight in, `sum_rows_add` folds the shared expert in: 120 dispatches and
+1.7 %), and one router matvec now feeds both the top-k and the shared gate.
What is left is kernel work, not launch work.

That leaves the megakernel as a *second* milestone, not the first. Collapsing
route → gate/up → SwiGLU → down → weighted sum into one or two dispatches is
still the way past llama.cpp, and the in-place `idx` read is the foundation it
needs, but the GDN and attention shaders are now the larger single items.

**Verdict:** the in-place design was adopted; the full layer fusion is the next
kernel project, behind the GDN shader work.

## What was actually adopted

- **Device-side routing** (`KORE_MOE_DEVICE_ROUTE=1`): routing, softmax, top-k
  and the router matvec all on device. 1.44× before the kernel fixes below,
  3.8× after.
- **Parallel top-k** (`moe_topk.wgsl`): 256 lanes, tree-reduced max and argmax
  rounds, normaliser kept on one lane for host-order summation. 21.8 ms off.
- **Row-major router** (`router_matvec.wgsl`): one workgroup per expert over the
  GGUF's native `[experts, hidden]` rows. ~15 ms off.
- **In-place expert read** (`idx` + `x_stride` in `gemv_q{2,3}_k_id`): the three
  per-layer gathers and the down staging buffer are gone; 1143 → 1023
  dispatches, ~165 MB less traffic per token.
- **`swiglu_rows` / `sum_rows_add`**: the routing scale rides out with SwiGLU
  (mode-2 `row_scale`) and the shared expert rides in with the `down` reduction
  (bias binding). 1023 → 903 dispatches, bit-identical to the split ops.
- **`flush_uploads`** for the bulk expert upload: GTT peak fell from ~16 GB to
  ~1.5 GB by submitting the staging belt instead of letting it pile up.

The batch-invariance rule is retained as the acceptance gate: every MoE
correctness check compares `max|d|` against the host path (5.6e-8 after the
router change) and the token ids against the oracle, per batch shape.
