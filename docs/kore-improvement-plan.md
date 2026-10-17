# KORE — Improvement Plan (phased, gated on verified behavior)

**Artifact of:** `docs/kore-systems-audit.md` (Meadows leverage-point audit).
**Sequencing rule:** promote what is already good (the verified 72.20+ tok/s baseline, the research), then add the missing structural pieces one at a time, with a **token-for-token regression test** before every commit.

---

## Verified anchors (do not break)

> **UPDATE 2026-10-17 — the baseline anchors below were NOT reproducible.** On the
> same card, AC power, identical harness: `HEAD` itself (F32 KV, pre-Q8) measures
> **33.7 tok/s** at ctx 5, **28.7** at 1024, **21.0** at 2048 — not 80 / 7.3 / 3.8.
> The machine currently thermal-throttles (GPU ~1500 MHz of 2100, ~86 °C, 50 W of
> an 80 W limit) and `llama-server` holds ~3.2 GB RAM with swap exhausted. The
> original 80 tok/s floor is therefore **retired as an absolute gate**; use A/B
> ratios under identical conditions instead.

| anchor | originally recorded | re-measured 2026-10-17 (HEAD) |
|---|---|---|
| short-context decode (ctx 5) | ≥ 80 tok/s | 33.7 tok/s |
| long context (1024 / 2048) | 7.3 / 3.8 tok/s | 28.7 / 21.0 tok/s |
| kore-vulkan build | 3 errors (Q8 half-built) | green (Phase 0 done) |

## Phase 0 status: PASSED (2026-10-17)

Shipped in commit `0d49fe7`:

- Q8 (INT8) KV path: GPU per-row quantization (`quantize_q8_rows`) + packed-u32
  append (`kv_cache_append_q8`) + fused decode (`flash_decode_gqa_q8`); no
  per-token device→host round-trip.
- **Correctness gate passed:** decode token-ids are byte-identical to the F32
  reference.
- **Found + fixed a real bug:** the repacked Q2_0 GEMV kernel was striding 8
  blocks/iteration with only 32 lanes/row, silently skipping ¾ of every weight
  row. Fixed to 2 blocks/iteration; GPU-vs-CPU parity test added.
- GPU tests: 13/13 green in `kore-vulkan`.
- A/B vs HEAD (identical conditions): ctx5 33.7→41.2, ctx1024 28.7→37.3,
  ctx2048 21.0→25.9 tok/s (+22–30%).

## Phase 0.5 status: GEMV OPTIMISED — the remainder is ~300 small dispatches

Phase 0.5 was originally opened because `bench_gemv` showed the Q2_0 GEMV
moving weight bytes at ~22–36 GB/s against ~287–318 GB/s for the F32 GEMV. That
number is a red herring: the Q2_0 kernel reads 14x fewer bytes per weight, so
"byte throughput" is not its ceiling and the kernel was never bandwidth-bound.

What actually shipped (`ed5a38d`, `983d620`, `b45e4ad`):

- **`unpack2x16float` for the block scales** (`ed5a38d`). The fp16 scale was
  decoded by a branchy hand-rolled `fp16_to_f32`, executed redundantly by all 32
  lanes on every k-iteration. One instruction replaces it.
- **Drop the shared-memory LUT** (`983d620`). The 256-entry vec4 code table was
  read on the critical path every iteration (4-way bank conflicts) and its 4 KiB
  plus init barrier cost occupancy. The four 2-bit deltas are now extracted
  arithmetically.
- **Coalesced code layout** (`f136cd5`) and **4 output rows per warp**
  (`b45e4ad`), which cuts the kernel's x-read traffic 4x.

Result: short-context decode **41.2 → 59.4 tok/s** (median of 7; cumulative
+44% over the P0 baseline, +76% over HEAD), with byte-identical token ids and
13/13 GPU tests green.

**The corrected cost model** (Bonsai-2 2B, ctx 5, 16.8 ms/token):

| component | time | notes |
|---|---|---|
| Q2_0 GEMV (all shapes) | ~11 ms | 1.78 B weights at ~145–200 Gweights/s |
| — of which LM head | ~2.5 ms | n=248320, the largest single shape |
| everything else | ~5.6 ms | ~300 dispatches: rms_norm, hadamard, swiglu, elementwise, argmax, attention/SSM |

- The GEMV now runs at **145–200 Gweights/s**, i.e. within ~15% of what
  llama.cpp's `mul_mat_vec_q` achieves with the same warp-per-row design on
  comparable hardware. Further micro-tuning here has low headroom.
- The 4-rows-per-warp change cut x traffic 4x and made the LM head ~20% faster,
  but left short-context decode **unchanged within noise**: x reads are largely
  cache-served, so traffic was not the bottleneck. It is kept for the
  bandwidth-bound offload case the engine is being built for.
- Decode runs at **90–92% GPU utilisation at the 80 W power limit**, so it is
  GPU-bound, not dispatch-bound: the win must come from doing fewer
  instructions, and the remaining lever is the ~300 small kernels, not the GEMV.

**Implication:** Phase 1 (paged KV) still matters for long context, but for
short-context tok/s the next lever is **fusing the small latency-bound
dispatches** (e.g. rms_norm+hadamard, and epilogue folds into the GEMV), not
further Q2_0 GEMV work.


---

## Phase 0 — close the build integrity gap (immediate)

**Goal:** kore-vulkan builds green with the Q8 path, on short context, with a regression test.

Takeaways from the audit:
- **F8 (buffers):** add a KV-budget guard so `new_state(max_ctx)` fails loudly instead of silently OOMing.
- **F5 (parallel systems):** one GPU KV representation; the Q8 path must be a *replacement* of the F32 path on the GPU decode, not a parallel dead branch.
- **F8 (regression):** token-for-token decode comparison (GPU Q8 vs CPU F32) at short context before merging.

1. Fix the 3 kore-vulkan compile errors:
   - `ElementwiseUniforms` gains `dst_offset`/`src_offset` fields (only used by the U8 elementwise copy path — the kv-cache append is the U8 path, so this stays clean).
   - `VulkanBackend` gains `kv_cache_append_q8_pipeline` + `flash_decode_gqa_q8_pipeline` fields, bound in `create_pipelines`, dispatched in `kv_cache_append_q8` and `fused_attention_decode_gqa_q8`.
   - `Q8GqaUniforms` declared with its `q` scale field and the `bytemuck::Pod`/`Zeroable` derives that the elementwise U8 call site requires.
2. Wire `fused_attention_decode_gqa_q8` / `kv_cache_append_q8` call sites in `kore-nn/src/qwen35.rs` (already partial).
3. **Verification gate:** run bonsai at short context (`8 gpu`); require output token-for-token equal to the F32 CPU baseline *and* ≥ 80 tok/s. A token mismatch = the Q8 decode is wrong; do not proceed.
4. Add a regression test (GPU `fused_attention_decode_gqa_q8` vs CPU F32) in `kore-vulkan` so a broken decoder fails CI.

---

## Phase 1 — paged KV + the information-flow fix (the highest leverage)

**Goal:** break the reinforcing loop (F3). 4096-context decode moves from ~2 toward double-digit tok/s.

Takeaways from the audit:
- **F1 (stock):** KC/VC as one contiguous F32 tensor is the binding constraint — replace with paged KV blocks (SOAR-style ring of fixed-size blocks, each block owned by one layer). This also makes the "max_ctx" a soft budget, not a hard OOM.
- **F2/F4 (info flow):** GPU decode reads INT8 KV (+ parallel F32 scales) instead of full-resolution floats. `kv_cache_append_q8` writes U8 rows + per-token scales; `fused_attention_decode_gqa_q8` reads U8 + scales and dequantizes per-thread in `flash_decode_gqa_q8.wgsl`. This is the Strata/sudoingX mechanism (1.58-bit ternary / INT8 KV + inline dequant).
- **F7 (rule):** fusing the attention decode into the KV append amortizes the per-token VRAM read to O(1) per block rather than O(seq_k).

1. In `kore-vulkan`: introduce paged KV blocks (fixed-size `[block_ctx, kv_dim]` per layer, free-list allocator; `max_ctx` becomes the number of blocks × block_ctx).
2. Rewrite `fused_attention_decode_gqa_q8` to: for the decode window, read only the current block's KC/VC (U8 + scales), dequantize per-thread, run the softmax + V-accumulate, and **return already-accumulated hidden states** so the next step's attention reads a bounded window (not the whole KV). Reuse the flash fuses so no extra passes.
3. Keep `fused_attention_decode_gqa` (F32) as the slow path only while the transition is verified.
4. **Verification gate:** bonsai at 4096 context (`4096 gpu`) — require decode to move from ~2 to double-digit tok/s (≥ ~10–20) while short-context (≥5 tok) stays ≥ 80 tok/s. A regression → short-context < 80, or a tile-boundary bug → wrong tokens.

---

## Phase 2 — throughput, only after Phase 1 is green

**Goal:** push the numbers with compute-friendly changes, now that the memory path is fixed.

Takeaways:
- **F9 (buffers):** add continuous/mtp batching so the decode reads the KV in a coalesced async pattern (already "in theory" for this workload; verify on the 8 GB card).
- **F6/F7:** if the remaining decode steps are still compute-light, look at argmax fusion + KV pre-allocation reuse — but **only after Phase 1 is verified**, because that is where the bloom is.

1. Continuous batching / multi-token prefill decode.
2. Async KV read pipeline (coalesce per-layer reads).
3. `argmax` fusion in the decode loop.

---

## Rollback policy (no parallel systems)

- If **any** phase regresses short-context below 80 tok/s, stop that phase and revert to the last green commit (the F32 decode path, which is verified at 72.20–81.82 tok/s).
- The map and tracker (`docs/wayfinder/`) are the source of truth for what is green vs. in-flight; log each gate as `PASSED / BLOCKED (reason)` with the measured numbers.
- Do not commit a half-built representation (dependencies between `kore-nn`, `kore-vulkan`, and the `Qwen35Gpu` struct are tight but not yet verified) — finish Phase 0 first, then Phase 1.

---

## Next decision (gated)

**Where we are.** Phase 0 is PASSED and Phase 0.5's GEMV work is done: short
context is **59.4 tok/s** (median of 7) against 33.7 for HEAD, with byte-identical
tokens. The GEMV is now within ~15% of llama.cpp's efficiency for this shape, so
the remaining short-context lever is the **~5.6 ms/token of ~300 small
latency-bound dispatches**, and the remaining long-context lever is still
Phase 1 (paged KV).

**Note on the retired gate.** The "≥ 80 tok/s" short-context floor appears
throughout the phases below but does **not** reproduce on this machine: HEAD
itself measures 33.7 tok/s @ctx5. It is retired as an absolute gate — gates must
be expressed as no-regression *ratios* measured under identical conditions
(same prompt, 1 warmup + 6 timed runs, clocks recorded), because the pinned
curve was taken under different power/thermal conditions.

Proposed order:

1. **Fuse the small dispatches** (short-context tok/s). Start with
   `rms_norm` + `hadamard_signed` (they run back to back ~24x/token on the same
   2048-vector) and fold the residual `add` into the GEMV epilogue. Target: cut
   the ~5.6 ms non-GEMV share by a third without touching numerics.
2. **Phase 1 (paged KV)** — the long-context bottleneck (F1–F4): decode reads
   grow as O(seq_k), which is what takes 4096-context from ~59 to ~18.7 tok/s.
3. **M1** (27B ternary + lean MTP graft) and **M2** (MoE expert offload) as
   planned.

