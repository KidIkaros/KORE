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

## Phase 0.5 status: MEASURED — Q2_0 GEMV is the dominant, inefficient cost

`bench_gemv` (commit `0d49fe7`) shows the Q2_0 GEMV moving weight bytes at
**~22–36 GB/s**, versus **~287–318 GB/s** for the F32 GEMV on the same card. A
10× drop in byte-throughput despite reading 14× fewer bytes per weight means the
Q2_0 kernel is **not** bandwidth-bound — the limiter is the access pattern
(the repack interleaves code words as `j*bpr + b`, so a warp's code reads are
scattered/broadcast rather than coalesced) and/or the per-element ALU/LUT work.
Decode is therefore bottlenecked here, not on FLOPs or dispatch count:
reconstructing a token from per-shape GEMV latency gives ~18 ms of Q2_0 weight
reads per token (of ~24 ms total), with the Q2_0 LM head (`n=248320`) alone at
**3.9 ms** (~16%).

**Implication:** raising tok/s (and making M1's 27B target reachable) is gated on
fixing the Q2_0 GEMV (coalesced code layout + drop the shared-memory LUT) and/or
the LM head. This is the highest-leverage kernel work and should precede the
paged-KV work in Phase 1.


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

Phase 0 is the immediate blocker (3 build errors) AND the highest-leverage contained fix. Propose: **finish Phase 0** (get the Q8 path green + regression-tested), then run the Phase 1 paged-KV + INT8 decode on the 8 GB card, and present the before/after numbers. This directly answers the audit's "what do we do."
