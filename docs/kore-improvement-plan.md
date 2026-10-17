# KORE — Improvement Plan (phased, gated on verified behavior)

**Artifact of:** `docs/kore-systems-audit.md` (Meadows leverage-point audit).
**Sequencing rule:** promote what is already good (the verified 72.20+ tok/s baseline, the research), then add the missing structural pieces one at a time, with a **token-for-token regression test** before every commit.

---

## Verified anchors (do not break)

| anchor | measured |
|---|---|
| short-context decode (≥5 tok) | ≥ 80 tok/s (dry 72.20, steady-state 79.81–81.82) — hard floor |
| long context (4096) | ~1.8 tok/s (memory-bound, measured) — the failure to fix |
| kore-vulkan build | currently 3 errors (Q8 path half-built) — the immediate blocker |

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
