# KORE — Systems Audit (Meadows Leverage-Point Lens)

**Component audited:** KORE's GPU decode / inference engine (`kore-nn` → `kore-vulkan`).
**Audit lens:** Donella Meadows, *Thinking in Systems* — identify the highest-leverage structural gaps, not parameter tuning.
**Audit date:** 2026-10-16. **Basis:** probed behavior on the user's actual hardware (8 GB RTX 2070 Max-Q, Qwen35 Q2_0, Vulkan).

---

## The system as read

**Stock (per layer):** KV cache, `KC/VC`, a single contiguous tensor `[max_ctx, kv_dim]` pre-allocated in VRAM.
**Flow 1 (write):** each new token's K/V is computed, then appended to KC/VC (`kv_cache_append_pair` / `kv_cache_append_q8`).
**Flow 2 (read, per token):** `fused_attention_decode_gqa` reads the **entire** KC/VC from VRAM, `[seq_k, kv_dim]` bytes per layer, then does the flash-decode softmax + V-accumulate.
**Feedback loop (reinforcing):** decode time ∝ seq_k. Short context → fast decode → system stays short; long context (chat, documents) → slow decode → system collapses. The "fast" regime and the "slow" regime are connected only by the cost of reading the whole KV.

This is a **memory-bound, O(n)-read-per-token** decode — measured, not assumed:

| context | decode tok/s |
|---|---|
| 5 | ~80 |
| 128 | 22 |
| 1024 | 7.3 |
| 2048 | 3.8 |
| 4096 | 1.8 |

---

## Findings (severity-ranked)

| # | Leverage point | Finding | Severity | Effort |
|---|---|---|---|---|
| **F1** | LP #10 (stock structure) | KC/VC is one contiguous F32 tensor `[max_ctx, kv_dim]` per layer — can't be slid/reused, forces synchronous pre-allocation. At 8 GB VRAM this is the binding stock. | **CRITICAL** | paged KV blocks (vLLM SOSP 2023) |
| **F2** | LP #10 + LP #5 (stock + info flow) | KC/VC stored F32 on GPU; decode shader reads unquantized K/V. The quantized K8V4/K8V0 modes in `kore-attention` live only in the CPU path and are never wired into the GPU. This is the ~4× VRAM-read loss behind F2. | **CRITICAL** | store KC/VC as INT8+scales; decode reads INT8 + scales |
| **F3** | LP #6 (reinforcing loop) | decode time ∝ context (O(n) KV read/token). Short-context regime fast, long-context collapses — the system structurally prefers short context; real chat/documents unusable on 8 GB. | **HIGH** | paged KV + fused INT8 decode + quantization |
| **F4** | LP #5 (missing info flow) | The GPU decode reads K/V as unquantized float arrays; the quantized K8V4/K8V0 modes in `kv_cache/src/kv_cache.rs` are dead on the GPU path. Strata/sudoingX's winning mechanism is exactly this info-flow compression. | **HIGH** | wire K8V4 (INT4 V, per-thread FWHT) + paged KV into the GPU decode |
| **F5** | LP #3 (self-organization / parallel systems) | The GPU path has two KV representations (F32 + U8) with no selection, and the CPU path has three (F32/Q8_0/K8V4) while the GPU uses only F32. Unwired parallel systems invite the stale-binary symptom that just blocked the build (the old F32 binary ran while the new Q8 declarations compiled separately). | **HIGH** | one GPU KV representation; gate the Q8 path + regression tests |
| **F6** | LP #2 (goals) | The speed effort has been optimizing matmul GEMM (already fast on the 8 GB card) because the "massive gain" framing was kernel-oriented. The measured bloom is the KV/attention memory path, not the matmul. | **MEDIUM** | re-target the speed effort at the KV/attention memory path |
| **F7** | LP #4 (rules) | The decode rule "read the entire KV every token" is an implicit un-parameterized rule that dominates behavior at realistic context. | **MEDIUM** | fuse the attention decode into the KV append (O(1) amortized per token) |
| **F8** | LP #8 (buffers) | No memory guard: `new_state(max_ctx)` allocates a fixed buffer that OOMs if context exceeds budget; no regression test on the Q8 decode path, so a broken decoder could pass the bonsai example by emitting wrong tokens. | **MEDIUM** | KV-budget guard + token-for-token GPU decode regression test |
| **F9** | LP #8 (buffers) | No async/coalesced KV read; per-token KV read is a separate streaming read with implicit sync (decode is compute-light, memory-bound; the sync hides throughput). | **LOW** | batch KV reads (continuous/mtp batching) |
| **F10** | LP #1 (paradigm) | Original 8–16 GB gaming-card assumption is still baked into the matmul-first architecture; the user's card is 8 GB Max-Q. Design lens should now be "memory-bound on 8 GB." | **LOW-MED** | re-ground in the 8 GB reality (already closed as an assumption in the map) |

**Most-valuable-first ordering:** F1, F2, F3 (structural, the actual bottleneck) → F4, F5 (information flow — the Strata/sudoingX mechanism) → F6, F7 (goals/rules) → F8–F10 (build + paradigm).

---

## Diagnosis in one line

KORE's decode is slow on the 8 GB card because the **stock (KV cache) is stored in the wrong shape for the flow (per-token full-VRAM reread)**. The highest-leverage fix is not more matmul GEMM — it is **restructuring the KV stock (paged blocks) + the information flow (INT8 decode)**.

---

## Current state (as of the pause point)

The KORE build is **NOT green**. My K8V4/INT8 surgery (`fused_attention_decode_gqa_q8`, `kv_cache_append_q8`) is implemented but the `kore-vulkan` build fails with 3 compile errors:
- `ElementwiseUniforms` has no `dst_offset`/`src_offset` fields.
- `VulkanBackend` struct has no field `kv_cache_append_q8_pipeline`/`flash_decode_gqa_q8_pipeline`.
- `Q8GqaUniforms` not declared.

So the architecture is declared but **not exercised** — the map must not mark the "memory-path fix" as done. The verified-good artifacts remain: the real-run baseline (≥72.20 tok/s, median 80.5), the research findings, and the tooling ladder. Those are the anchors to re-verify against after the structural fix.

---

## What "done" looks like (the destination)

A KORE decode path where, on the same 8 GB card:
- short-context decode stays **≥ 80 tok/s** (baseline holds),
- 4096-context decode **moves from ~2 toward double-digit tok/s** (the KV/attention memory path is fixed),
- the GPU KV path is **one representation**, wired and tested, not parallel-and-dead,
- the build goes green on the memory-path changes (no stale-binary symptom).
