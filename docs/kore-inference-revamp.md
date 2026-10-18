# KORE Inference Revamp — Plan (v2, research-hardened)

**Focus for this effort:** the inference engine.
**Constraint:** KORE's other parts (autograd, optim, training, `kore-btes`, `kore-clifford`, `kore-data`, `kore-edge`, Python bindings) stay. Nothing is deleted. This plan only redirects where the *attention* goes.

**Goal (one line):**
> Run models that are far bigger than the available VRAM — including MoE models — at high tokens/sec, on whatever hardware is present, in pure Rust.

**Reference targets (why this is believable):**

| Rig | Model | tok/s decode | Provenance |
|---|---|---|---|
| RTX **2070 Super 8 GB** (Turing sm_75, same class as our Max-Q) | Ternary Bonsai-2 **27B** (PTQ1_0, ~2-bit) | 42.6 (no draft) → **52.6** (MTP n-max 1); JIT-sm75 59.5 | sudoingX `sweeps/rtx2070super-8gb.md` |
| RTX **5070 12 GB** | Qwen3.8-Flash-Next **~125B MoE** (Q2_0) | 87–93 @1–4K, 74 @128K, 60 @262K | Strata `docs/DETAILS.md` |
| RTX 3000 Ada Laptop 8 GB | Ternary 27B + lean MTP | 27.8 base → 36.1 (+33%, lossless) | sudoingX `sweeps/rtx3000Ada-8gb.md` |

Our `older` v1 plan named these but treated them as inspiration. They are now the **acceptance bar**, and the mechanisms behind them are no longer guesses — they are read out of both codebases (Section 1).

---

## 0. Decision (record this)

- The paradigm for this effort is **inference-engine / big-model-on-small-hardware**.
- `docs/inference-engine-design.md` says KORE is *not* a serving product and that "serve tokens fast" is subordinate. That framing is **superseded for this effort**; the other crates and goals it protects remain.
- Keep the design doc's hard rules that still apply: safetensors/GGUF ingress, no C/C++ in the runtime path, backend trait (not a monolithic op), **correctness gate before any perf claim**.
- **New hard rule (from the references):** a capability is only "done" when its output is **byte-identical across the batch shapes the runtime actually uses** (draft/verify windows, batch slots). This is what makes speculative decoding *lossless*; without it spec decode is throughput-only.

---

## 1. What the two references actually do (read from source, not the README)

### 1.1 Strata — 125 B MoE on 12 GB

**Who does what (the corrected memory model — this matters):**

- **VRAM:** attention + DeltaNet/GDN mixers, gated-residual weights, routers, shared experts, output head, the MTP draft layer, the KV cache (**only its most-read part from 64 K; the rest streams from RAM**), and an **adaptive expert cache** that fills the *remaining* VRAM with the most-used experts and keeps re-learning them during the conversation.
- **RAM:** **all 24,576 experts, pinned.** The CPU computes the experts the GPU does not hold **in place, concurrently** with the GPU (AVX2/AVX-512 + AVX-VNNI kernels).
- **SSD:** a **28.8 GB n-gram table** — read a few rows per token through the OS cache. **It is a drafting source, not weight storage.** Weights touch disk only in the optional low-RAM mode (`--mmap-experts` / `experts.bin`), which Strata itself calls *"much slower"*.
- **More VRAM beats a faster GPU:** ~**700 experts per extra GB**, and every expert on the GPU is one the CPU does not compute.

**The mechanisms (each independent, each measured):**

1. **MoE sparsity** — ~10 of 24,576 experts per token. Per-token work = small dense model. *The fit trick.*
2. **Aggressive quant** — Q2_0 / IQ2_XS / IQ3_S ≈ 2–3 bit, plus a **one-time packed CPU layout** (`experts.bin`, ~40 GB) so the CPU expert kernel is fast without dequant-to-FP16.
3. **Concurrent CPU+GPU expert execution** — GPU does cached experts; CPU does misses *at the same time*. Plus `--pcie-frac`: a tunable share of misses is copied to the GPU over PCIe instead of computed on CPU (tuned per-PC by `--calibrate`; fast PCIe + slow CPU want more).
4. **Adaptive expert cache** — swaps learners (`--adapt-every`, `--adapt-swaps`, `--adapt-decay`, `--adapt-min-gain`), and `--adapt-async 1` moves the swaps on a helper thread *between* decode windows instead of blocking a window.
5. **Router look-ahead prefetch** — apply the *next* layer's router to *this* layer's input to predict next experts, then `madvise`/`pread` their pages before they are needed (`STRATA_LOOKAHEAD`, `STRATA_IO_PREFETCH`).
6. **Guess-then-check (lossless)** — model's own **MTP layer drafts ≤3 tokens**; one pass over all 48 layers verifies them → **2.4–3.2 tokens/pass**; plus **prompt/suffix lookup** drafts ≤5 tokens from earlier context (code edits 6–11 % faster). Draft head only proposes from a **vocabulary subset** (~180 MiB).
7. **8192-token batched prefill** — chunk up to 8192 tokens; **the next layer's experts stream over PCIe while the current layer's attention runs**; quantized MMQ kernels (no FP16 expansion); PLE block whole-chunk. → 1,299 tok/s @4K, 2,171 @32K.
8. **KV streaming from 64 K** — `--kv-resident 32768`: keep only the attention window in VRAM, the rest pinned in RAM (≈13.7 KB RAM/token). At 262 K: 50.9 → 62.6 tok/s *because it frees VRAM for ~2,300 more experts*.
9. **Quantized KV** — INT8 default; **K8V4** (K 8-bit, V Hadamard-rotated 4-bit) → 23 % less KV → more experts cached (3090 @198 K: 85 → 99 tok/s). Both stream.
10. **Hardware-agnostic engine** — the *same* C++ engine compiles for CUDA **and** HIP; helper/2nd GPU is *slower* than the CPU pool, so multi-GPU is not the play.

**Explicitly measured as *not* worth it:** a helper second GPU (per-layer round trip costs more than the CPU pool needs); multi-slot batching on a small card (buys *latency*, costs 11–24 % per-request speed — it only adds throughput when experts mostly fit VRAM or on a layer split).

### 1.2 sudoingX — dense ternary 27 B on 8 GB

- **The 8 GB lift is two things, stacked:**
  - **A grafted MTP head** that the model doesn't ship: extract `blk.64.*` (MTP decoder block + `nextn.eh_proj/enorm/hnorm/shared_head_norm`) from a donor **Qwen3.8-27B** GGUF, merge with `qwen35.block_count 64→65` and `qwen35.nextn_predict_layers=1`. → 27 B on 8 GB goes **42.6 → 52.6 tok/s** (+23–26 %).
  - **A ternary mat-vec kernel**, not a generic one: planar-transposed activation layout (the 128 quants a thread needs become 8 aligned 16-byte pieces + 1 scale piece) **and** flattened `(row group, K block)` work items so *no thread idles* (the stock kernel left 88 of 128 threads idle on the K=5120 projections that hold ¾ of the weights). dp4a; tensor cores only pay at ≥8 columns. **+53 % on a bandwidth-limited 3060, only +7 % on a bandwidth-saturated Ada 8 GB** — i.e. this lever is a *bottleneck-removal*, not free speed.
- **`GGML_CUDA_BATCH_INVARIANT=1`** makes 1–4-column batches **bit-identical** — required for the MTP graft to be *lossless* greedy. Costs ~12 % on one build; worth it.
- **KV:** q4_0 K **and** V, in-place flash attention (no F16 scratch).
- **The depth wall is *exactly* KORE's bug:** at 131,072 ctx the 16 full-attention layers read **4.8 GB of KV per token**, so *flash attention — not the projections — sets the pace* (88 ms/token, projections 25). MTP helps *more* at depth because a verify pass reads KV once for 3 tokens.
- **Hadamard-basis gotcha:** Bonsai keeps `token_embd.weight` in a rotated basis; the MTP graph must apply the inverse or it is refused at load. (KORE has Hadamard work in `kore-clifford`/`kore-btes` — directly relevant to graft work.)

---

## 2. Pinned baseline (do not break)

Measured on RTX 2070 Max-Q 8 GB, Vulkan, Bonsai-2 **2B** Q2_0 (`crates/kore-cli/examples/bonsai.rs`, `kore-tools/run-kore-dryrun.sh`):

| context | decode tok/s |
|---|---|
| 5 | ~80 (median 80.5, 7/7 ≥ 72.2) |
| 128 | 22 |
| 1024 | 7.3 |
| 2048 | 3.8 |
| 4096 | **1.8** |

**Hard floor:** short-context (≥5 tok) decode stays **≥ 80 tok/s** through every change (rollback trigger).
**The failure to fix:** decode is O(seq_k) — every step re-reads the *entire* KV cache (`kore-systems-audit.md` F1–F4). The matmul path is already fine. This is the *same* wall sudoingX names.

---

## 3. Mechanism map (inspiration → KORE gap → build) — updated

| Lever | Role | KORE status | Action |
|---|---|---|---|
| Ternary weights | fit | ✅ has (`kore-kernels/q2_0`, `kore-btes`) | keep; add IQ2/IQ3 + auto-select |
| Small-batch ternary mat-vec | speed | ✅ has; tuning in flight | finish tuning; **add planar-transposed layout + flattened work items + batch-invariant reduction** (sudoingX recipe) |
| **MoE sparse activation** | **fit + speed (the trick)** | ❌ | **build** (top-k router, expert tensors, per-expert residency) |
| **Concurrent GPU+CPU expert execution** | **fit (the real engine)** | ⚠️ `layer_sharder` is sequential | **build two-tier concurrent executor** (was under-specified) |
| **Adaptive expert cache** | fit | ⚠️ static LRU-by-layer | **make it learn hot *experts*; add swap/async knobs** |
| Pinned-RAM staging ring + PCIe share | fit | ❌ | **build** (`pcie-frac` equivalent) |
| Router look-ahead prefetch | fit | ⚠️ `prefetcher` is layer-granular | **extend to next-layer expert prediction** |
| Quantized + **streamed** + paged KV | context fit + speed | ⚠️ Q8 red, K8V4 CPU-only, no streaming | finish Q8, add K8V4 GPU + **KV streaming past a threshold** |
| **Batched/chunked prefill (8192)** | speed (real chat) | ❌ (1 token/`forward`) | **build**, with next-layer expert overlap |
| **MTP spec decode + graft tooling** | speed (1.5–1.8×, **lossless**) | ⚠️ CPU only, no graft tool | **port to GPU + add external-head graft** |
| **Prompt/suffix lookup drafting** | speed (code/edits) | ❌ | **build** (cheap, independent) |
| **Batch-invariant kernels** | lossless spec | ❌ | **build + test (1–4 cols bit-identical)** |
| Hardware-agnostic planner + **calibration** | universality | ⚠️ hardcoded | **probe + measure + auto-place** |

---

## 4. Target architecture (four layers)

1. **Architecture registry (incl. MoE).** Dense (qwen35/llama) + MoE (top-k router, expert FFN tensors, per-expert residency, shared expert). Adding an arch stays a module, not a core edit.
2. **Residency / offload planner (hardware-agnostic) + calibration.** Probe (VRAM, RAM, PCIe BW, CPU expert throughput, disk) → placement plan (VRAM hot experts / pinned RAM / mmap'd) with async prefetch and a **PCIe share** knob; expose Strata-style `--calibrate` that *measures* the knobs instead of guessing.
3. **Fast decode kernel layer.** Finish Q8 → add K8V4 GPU + paged KV + **KV streaming**; **batched prefill with next-layer expert overlap**; **MTP + prompt-lookup** on the fast path; **batch-invariant** reductions; PDL/fused dispatch; grouped-MoE kernels.
4. **Backend parity.** wgpu (Vulkan/Metal/DX12) as the agnostic base; CUDA feature path for the kernel-heavy wins (dp4a trit decode, PDL, MMA at ≥8 cols); CPU SIMD fallback. Same model code across all. (KORE already has the wgpu compute base; `dot4I8Packed` + bit ops express the ternary mat-vec.)

---

## 5. Phases (each ends on a gate)

### P0 — Unblock and pin (immediate, hours)
- Fix the 3 `kore-vulkan` build errors; get the Q8 KV path compiling and running.
- Add regression test: GPU `fused_attention_decode_gqa_q8` vs CPU F32, **token-for-token**.
- Add `bench-context` harness: decode at 5/128/1024/4096 + prefill tok/s + correctness gate + **peak VRAM/RAM**.
- **Reclaim VRAM** (display off the card where possible) and record free RAM — see 11.1.
- **Gate:** build green; short-context ≥ 80; 4096-ctx improved vs 1.8 with token parity.

### P1 — Generic + MoE loading ("any HF model" becomes true)
- MoE arch module (top-k router + expert tensors) + loader from GGUF/safetensors.
- Broaden the loader so dense HF models beyond Bonsai load on the fast path.
- **Gate:** a MoE model loads and runs **correctly on CPU**; token parity vs reference.

### P2 — Offload engine (punch above weight class)
- Two-tier executor: GPU cached experts **concurrently** with CPU misses; **pinned-RAM staging ring**; adaptive hot-expert cache; PCIe share; router look-ahead prefetch.
- Corrected tier model: **all experts in pinned RAM** by default; mmap'd `experts.bin` as the low-RAM fallback; SSD n-gram table optional (P3 drafting).
- **Gate:** a model whose weights ≫ 8 GB runs; report tok/s + peak VRAM/RAM + expert hit rate.

### P3 — Fast-path kernels + speculative decode
- Finish Q8; add K8V4 GPU dequant-inline; **paged KV** + **KV streaming** past a threshold; split-K parallel decode; **in-place quantized flash attention (no F16 scratch)**.
- **Batched/chunked prefill** (512 → 8192 tokens/dispatch) with next-layer expert streaming overlapped with attention.
- **Spec decode:** port `transformer.rs` draft/verify to the GPU path; **MTP graft tool** (import external `nextn.*` head + Hadamard-inverse fix); **prompt/suffix lookup drafter**; **batch-invariant kernels**.
- **Gate:** short-context ≥ 80 holds; 4096-ctx into double digits; spec decode **byte-identical greedy**; prefill ≫ current.

### P4 — Hardware-agnostic planner + parity
- Auto-detect → auto-**calibrate** → auto-place → auto-quant; backend parity tests (CPU/Vulkan/CUDA); community "sweep" rows (one row per machine, same as the references).

### P5 — Multi-model serve (only after P2–P4 are real)
- Multi-slot batching (labelled honestly: latency win on small cards, throughput win only when experts fit VRAM / on a split), conversation parking, prompt cache checkpoints.

---

## 6. Milestones ("punches above our weight class" on 8 GB)

- **M1 (dense big):** Ternary Bonsai-2 **27B** + lean MTP graft on the 8 GB Max-Q at **≥ 30–50 tok/s** short-context, double digits at 32–64 K, **lossless**. (Reference: 42.6/52.6 on a 2070 Super.)
- **M2 (MoE beyond VRAM):** a **MoE** model whose weights ≫ 8 GB at usable tok/s via concurrent expert offload. Entry rig: Strata's **Coder** (29.6 GB, 32 GB RAM) before full Qwen3.8-Flash-Next Q2_0.

---

## 7. Speculative decode design (the cheapest big win) — details

- **MTP graft** is a deliverable, not a footnote: a Rust tool that (a) extracts a donor `blk.N.nextn.*` block, (b) merges it into the target with the `block_count`/`nextn_predict_layers` header edits, (c) applies the **Hadamard inverse** to the draft's embedding lookup when the host model's table is in a rotated basis, (d) verifies `merge --strip` reproduces the original bytes.
- **Batch invariance is a first-class kernel requirement.** Every kernel on the draft/verify path must produce the same bits for 1…N rows. Test: same prefix, last N tokens in one batch, next-token logits identical for N = 1..4.
- **Two draft sources:** the grafted MTP head (general) and **prompt/suffix lookup** (cheap, big on code/quoted text). Both are verified by the target → same output.
- **KV rollback:** verify needs cache truncation — `kore-attention/kv_cache.rs::truncate()` already exists; keep it exact across streaming.

---

## 8. Measurement protocol (applies to every claim)

- **Correctness first:** token-for-token vs reference *and* byte-identical across batch shapes before any tok/s claim.
- Report **raw tok/s** *and* the baseline context table side by side; never conflate.
- Pin: GPU + VRAM, CPU + RAM, driver, backend, build profile, PCIe; prompt/context length, max new tokens, KV dtype, weight dtype, batch size, n-max.
- Report **prefill** and **decode** separately; **expert hit rate**; peak VRAM/RAM; and for offload lanes, **blobs/MB from the file tier**.
- Short-context ≥ 80 is the rollback trigger.

---

## 9. Validation rigs (answers a former open question)

- **Dense large:** **Ternary Bonsai-2 27B (PTQ1_0) + lean Qwen3.8-27B MTP graft** — same arch family as the Bonsai-2 2B we already run, ternary, 8 GB-proven, donor recipe published.
- **MoE entry:** Strata's **Coder (IQ1_M, 29.6 GB)** — fits 32 GB RAM, 10-of-… routing.
- **MoE full:** **Qwen3.8-Flash-Next Q2_0 / IQ2_XS** (Strata's model).
- **CUDA path:** defer as a *feature-gated accelerator* over the wgpu base; the hard kernels are CUDA-first in both references, but the wgpu base can express dp4a + bit ops.

---

## 10. Risks / anti-patterns / do-NOT-do (evidence-backed)

- **Scope:** MoE + concurrent offload + streaming KV + batched prefill + spec decode is **weeks**. Sequence by gate; never merge a half-built representation.
- **Do NOT stream weights from SSD per token.** All experts live in RAM; SSD is the *low-RAM fallback* ("much slower") and otherwise holds the n-gram draft table.
- **Do NOT add a helper/second GPU** for this — it measured *slower* than the CPU pool.
- **Do NOT ship a duplicate embedding table** (the "fat" MTP file's +682 MiB collapses an 8 GB card: 37 → 14 tok/s).
- **Do NOT chase tensor cores below ~8 columns** for ternary small-batch — dp4a wins; MMA only pays at ≥8.
- **Do NOT assume the mat-vec kernel is free speed** — measure; on a bandwidth-saturated card it was only +7 %.
- **Do NOT allocate KV naively** — the references show non-monotonic window cliffs ("holes"); size and place KV deliberately, and prefer streaming.
- **Prioritize fitting more experts over a faster GPU** (~700 experts/GB).

---

## 11. Decisions (resolved)

1. **First milestone: M1 (dense 27 B + MTP graft), then M2.**
   Rationale: M1 reuses the ternary path KORE already has and produces a measurable, **lossless** 30–50 tok/s number fastest. M1 is a *fits-barely* case (the lean file is ~6.3 GB, resident in VRAM at 7.9 GB peak) — so M1 exercises P0 + P3 (ternary kernel, Q8/K8V4 KV, batched prefill, MTP, batch invariance) but **does not** exercise the offload engine. The **offload engine (P2) is prototyped on the CPU lane in parallel** with M1 so it is ready the moment M2 starts. The MTP **graft tool is a reusable asset** (it also grafts heads onto MoE models), so M1 spends effort M2 keeps.
   Note the framing: M1 proves the toolchain and gives a headline number; **M2 (MoE) is where "far bigger than VRAM" actually happens** and where the concurrent offload engine earns its keep.
2. **CUDA: later.** wgpu (Vulkan/Metal/DX12) is the agnostic base; CUDA is a feature-gated accelerator added once kernels are proven. `dot4I8Packed` + bit ops express the ternary mat-vec on wgpu.
3. **Spec decode: MTP graft first, prompt/suffix lookup second.** The graft is the bigger structural lever and is shared by both references; the lookup drafter is a cheap, independent add-on slotted in after.

### 11.1 Reclaim GPU and RAM space (do this before/while measuring)

Both references show headroom translates straight into capability:
- **VRAM:** a desktop/display takes ~700 MiB — sudoingX measures 32K context with a desktop vs **48 K on a clean card** from 7,776 MiB free. Reclaiming ~700 MiB is worth a whole context tier (and ~**700 more cached experts per GiB** on Strata's model). Run `bench-context` with the desktop off the GPU where possible.
- **RAM:** all experts are pinned in RAM by default; DIMMs running below rated speed slow the CPU half. Free page cache / close browsers before offload lanes, and record free RAM in every offload result.

---

## 12. Immediate next action

**P5d is landed** (rewind + prompt-lookup spec decode, byte-identical,
1.29-1.50x on prompt-echo workloads; see the progress log). What is left, in
order of value, all of it optional:

1. **`GpuState` snapshot/restore -> prefix caching.** A different feature from
   spec decode: spec decode rewinds *within* a batch, prefix caching rewinds
   *between* calls. It removes the repeated prefill cost of every multi-state
   benchmark and is the piece that makes a shared system prompt cheap.
2. **The m>1 batched attention.** 6 full-attention layers x m per-column calls
   -> 1 call with `seq_q = m` and a per-row causal mask, plus its per-row
   `slice_rows`/`scatter_row`. Structurally bit-identical if the masking and
   sharding line up; ~10% of both prefill and a batched decode.
3. **A chunked parallel GDN scan** (~10% of a chunk, replaces the host-side
   per-row loop inside DeltaNet layers).
4. **Systemic: have `alloc_storage_buffer` return `PooledBuffer`** so recycling
   cannot be forgotten at the ~30 call sites (the m>1 decay was exactly a
   forgotten recycle, already fixed locally).

Everything that reassociates a summation (`gdn_step` threads, `dot_wide` FMA)
still waits on the user's explicit tolerance decision -- not a token-id diff.

> **SUPERSEDED 2026-10-17 -- see §14.** Coverage was chosen over speed, so the
> active next action is **P6.1** (quant-agnostic resident weights in the qwen35
> path); the items above become **P7** and **P8**, run interleaved after P6.
> The reassociation tolerance decision **has** been taken (unblocked, per-lever
> written bound) -- that is **P9**, run last. §14.8 has the ordered first step.

---

## 13. Progress log

### P0 — DONE
- Q8 KV path real and GPU-side (packed-`u32` `flash_decode_gqa_q8` + `kv_cache_append_q8`, `quantize_q8_rows`); quantization happens on the GPU, no CPU round-trip.
- `bench-context` + `bench_gemv` + `bench_chain` harnesses stand up.
- Token parity verified: GPU decode token ids are byte-identical to the F32 CPU reference.

### P0.5 — DONE (GEMV optimised), then CLOSED OUT
Shipped (`ed5a38d`, `983d620`, `b45e4ad`): `unpack2x16float` block scales, LUT
removed, coalesced code layout, 4 output rows/warp. Short-context **33.7 →
59.4 tok/s** over HEAD, byte-identical ids.

### P0.6 — dispatch fusion: SHIPPED, MEASURED NEUTRAL — CLOSED OUT
- `add_rms_norm` folded into every norm site via a carried *pending* residual.
- `hadamard_signed.wgsl` rewritten (double-buffered, 9 unrolled stages, 1
  barrier/stage, no register array).
- **`had_pre.wgsl`** — 4-mode fused pre-op + Hadamard (`add+rms+had`,
  `rms+had`, `swiglu+had`, `sigmoid_mul+had`). Replaces 79 of ~310 dispatches
  per token. Bit-identical (asserted exactly, not epsilon) against the
  standalone kernels.
- **Result: 58.20 → 58.16 tok/s median (6 rounds, rotating interleaved
  harness).** Neutral.

**The corrected mechanism:** the per-dispatch *launch* cost on this path is
~0 — the decode loop encodes 48 dispatches per command buffer and the GPU
never drains between them. What `bench_chain` measures as ~11 µs per dependent
dispatch is the kernel's **execution latency**, which fusion does not remove.
So the decode loop is bound by the serial execution latency of its kernels, not
by dispatch count. Fusion is kept (bit-identical, −25% CPU encoding and VRAM
churn, matters when the host is the bottleneck, e.g. MoE offload) but must not
be budgeted as a tok/s lever.

### P0.7 — Q2_0 GEMV re-diagnosis — CLOSED OUT
`bench_gemv`: Q2_0 pinned at 56–62 GB/s on *every* shape, Q4_K at 130–160 GB/s
on the same shapes. But bytes/s is the wrong metric: Q2_0 does ~206 Gweights/s
vs Q4_K's ~231. Diagnostics (patch → time → revert):

- removing the code load + extraction: 2.34 → 2.29 ms → **not** unpack-ALU-bound
- removing the k-loop `x` load: 2.34 → 2.51 ms → **not** x-traffic-bound
- removing *all* k-loop global loads: 2.34 → 1.79 ms → **76% is instruction latency**
- removing the loop `break` (to allow unrolling): → 2.66 ms → worse (register pressure)

Verdict: latency/occupancy-bound on its own dependency chain; ≤12% prize for a
deep rewrite. Stays closed.

**Re-checked (2026-10-17) after the attention win — the GB/s gap is a mirage.**
Fresh `bench_gemv` on identical shapes and identical grids:

| shape | n | k | Q2_0 GB/s | Q2_0 Gweights/s | Q4_K GB/s | Q4_K Gweights/s |
|---|---|---|---|---|---|---|
| emb_head | 248320 | 2048 | 54 | 193 | 135 | 240 |
| f32 streaming | — | — | 280-315 | — | — | — |

Q2_0 reads 0.28 bytes/weight against Q4_K's 0.56, so *at equal weight
throughput* a 2x lower GB/s is expected. In weights/s the two formats are within
~20%. Q2_0 is not 2.5x off; it is ~1.2x off, and the gap is the unpack work
(4 weights/byte, ~6 ops/weight vs Q4_K's larger 256-element block amortizing
loop overhead over 4x more weights per iteration).

The split-K hypothesis (too few workgroups, as with attention) is **refuted for
the shapes that matter**: `emb_head` launches ceil(248320/16) = 15520 workgroups
= 62080 warps, far more than the ~1150 the GPU can hold. Ranking the shapes by
weights/s shows a parallelism gradient only among the *tiny* shapes
(attn_out n=2048: 105, qkv n=6144: 157, emb_head: 193), and those carry a few MB
each. So there is real headroom but it is in the **inner loop** (fewer
instructions per weight: a `dot()`-based code unpack instead of 4 scalar
multiplies, and/or a wider per-lane tile), worth ~20% on the token, not 2.5x.

### P1a — long-context decode: SPLIT-KV SHARDING — DONE, 2.18x at ctx 4096
The P1 diagnosis ("the contiguous KV tensor / the O(seq_k) read *volume* is the
binding constraint") was **wrong**. The KV representation was already fine (Q8).
The binding constraint was **how few workgroups the decode kernel launched**:
`fused_attention_decode_gqa_q8` has a grid of `(num_heads x seq_q)` =
**(8 x 1) workgroups of 64 threads for the whole GPU**, each walking the entire
window. It is memory-*latency*-bound, not bandwidth-bound — evidenced by the F32
kernel reading 4x the bytes and being only 1.9x slower.

Fix: `flash_decode_gqa_q8_split.wgsl` (grid `(num_heads, n_splits, 1)`, shard's
own running max) + `flash_decode_gqa_q8_combine.wgsl` (log-sum-exp merge, one
workgroup/head). `n_splits = ceil(seq_k/64)` capped at 64; below 256 keys the
single-group kernel is kept. `KORE_ATTN_SPLIT_MIN` overrides the threshold.
Sharding is an optimisation only — `attn_can_shard()` falls back to the
single-group kernel for any shape the shard kernel cannot stage.

**Kernel level** (`bench_attn`: every arm warmed up and then interleaved in the
*same* loop with the arm order rotated per repetition; medians of 21):

| seq_k | single ms | best split ms | n_splits | speedup | GB/s | max abs diff |
|---|---|---|---|---|---|---|
| 256 | 0.338 | 0.123 | 16 | 2.7x | 2.1 | 3.7e-8 |
| 512 | 0.606 | 0.132 | 32 | 4.6x | 4.0 | 4.5e-8 |
| 1024 | 1.117 | 0.151 | 32 | 7.4x | 6.9 | 4.1e-8 |
| 2048 | 1.232 | 0.139 | 32 | 8.8x | 15.0 | 3.5e-8 |
| 4096 | 2.778 | 0.192 | 64 | 14.5x | 21.9 | 3.3e-8 |
| 8192 | 6.369 | 0.302 | 64 | 21.1x | 27.8 | 3.5e-8 |

**End to end** (`bench_context_cool`: prefill → cool the die to 68 C → 8 untimed
warmup forwards → 40-token timed window; medians; AC / 80 W; Bonsai-2 Q2_0):

| decode ctx | 13 | 1032 | 2056 | 4104 | 8200 |
|---|---|---|---|---|---|
| sharded ms | 14.80 | 15.05 | 15.11 | 15.34 | 16.17 |
| sharded tok/s | 67.6 | 66.4 | 66.2 | 65.2 | 61.9 |
| unsharded ms | — | 19.29 | — | 33.48 | 52.63 |
| unsharded tok/s | — | 51.8 | — | 29.9 | 19.0 |
| sharding speedup | — | 1.28x | — | **2.18x** | **3.26x** |

Prefill through the same harness: Bonsai prefills one token at a time, so prompt
throughput *is* decode throughput, and it gains the same way:

| prefill ctx | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| sharded tok/s | 64.2 | 63.0 | 57.8 | 40.4 |
| unsharded tok/s | 58.0 | — | 33.1 | 12.8 |
| sharding speedup | 1.11x | — | 1.75x | **3.15x** |

The long-context curve is now flat: 14.8 → 16.2 ms (+9%) from ctx 13 to 8200,
against 33.5 ms for the same binary with sharding disabled. **Token ids are
byte-identical** between the sharded and unsharded arms and across contexts; the
merge perturbs softmax by ~3e-8, which does not move argmax.

Corrections this forces on the plan:
- **Paged KV is a memory-management feature, not a tok/s lever.** Do not budget
  it as a decode-speed win; F7's "make the per-token read O(1) per block" was
  never the fix.
- The remaining lever on this kernel is the **read pattern** (8 useful bytes per
  cache line per instruction) — up to ~6x more (28 → >100 GB/s) and the thing
  that would matter at 32K-262K context. That is a KV *layout* change, not a
  paging change.
- **Measurement lesson, and it invalidated two earlier A/B attempts.** Decode was
  being sampled *immediately after* the prefill, so a ctx-5 arm ran on a cold die
  and a ctx-8192 arm on a heat-soaked one: `bench_context` conflated *context*
  with *heat*. A cooled die also idles at 300 MHz, so the first ~5 forwards
  measure the clock ramp (81 ms/token for the first four vs 15.5 for the last
  four), not the model. Long-context numbers must be cool-die + clock-ramped:
  `crates/kore-cli/examples/bench_context_cool.rs`. On heat alone the old hot
  ctx-4104 reading was 1.22x the cool unsharded one (40.8 vs 33.5 ms), and the
  old "2.08x" was a mix of the real sharding win and that heat.

### REPRIORITISED (this supersedes decision 11.1)
1. **M2 (MoE expert offload) ahead of M1.** A 27B dense ternary model is ~7.6 GB
   of weights and cannot fit 8 GB with any KV cache — **M1 is infeasible on this
   hardware**. A MoE with 2–3 B active params runs 40–60 tok/s here with ~30 B
   of weights streaming from RAM. M2 is the only shape that delivers "far bigger
   than VRAM". **Cannot be built or validated on this machine** (15 GB RAM, no
   model > 1.5 GB on disk) — leave it for the bigger box.
2. **P1a (split-KV decode sharding) DONE** — see above; this was the actual
   4096-context collapse, it is fixed on this hardware, and it is worth 1.28x
   (ctx 1k) to 3.26x (ctx 8k) end to end.
3. **P2 (offload engine)** — prerequisite for offloaded-KV MoE; untestable here.
4. **M1** only if a leaner representation lands.

### P1b — where the remaining decode time goes (measured, 2026-10-17)
Cold-die decode at ctx 13 is **14.55 ms** (67.4 tok/s). Attribution:

- **~10-11 ms: the Q2_0 GEMVs** (the model's 1.67 G weights at the rates
  `bench_gemv` reports: layer shapes ~157 Gweights/s, lm_head 509 M weights at
  193 Gweights/s = 2.64 ms, which reproduces the observed total).
- **~0.1 ms: attention** at this context, ~0.5 ms at ctx 4104.
- **~3.5-4 ms: everything else** — norms, GDN/delta scans, `had_pre`, KV
  quantize+append, argmax — i.e. ~24 layers x ~6-8 small dispatches, and per
  P0.6 that time is per-dispatch *execution* latency, not launch overhead.

So the GEMV is ~70% of the token and is the only remaining large, measurable
lever on this hardware. See the corrected P0.7 note above for its ceiling.

### Measurement caveat to report to the user
On battery this laptop pins the GPU to a **30 W** limit (default 80 W) and tok/s
drifts 57 → 21 within one session. All micro-optimisation claims must be
re-measured on AC at 80 W; only interleaved order-rotating A/B is valid on
battery. The fusion number above was taken while charging at 80 W.


### P1c — the small-op lever is HOST-side, and it is fixed (measured, 2026-10-17)

The census (`KORE_OP_COUNT=1`, `crates/kore-cli/examples/op_counts.rs`) gave
**345 dispatches/step** at ctx 10 (151 `gemv_q2_0`, 79 `had_pre`, 54 `gdn_*`,
30 KV-quantize/append, 18 `hadamard_signed`, 6 `qknorm_rope`, 6 attention, 1
argmax). The surprise was not the dispatch count but that a step drew **529
storage buffers per step, 100% of them fresh `create_buffer`**: `Drop` can only
park a buffer (the encoder under construction still references it) and the pool
was drained only in `sync()`.

`bench_chain` was extended with the fused `had_pre` modes and shows the cost is
per *output buffer*: at h=2048, `indep` (enqueue) is 4.3 us for a 1-output op,
7.5 for 2 outputs, 12-13 for 3 -- ~3 us each. Reclaiming the parked list from
`Queue::on_submitted_work_done` (`fbcbe71`) took it to 0 `create_buffer`/step;
`had_pre` m0 `indep` fell 13.07 -> 6.73 us and `dep` 25.03 -> 20.03 us. End to
end, an interleaved 9-round order-rotating cool-die A/B gave 109.83 -> 112.75
tok/s, every paired round favouring the pooled arm, tokens byte-identical.

### P1d — refuted: `had_pre` is not barrier-bound (do not retry this way)

`had_pre` (79/step) is the largest non-GEMV cost: 20.0 us `dep` at h=2048 and
31.2 us at h=6144, growing with h because modes 0/1 recompute the **whole-row**
sum of squares in every 512-block workgroup. Collapsing its 8-barrier 256->1
tree to 2 barriers by running the tree serially in one thread (same pairing and
order, hence bit-identical) made it *much worse*: 25.0 -> 42.3 us at h=2048,
36.8 -> 52.7 at h=6144. 255 dependent shared-memory reads cost more than the six
drained barriers they save. The h=1,2 butterfly stages pair only a thread's own
four elements, so they *can* be done in registers (bit-exact, untaken), but they
were worth ~1 us in the bundled run -- inside noise. Barriers are cheap.

Consequence: the remaining cost is the redundant whole-row read itself, and the
only fixes that remove it (a separate per-row reduction dispatch, or per-block
partial sums combined with atomics) change the summation order, which forfeits
the byte-identical-token gate. Needs a numeric-tolerance gate first.

### P3a — MoE expert offload: the engine exists and is byte-exact (2026-10-17)

Committed `5a4c936` (engine + tests) and `2ad5581` (host-path fixes + bench).

The capability goal is "run a model whose weights exceed VRAM". Dense models
have no slack: M1 (27B ternary, ~7.6 GB of weights) does not fit in 8 GB. An MoE
does, because its experts are a big *sparsely-used* bank -- the model is huge,
the per-token working set is not.

**Policy chosen: promote-to-VRAM** (upload the expert on a miss, evict LRU,
compute with the *same* kernel on the *same* bytes). That keeps the doctrine:
residency changes where the weights live, never the arithmetic, so a starved
layer is bit-identical to a fully-resident one -- asserted with `assert_eq!` on
the f32 output, not a tolerance (`resident_and_starved_runs_are_bit_identical`).
The alternative -- compute a miss on the host CPU and overlap it with the GPU
(Strata's design) -- is cheaper in transfer but the CPU twin sums a row in a
different order, i.e. a different answer, so it is documented as an option and
deliberately not wired in. The host twin *is* used, as an independent numerical
reference for the device path.

Files: `crates/kore-nn/src/moe.rs`; `DeviceBackend::upload_u8` +
`KoreVulkanBuffer::from_slice`; `crates/kore-cli/examples/bench_moe.rs`.

**Cost curve** (`bench_moe`, 4 layers x 32 experts, top-2, hidden 1024 / ffn
2048, 216 MiB bank, 256 steps, uniform routing): resident 2/4/8/16/32 of 32 gives
314.8 / 338.3 / 378.3 / 496.3 / 1617.5 tok/s at 4.5 / 11.9 / 24.5 / 49.6 / 100 %
hit and 12.89 / 11.89 / 10.19 / 6.80 / 0.00 MiB streamed per token. The tax is
the streaming and it tracks the miss rate, not the bank size. Zipf-ish routing
(the realistic case) is uniformly better at the same capacity.

**Two host costs sat on the miss path; both removed.** (1) `to_device` repacked
GGUF->GPU layout on *every* promotion -- a per-byte host permutation paid per
miss; now repacked once and retained. (2) The upload materialised an owned `Vec`
via `Storage::from_bytes` because `Tensor::to(device)` needed something to own,
costing a full-size copy *and* a fresh allocation; `queue.write_buffer` takes a
slice, so `upload_u8` now writes straight from the borrowed bytes (default trait
impl preserves the old two-step for other backends).

Paired, interleaved A/B (256 steps, 8 runs/config, every round favouring the new
path): resident 2 -> 198.0/317.2 (1.60x), 4 -> 214.2/337.8 (1.58x), 8 ->
247.0/377.6 (1.53x), 16 -> 338.3/494.0 (1.46x). All-resident is unchanged --
the check that the win is where it is claimed to be. Noise floor on these short
runs is ~20%, measured by A/B-ing two builds differing only in a redundant
clone, so only within-run paired comparisons count.

**Blocked.** No MoE checkpoint exists on this machine and `gguf_min.rs` has no
writer, so a synthetic-MoE GGUF cannot be built either. The kernel path is
exercised for real; the *model* path is not. Next, in order: GGUF MoE reading
(`expert_count`, `ffn_*_exps.weight` stacked `ne=[n_embd,n_ff,n_expert]`),
shared experts, a device-side router (the current one costs one small readback +
sync per MoE layer), and prepack-on-disk to drop the retained host copy.

### P3b — the MoE model half: a GGUF loader (2026-10-17)

Committed `69808ec`. `MoeConfig::from_gguf` reads the routing keys and
`MoeLayer::from_gguf` reads the three stacked `ffn_{gate,up,down}_exps.weight`
tensors plus the `ffn_gate_inp.weight` router, and cuts each bank per expert.

The layout is the reason this is short: an `*_exps` tensor is
`ne = [n_embd, n_ff, n_expert]` and `ne[0]` is fastest-varying, so the expert
index is the *outermost* dimension and one expert is a contiguous slab -- no
gather, no index math beyond a multiply. `from_stacked` is the whole of the
byte-layout logic (it does not touch a file), so the test asserts the cut
directly: four concatenated matrices must come back out as the four experts byte
for byte, and a layer built from the bank must be indistinguishable on the
device from one built by hand from the pieces -- `assert_eq!`, not a tolerance.

Design choices worth keeping: absent `expert_count` is `Ok(None)` (a dense
layer), which is what lets one loader walk a mixed dense/MoE model without
knowing the arch in advance; `expert_weights_norm` defaults to **off**, matching
upstream, because defaulting it on silently rescales a router that does not
renormalise; and every dim/length is checked against `expert_count` rather than
trusted.

A finding from writing the tests: a bank carries no shape. Gate is
[hidden, expert_ffn] and down is [expert_ffn, hidden], and both are `m*n/64`
blocks -- identical lengths for any expert, square or not. So length checking
guards against truncation and *cannot* catch a transposition; which tensor is
which is a naming question and nothing else. The test says so out loud rather
than pretending the check is stronger than it is.

Testing is split along the same lines as the honesty: `from_meta` takes the key
lookups as closures, so the reading rules (required keys, defaults, partial and
out-of-range sets) are tested without a file; `from_stacked` is tested on
synthetic banks; the "dense" path is tested against the **real Bonsai
checkpoint**, using the model's own architecture prefix, so it is a claim about
that model. What is *not* tested is the combination -- no MoE checkpoint, so no
stacked bank has come out of a real file and the router has not met a real
activation. The per-layer seam (ask `from_gguf` instead of building a dense FFN)
is deliberately unwired rather than written blind.

Still open for this lever: shared experts (resident, never evicted) and a router
bias; the device-side router (one small readback + sync per MoE layer today);
and prepack-on-disk so the bank already *is* the GPU layout.

### P3c — the spec-decode payoff, measured before it was built (2026-10-17)

The user chose prompt-lookup spec decode as the next lever. The whole case for
it is that a `k+1`-column verify costs about what one column costs, because
`gemv_q2_0`'s workgroup count is `(n/16, ceil(m/4))` -- identical at `m=1` and
`m=4`, so the weight read is shared. That assumption was never measured, and it
is wrong.

`bench_spec` (`6483abf`) walks the real GEMV op list of one Bonsai-2 decode step
-- 151 ops, 504 MiB of packed Q2_0, shapes read out of the checkpoint -- and
times it interleaved, median of 8:

    batch   step ms   per-token   vs m=1   eff GB/s
        1     4.619       4.619     1.00x      114.5
        2     7.421       3.710     1.245x      71.3
        4    12.975       3.244     1.424x      40.8

A first, *swept* version of this bench said `m=2` was free (0.99x) -- an artifact
of a cold first `m=1` (7.64 ms) against a warm `m=2`; the closing `m=1` came in
at 4.84 ms. Interleaving and taking medians collapsed it to the table above,
where min ~= median. Lesson restated: on this box, a sequential sweep measures
the die's clock, not the thing under test.

114 GB/s effective against a ~384 GB/s part says the kernel is not stream-bound,
so the read is not what batching buys. Each extra row costs ~0.6 of a fresh one,
which fits the two-term model the shader header already recorded (58% of the
`m=1` cost on the load side; 12 loads per 64 weights of which 4 scale with `m`):
`0.58*(8+4m)/12 + 0.42*m` = 2.84x at `m=4` vs 2.81x measured. So the GEMV half
of a 3-draft verify is worth ~1.4x per token, not ~4x.

The useful part is *which* term that is. `0.42*m` is per-column arithmetic, and
it is not irreducible: `dot_wide` recomputes `byte_deltas(ww, j)` once per
column, though for a given row the code word -- and its deltas -- are identical
across columns. Hoisting it out is common-subexpression elimination: same
arithmetic, same summation order, so bit-identical, *including at m=1 where
nothing changes*. 64 delta computations per k-iteration at `m=4` become 16,
worth roughly 1.4x -> 3x per token. Not attempted here: it changes nothing that
ships today, and preserving the summation order needs 16 live partial `vec4`s,
a real register-pressure risk in an already-hand-tuned kernel. Do it when a
verify pass exists to measure against.

Conclusion for the lever: prompt-lookup spec decode is blocked on a
**multi-column forward**, which is the same machinery as chunked prefill --
multi-row `had_pre`/`hadamard_signed`, per-row positions for `qknorm_rope`,
`m` queries in `flash_decode_gqa_q8`, an `m`-step loop in the GDN ops, `m` rows
for KV append/quantise/argmax, and `GpuState` snapshot+restore for rejecting a
draft. Build order: `m=2` bit-identical to two `m=1` steps over the *whole*
forward first, then `m=4`, then the drafter and acceptance loop, then the
kernel optimisations -- with `bench_spec` as the instrument, since it measures a
shader change with no forward at all.

## P4a -- pricing the dispatch groups (done, `b262115` + `d13035b`)

This adapter exposes no GPU timestamps, so the next instrument was an ablation:
`KORE_SKIP_OPS=label[,label...]` drops a group in `record_dispatch` before it
costs anything, then the step is re-measured. The chain keeps every other shape
and dependency, on stale data -- a timing probe, never a correctness one. `all`
skips everything and so prices the host side outright.

At ctx ~200, 200 steps, interleaved, against same-round baselines of 8.45/8.82:

    all                   0.26 ms   <- 345 dispatches, 529 draws, 7 submissions
    (drop gemv_q2_0)      3.53      => GEMV  ~5.0 ms   59%
    (drop had_pre)        6.84      => had_pre ~1.6   19%
    (drop gdn_step)       7.64      => gdn_step ~0.85  10%
    (drop anything else)  8.36-8.78 => noise

That kills the host-bound reading (0.26 ms for the entire submission path) and
replaces "194 dispatches at ~22 us each", which was arithmetic on a subtraction,
with two kernels worth naming. Several groups show *negative* deltas -- removing
an op can shorten a neighbour's stall -- so only deltas above ~0.3 ms count.

`bench_spec` gained the per-op half, grouped by tensor role and sorted:

    token_embd   1.040 ms  n=248320  137 GB/s
    ffn_gate     0.754     (x24)     113
    ffn_up       0.749     (x24)
    ffn_down     0.651     (x24)
    attn_qkv     0.565     (x18)     113
    attn_gate    0.283     (x18)      75
    ssm_out      0.269     (x18)      79
    attn_q        0.141     (x6)     100
    attn_output  0.091     (x6)       78
    attn_k/v     0.054 ea  (x6)       33   n=512 -> 32 wg

The narrow ops are 3.5x slower per byte -- the same too-few-workgroups disease
split-KV cured -- but they are 1% of the step, so that is a curiosity. The wide
ops sit at a uniform 100-137 GB/s *on weight bytes*; with x re-read once per 16
rows the real figure is ~1.9x that, i.e. ~54% of a ~384 GB/s part. Uniform
across n=2048..248320, so the ceiling is the load stream, not occupancy.

Both big non-GEMV kernels need a summation restructured, so both land on the
same numeric-tolerance gate: `had_pre` recomputes its row sum in every
512-block workgroup and wants more rows per dispatch (the multi-column
forward); `gdn_step` is 16x128 = 64 warps total, ~2 per SM, and wants more
threads inside each column's i = 0..127 walk.

**Refuted, and worth the A/B it cost.** `gdn_step`'s `state[j*dv + i]` puts a
warp's 32 threads 512 bytes apart -- 32 lines per load instruction. Transposing
to `state[i*dv + j]` is bit-identical (state is zeroed at allocation, bound to
nothing else, `bonsai` token ids unchanged) and **4.1% slower**: 9.38 vs 9.01
ms/step, median of 5 alternating rounds of two saved binaries. Each 128-byte
line serves 32 consecutive iterations of one thread in the row-major form; the
transpose trades that reuse for a fresh line every instruction. Recorded so it
is not re-attempted.

## P4b -- the had_pre dedup, measured before built (refuted, `a85de61`)

The user chose the bit-identical dedup of `had_pre`'s row-wide sum of squares.
Before writing it, the *ceiling* was measured: skew the shader to skip modes
0/1's sum entirely -- its whole-row reads and its 8-barrier tree -- build, save
the binary, revert. Five alternating rounds against HEAD:

    HEAD          8.79  8.81  8.84  8.91  8.97   median 8.84
    sum skipped   8.42  8.47  8.58  8.63  8.71   median 8.58
    paired delta  0.37  0.34  0.26  0.28  0.26   -> 0.28 ms, 3.2%

Ceiling, and it is 5x smaller than the 1.6 ms `had_pre` costs. A real dedup
must *add* a pass (dispatch + its exposed latency) to collect it, so the net is
under 3%. Not built. The redundancy is *parallel* work on workgroups that are
idle anyway, so it costs occupancy rather than latency -- the kernel header's
original reasoning, now measured.

had_pre's 1.6 ms is the pre-op, the 9-stage Hadamard barrier chain, and a grid
of 4 workgroups (h=2048) / 12 (h=6144) against 36 SMs. Rows per dispatch is the
fix => multi-column forward. gdn_step (10-17%) is the only remaining small-op
item that would restructure a summation, so it alone keeps the tolerance gate.

Lesson, second time this session: measure the share before building the fix.

## P5 -- multi-column forward (started, `7311d99` + `cf0462b`)

User redirected to the m=2 forward after the had_pre route closed. Surveyed the
ops **before** building, which shrank the job:

Already capable (no work):
- `flash_decode_gqa_q8` -- grid `(num_heads, seq_q, 1)`, `max_t = (seq_k - seq_q)
  + q_row + 1` under is_causal = exactly an `m`-query causal chunk.
- `had_pre` -- grid `rows x blocks`, `row = gid / blocks`, `base = row * cols`,
  and modes 0/1 sum **per row**.
- `hadamard_signed` (flat block grid), `quantize_q8_rows`, `argmax_last_dim`
  (rows from `shape[0]`).

Missing:
- `qknorm_rope` per-row `pos`   <-- DONE in P5a
- `kv_cache_append_q8` row offset
- `gdn_gates` m-row variant
- `GpuState` snapshot/restore
Cannot batch: `gdn_conv`/`gdn_step` (sequential scan) => host-side m-loop,
bit-identical by construction, dispatch count unchanged. GDN is 28% of a step
and does not amortise, so m=2's ceiling exceeds the GEMV half's 1.24x but not by
the isolated 1.4x-at-m=4.

### P5a -- qknorm_rope rows (done, `cf0462b`)

Row dimension via `wg_id.y`; every access offset by its row stride; RoPE angle
uses `pos + row`. `rows` derived from the input shape in the dispatch, so all
existing `[1, ...]` callers are unchanged and the whole thing is a no-op at
rows=1 (all offsets `row*stride == 0`).

Verified: new `test_vulkan_qknorm_rope_per_row_positions_bit_identical` (rows=3
at pos0 == three rows=1 calls at pos0..pos0+2, `assert_eq!` on raw f32, not
epsilon) -> 18/18 GPU tests; `bonsai` token ids unchanged; interleaved A/B of
the step 9.04/8.99/9.10 before vs 8.95/8.98/9.10 after (no regression).

Next: `kv_cache_append_q8` row offset, then `forward_batch` around a sequential
GDN loop, then `m=4`.

### P5b -- `forward_batch` LANDED, bit-identical, and the context decay was an allocation bug (done, `bb00ce2`)

`Qwen35Gpu::forward_batch(cpu, tokens, pos0, states)` advances `m = tokens.len()`
positions and returns one greedy token per column; `m > 4` is rejected rather
than silently degraded (the ternary GEMV holds four columns per accumulator).

Batching boundary as designed: batch `had_pre(0)` / `gemv(qkv)` / `gemv(z)` ->
host-side per-row loop (`gdn_gates`, `gdn_conv`, `gdn_step`, `scatter_row` into
an `[m, ssm_inner]` staging `o`) -> batch `had_fwd` / `gemv(out)` / `ffn`. No
change was needed to `kv_cache_append_q8` (its append is a contiguous range copy
on both the packed INT8 K/V and the f32 scales) or to `hadamard_signed` (which
needed a rank fix first: it returned `[numel]` unconditionally, folding away `m`).

`verify_m` gate, extended past `ATTN_SPLIT_MIN_SEQ_K` so the sharded path is
actually exercised (325 positions, not 37): **m = 1, 2, 4 all 325/325
bit-identical to sequential decode, state probe agreeing**; token ids
byte-identical to the P5a baseline; `m = 1` proves `forward_batch == forward`.

**The decay, and the two causes.** Cool-die interleaved same-context A/B
(`bench_multi_cool`, the new harness -- `bench_multi` builds one state from ctx 0
so it measures a heat-saturated die and is only valid for short-context ratios):

| ctx | m=2 ratio | m=4 ratio |
|---|---|---|
| 320 | 1.30x | 1.63x |
| 1024 | 1.01x | 1.42x |
| 2048 | **0.80x** | 1.23x |

1. `m > 1` could not use the split-KV kernel (`seq_q == 1` only), so attention
   ran the latency-bound single-group kernel for the whole batch. Fixed by an
   explicit `seq_k` on both kernels that *narrows* the window, called once per
   column with that column's own `len - m + r + 1` and the shard count `forward`
   would have used there -- bit-identical by construction, and it reads only to
   each row's own `max_t`.
2. That alone changed nothing. A **call-level ablation** (skip the attention call
   and re-measure) priced in-situ attention at 9.9 ms/step for 12 calls
   (0.83 ms/call) at m=2/ctx2048, against 0.146 ms/call isolated and 0.095
   ms/call for the same kernel at m=1. **The split kernel's `ml`/`acc` scratch
   buffers were allocated via `alloc_storage_buffer`, bound directly, and never
   wrapped in a tensor -- so only `KoreVulkanBuffer::Drop` ever recycled them,
   and they were destroyed and re-created on every call** (270 KB at ctx 2048).
   A per-call allocation is invisible at m=1 and multiplied by m at m>1.
   `PooledBuffer` gives scratch the same recycling as everything else.

After: **m=2 1.37x and m=4 1.62-1.66x, flat from ctx 320 to ctx 2048.**

**Recorded negatives / method notes:**
- `KORE_SKIP_OPS` cannot see allocation costs -- it skips dispatches while still
  allocating, so it would have reported attention as nearly free. Call-level
  ablation is the instrument for that class of cost.
- A diagnostic `KORE_ABLATE_ATTN` flag was added to find this and then **removed
  before commit**: a library knob that silently emits wrong tokens is a footgun,
  and `KORE_SKIP_OPS` is the house tool.
- `bench_multi`'s long-context numbers are invalid (41.1 ms/token at ctx 1024 is
  a saturated die, not a harness bug).

Next: `GpuState` snapshot/restore (prefix caching; also removes the repeated
prefill cost of multi-state benches), then the n-gram drafter with HyperQwen's
adaptive trigger, then the chunked parallel GDN scan.

### P5c -- chunked prefill LANDED at 2.13x (done, `f28a580`)

`forward_batch` was in-tree but unreachable: a single stream has one token to give
at a time, so the multi-column pass bought nothing a user could see. Prefill is
the consumer that needs no draft to be accepted -- `n` consecutive positions of
one sequence, which is exactly the causal chunk the pass was built for.

`Qwen35Gpu::prefill(prompt, pos0, states) -> u32` consumes the prompt in chunks
of the GEMV's four-column width (`PREFILL_CHUNK = 4`) and returns the token that
follows it. The shared core `forward_pass(..., Head)` gained a head mode:

- `Head::Skip` -- state only, no token (every chunk but the last);
- `Head::Last` -- project the final column only, via `ht.slice_rows(m-1, 1)`;
- `Head::Every` -- what `forward_batch` still passes.

Safe because the head neither feeds the recurrent state nor couples rows.
`generate_greedy`, `bonsai`, `bench_multi_cool`, `bench_context_cool` all prefill
through it now.

**Gate.** `verify_m` grew a prefill leg (returned token + probe vs sequential):
m = 1, 2, 4 all 325/325 bit-identical, state probe agrees, prefill agrees.
`bonsai` token ids byte-identical to the pre-P5c baseline. `bench_prefill` (new)
is the timing gate: interleaved ABBA, fresh state per arm, both arms probed at
the same position and required to agree, ctx 512, median of 6, order flipped per
round: **104.4 -> 222.8 tok/s = 2.13x**.

2.13x > the 1.62x `bench_multi_cool` reports for decode at m=4, and correctly so:
a verify pass runs the head for all four columns, while prefill runs it once per
chunk (the head is ~a fifth of a step's weight stream on Bonsai-2).

Remaining here: the three GDN ops still run the host-side per-row loop inside a
chunk, so they cost exactly 4x the m=1 figure; a chunked parallel scan would
collapse that to ~1x -- roughly 10% of a chunk, not another 1.6x.

**Incident:** the box's root filesystem hit 100% during this work. Removed
`target/debug/incremental` (8.5 GB) and `target/debug/examples` (2.7 GB); no
git-tracked file touched.

Next: `GpuState` snapshot/restore, then the n-gram drafter with HyperQwen's
adaptive trigger (the only consumer of the batch left, and the one that needs
rollback), then the chunked parallel GDN scan.

### P5d -- rewind + prompt-lookup spec decode LANDED, byte-identical, 1.29-1.50x (done, `acf34f9` + `584b471`)

**The pre-build measurement was right, and it was the thing that made this
shippable.** P3c replayed an n-gram drafter over real greedy decodes before any
rollback machinery existed. It said: `k=3, min_match=4` is the optimum (1.31x
projected on a coherent story continuation, 94% of drafts accepted, 29% of steps
fired), and -- the load-bearing negative -- firing on *any* match is **worse than
not drafting** (1.11x at `n>=1`, and 0.91x on an earlier sweep). The trigger is
the product, not the matcher.

**Rewind (`acf34f9`).** A verify batch advances m positions and only learns at
the end how many of its tokens the model would have chosen. Cutting it short
needs two mechanisms because the two states are different:

- KV cache: an integer. Append-only, every read inside `len`, so the stale bytes
  above the restored `len` are overwritten by the next append. No copy.
- DeltaNet state: **cannot** be rewound -- `gdn_step` updates `state` in place,
  and a recurrent state that has consumed a token cannot un-consume it. Copied.
  18 layers x 3 slots x (1.0 MB state + 72 KB hist) ~= 57 MB, filled by 2
  `copy_buffer` calls per column per DeltaNet layer.

`Rewind` captures rows `0..m-2` (the state after all `m` columns *is* the live
one, so it is not duplicated); `rewind(keep)` with `keep == cols` is a no-op;
`keep == 1` undoes the whole batch. `forward_verify` is `forward_batch` plus the
capture, so the arithmetic is bit-identical. New primitive `copy_buffer`: the
trait's first device->device memcpy, built on the same elementwise `copy_offset`
dispatch (`op_type 8`, both offsets 0) that `scatter_row` used.

**Gate.** `verify_m` grew a rewind leg: for every `keep in 1..=cols`, and for
`cols in {m, m-1}` (so the arena is filled by two passes of *different* widths
back to back), a probe step from the truncated state must agree with an
independently built sequential state. Green at m = 1, 2, 4 (7 keep-points at
m=4, 21 positions bit-identical to sequential, state probe agrees, prefill
agrees).

**Drafter (`584b471`).** `crates/kore-nn/src/spec.rs`: `DraftPolicy`
{k, min_match, max_match, search_back}, `longest_match`, `SpecStats`,
`Qwen35Gpu::generate_spec`. The text is its own drafter -- no weights, no second
model, no tree, no beam. Greedy acceptance is exact, so a wrong draft can only
waste a pass, never change the output.

`longest_match` is one reverse pass over candidate match *ends* with a
one-comparison fast path (most ends are not `t_next`, so the extension loop is
rarely entered). It is deliberately **the same function** `draft_headroom`
replays, so the projected and measured acceptance are the same code:
replay 28.8% fire / 94.2% acc / 1.31x vs measured 29.3% / 94.3% / 1.288x on the
same prompt. `search_back = 16384` bounds the scan's worst case.

**End-to-end (`bench_draft`, new).** Every policy is an arm of ONE interleaved,
order-rotating run; every arm's token stream must equal plain greedy element for
element before any timing is printed. All arms identical, 192-256 tokens:

| workload | k=3 n>=4 | k=3 n>=8 | k=2 n>=4 | k=1 n>=4 |
|---|---|---|---|---|
| story continuation | **1.288x** | 1.280x | 1.251x | 1.179x |
| code continuation | **1.495x** | | | |
| paragraph rewrite | 1.482x | 1.451x | 1.398x | |
| list continuation | 1.342x | | | |

`P(j) = 2/0/1/38` at k=3 n>=4: of 41 fired steps, 38 accepted all three drafts.
9 host-side unit tests in `spec.rs` (longest wins / ties to the most recent /
`k` caps the draft but not the reported length / match at the very end drafts
nothing / `search_back` bounds the scan / `min_match` gating / k clamp).

**Honest envelope.**

- No 4-gram match -> a plain step, ratio exactly **1.00x**, overhead is a
  microseconds host scan. (The Q&A arm that would have shown this emitted EOS
  immediately -- 0 tokens -- so it is not usable as evidence; the structural
  argument carries it.)
- Degenerate repetition -> ~1.6x, but that is the m=4 ceiling and an artifact of
  a 2B ternary model looping. Reported for completeness, **not** a workload.
- `m > 4` would not help even if the GEMV could take it: the fit
  `GEMV(m) ~= ceil(m/4)*A + c*m` says the per-column term dominates.

**This spends the last ungated decode lever.** What is left on the decode side
all reassociates a summation (`gdn_step`'s thread count, `dot_wide`'s FMA chain)
and sits behind the explicit tolerance gate the user has not taken, or needs the
`gemv_q2_0` weight-axis redesign. Next, in order of value: `GpuState`
snapshot/restore for **prefix caching** (a different feature: spec decode
rewinds *within* a pass, prefix caching *between* calls), then the m>1 batched
attention (6 layers x m calls -> 1), then a chunked parallel GDN scan (~10% of a
chunk).

### P5e -- the payoff guard: the policy stops assuming this laptop (done, `ed69032`)

**Why it exists.** P5d's break-even is `1 + j > C(m)/C(1)`, and `C(m)/C(1)` is a
device property. On this box an m=4 fire costs ~2.3 plain steps, so n>=4 clears
break-even by a wide margin. On a machine where a batched pass is relatively
more expensive, the same policy is a net loss; on one where it is relatively
cheaper, the trigger is needlessly timid. The engine has to work anywhere, so
the policy is not allowed to assume the ratio.

**What it does.** `generate_spec` times the passes it is already running -- the
last 16 plain steps and the last 6 fires -- and compares **tokens per
millisecond**. A fire is allowed only if it beats a plain step (with a 2%
tolerance so noise cannot pause a good policy). That comparison *is* the
objective, so it cannot suppress a fire that is genuinely profitable on whatever
backend is running.

**Measured, one interleaved run, story prompt, 256 tokens, all six arms
byte-identical to greedy:**

| policy | guarded | unguarded |
|---|---|---|
| k=3 n>=4 (the tuned default) | **1.299x** | 1.289x |
| k=3 n>=1 (deliberately too weak) | **1.239x** | 1.077x |
| k=2 n>=1 (deliberately too weak) | **1.212x** | 1.135x |

On the tuned policy the guard fires on *exactly the same steps* as the unguarded
arm -- same `fire%` 29.3%, same `P(j)=2/0/1/38` -- so the 0.01 spread is noise
and the guard costs nothing where drafting already wins. On the weak triggers it
recovers 0.16x and 0.08x by refusing the fires that were being rejected.

Also measured: a register-rewrite prompt at 192 tokens hits **1.642x** with
92.2% fires and 100% of 47 fires landing all three drafts -- the m=4 ceiling,
reached because the model reproduced the instruction's own phrasing. Guarded and
unguarded identical there too.

**Two refuted guard designs, recorded because the failure mode is subtle.**
Both an "emitted vs plain-step equivalents" window (normalising each sample by a
baseline that *includes* it) and an EMA-baseline variant drifted *against*
drafting on a rising-context run: because plain_ms climbs as attention grows,
each plain step normalised to slightly more than 1 equivalent and the error
accumulated over the window. On the tuned policy that artefact cost **3.5%**
(1.284x -> 1.239x) and suppressed a quarter of the fires that should have fired.
The fix is to store raw ms and normalise once, at comparison time, on both
sides -- or better, compare tokens/ms so there is no normalisation at all. *A
guard measured by proxy finds the proxy's bias.*

**Properties that make it safe to default on.** (1) It cannot change the output:
it only chooses which steps are batched, and a batch a draft does not survive is
rewound, so `bench_draft`'s element-for-element identity gate still applies.
(2) It cannot lock out: a paused guard spends one fire every 16 plain steps to
re-probe, so a run that turns repetitive halfway through is picked up again with
no external signal. 4 host unit tests cover optimistic bootstrap, pausing, not
pausing, and re-probing.

**Reachable.** `bonsai <gguf> <tok> <prompt> <n> spec [k:min_match]` runs the
GPU path with drafting and prints the SpecStats; `cpu`, `gpu` and `spec` all
reproduce the saved token-id witness byte for byte. Policy is a runtime value.

**Next**, in order of value: `GpuState` snapshot/restore for prefix caching,
then the m>1 batched attention (6 layers x m calls -> 1), then a chunked
parallel GDN scan (~10% of a chunk).

---

## 14. P6 -- Coverage: run real HuggingFace models (the objective)

**Decided 2026-10-17:** the remaining gap against the goal ("run any model
within reason from HF, including MoE, on 8 GB, hardware-agnostically") is **not
tok/s. It is coverage.** P5 spent the decode-side performance levers down to
single-digit percentages on one 2B model; P6 spends the *capability* levers --
what can run at all. The user has chosen performance (P7) *after* coverage, and
has unblocked the reassociation levers (P9) under the §0 discipline.

### 14.0 The box, measured (2026-10-17)

| Resource | Measured | Consequence for this phase |
|---|---|---|
| GPU | RTX 2070 Max-Q, **8192 MiB total, 5 MiB used** | ~8 GB usable; display already off the card (§11.1 satisfied) |
| RAM | **15.4 GB total / 11.5 GB available** | the pinned tier is *small*; see below |
| CPU | i7-10750H, 6c/12t, **AVX2 + FMA + F16C only** | no AVX-512, **no AVX-VNNI** |
| Disk | Samsung NVMe, **~2.0 GB/s** measured, **17 GB free** | streaming is fast enough to matter |
| Reference | **no llama.cpp**; `transformers 5.3.0`, `hf` CLI, `huggingface_hub` present | token-parity reference = `transformers` or KORE-internal gates |

Three consequences that change the earlier plan:

1. **The file tier is load-bearing, not a fallback.** §10 says "do NOT stream
   weights from SSD per token" -- that was written for the references' 32 GB
   RAM. Here RAM is 11.5 GB, so any model over ~12 GB *cannot* be RAM-resident.
   The tier model on this box is VRAM (8) + RAM (~10) + NVMe (2.0 GB/s), and a
   per-token expert working set of <= ~1-2 GB is streamable. Re-state §10's rule
   as: *do not stream the whole weight set per token*; streaming a bounded
   expert working set is now the plan.
2. **The CPU expert lane is weaker than the references'.** AVX2/FMA/F16C, no
   VNNI. Strata's concurrent CPU-compute lane is a smaller bet on this box.
   Still measure it (P6.4); do not *design around* it.
3. No `llama.cpp` means no quick external oracle. The gates that need no oracle
   -- `verify_m` bit-identity, offload==all-resident byte-identity -- carry most
   of the weight; `transformers` covers the one-time arch-parity check.

### 14.1 The finding that reorders the work: it is quant coverage, not the arch key

Verified in-tree:

- `qwen35` (dense, the fast path) holds weights as **`Q2W` -- Q2_0 only**
  (`qwen35.rs` `q2w()` + `gemv_q2_0`). It cannot load a K-quant.
- The MoE expert bank is **Q2_0-only** (`MoeConfig::expert_bytes` /
  `MoeLayer::from_stacked`: 18 B per 64 weights; `from_gguf` hard-errors on
  anything else).
- The **generic** `llama.rs` path already accepts **Q4_K / Q6_K packed,
  VRAM-resident** (`Linear::from_q4_k/from_q6_k`, `gemv_q4_k_split3`,
  `gemv_q6_k_split3`, `fused_qkv_q4_k/q6_k`) -- but it is a *separate* engine
  with none of P5's features (no split-KV long context, no chunked prefill, no
  rewind, no spec decode).
- **The real GGUFs sampled do not ship Q2_0** (see 14.4): they ship Q3_K_M /
  Q4_K_M / Q6_K / Q8_0. Bonsai-2B is the exception, not the rule.

So the load-bearing blocker is that **the fast path's weight representation is
hard-coded to Q2_0**, while the kernels for Q4_K/Q6_K already exist one engine
away. P6.1 attacks exactly that, and it unblocks dense *and* MoE at once.

(Third engine noted: `kore-serve/src/layered` -- its `layered_gguf_test` is the
hang in §13. Treat as a consolidation target in P8, not a vehicle here.)

### 14.2 The family to target: `qwen35moe`

Read from the HF API: the GGUF arch key for the Qwen3.5/3.6/3.8 MoE line is
**`qwen35moe`** (confirmed on `logic65/Qwen3.8-Whittle-MoE-27B-A17.8B-GGUF`).
That is the **MoE sibling of the `qwen35` we already implement** -- same
gated-DeltaNet + full-attention interleave, same rope/SSM/hadamard metadata.
Consequence: a `qwen35moe` model inherits split-KV, chunked prefill, rewind and
spec decode **by construction**. The only new component is the FFN.

Recorded negative: **the HF tag `qwen3_5_moe` is not a proxy for the arch key.**
`unsloth/Ornith-1.0-9B-GGUF` carries that tag but is `architecture: qwen35`
(dense). Always read `general.architecture` from the GGUF header, never the tag.

### 14.3 P6 phases (each ends on a gate)

- **P6.1 -- quant-agnostic resident weights in the qwen35 path.** Generalise
  `Q2W` to a quant-dispatched weight type that accepts Q4_K (first) then Q6_K,
  reusing the existing `gemv_q4_k*`/`gemv_q6_k*` kernels. Keep Q2_0 working
  (Bonsai must stay byte-identical).
  **Gate:** a small real Q4_K `qwen35` GGUF runs and matches `transformers`
  token-for-token; `verify_m` green at m=1,2,4.
- **P6.2 -- `qwen35moe` arch + MoE FFN.** Accept
  `general.architecture == "qwen35moe"`; read `expert_count` /
  `expert_used_count` / shared-expert metadata; load the 3-D
  `ffn_{gate,up,down}_exps.weight` banks (contiguous per-expert slabs); wire
  router -> top-k -> per-expert GEMV -> weighted combine, reusing `MoeLayer`.
  Extract the FFN as a seam (`Dense` | `Moe`) so both paths share it.
  **Gate:** `transformers` token-parity on `tiny-random/qwen3.5-moe`; `verify_m`
  green on the MoE arch.
- **P6.3 -- quant-aware expert banks.** Generalise `expert_bytes` /
  `from_stacked` to dispatch on quant (Q4_K). Prereq for P6.4 on any real
  checkpoint.
  **Gate:** offloaded run is **byte-identical** to all-resident (the §0
  invariant, already proven at synthetic scale in P3a).
- **P6.4 -- offload on *this* box.** VRAM hot experts + RAM tier + NVMe
  streaming; measure hit rate, tok/s, peak VRAM/RAM, and **MB/token from the
  file tier**. This is where the 11.5 GB RAM constraint bites and where the
  revised tier rule from 14.0 is tested.
  **Gate:** a model whose weights exceed 8 GB VRAM runs; report the honest
  curve (no cherry-picked prompt).
- **P6.5 -- dense coverage (the non-MoE half of "any model").** Make the
  generic transformer loader **prefix-driven**: resolve `general.architecture`
  -> metadata prefix, add `qwen2`/`qwen3`/`mistral`/`gemma`. Today it hard-codes
  `llama.*` and dies at "missing llama.block_count" for every other family.
  **Gate:** a real `qwen2`/`qwen3` GGUF runs and matches `transformers`.

### 14.4 Checkpoint selection (verified via the HF API, not guessed)

| Role | Candidate | Notes |
|---|---|---|
| Arch parity (tiny) | `tiny-random/qwen3.5-moe` | safetensors, random weights; shape/logit plumbing vs `transformers` in minutes |
| Dense Q4_K fixture | `unsloth/Ornith-1.0-9B-GGUF` (Q4_K_M) | arch **`qwen35`** (dense), 9 B; exercises P6.1 on a *real* model |
| MoE offload demo | `logic65/Qwen3.8-Whittle-MoE-27B-A17.8B-GGUF` (Q3_K_M/Q4_K_M) | arch **`qwen35moe`**, 26.9 B params, 262144 ctx |

**Sizing, resolved by range-reading the GGUF headers (2026-10-17, no download):**

| File | Size | Arch | Quant | Usable? |
|---|---|---|---|---|
| `POCKET-35B-IQ1_M` | 8.24 GB | `qwen35moe` | IQ1_M | no -- IQ kernel absent |
| `POCKET-35B-Q2_K` | 12.94 GB | `qwen35moe` | Q2_K | no -- Q2_K kernel absent |
| `Whittle-27B-Q3_K_M` | 13.87 GB | `qwen35moe` | Q3_K_M | no -- Q3_K kernel absent |
| `Whittle-27B-Q4_K_M` | 17.37 GB | `qwen35moe` | **Q4_K_M** | **yes -- the only supported-quant real MoE** |
| `POCKET-35B-Q4_K_M` | 21.17 GB | `qwen35moe` | **Q4_K_M** | yes, but larger for no gain |
| `Whittle-27B-Q6_K` | 23.14 GB | `qwen35moe` | **Q6_K** | yes, larger still |

Both repos are the **same arch we now run**, which removes the last structural
unknown from the MoE demo. Whittle's header: `expert_used_count = 16`,
`expert_feed_forward_length = 192`, `expert_shared_feed_forward_length = 5120`,
`ssm.inner_size = 6144`, `ssm.state_size = 128`, `ssm.time_step_rank = 48`,
`ssm.group_count = 16`, `head_dim = 256`, `full_attention_interval = 4`, 1171
tensors. POCKET: `expert_count = 256`, `expert_used_count = 8`, `block_count =
40`, `embedding_length = 2048`.

**Consequence:** the demo file is **Whittle-27B-Q4_K_M at 17.37 GB**, and it
needs *both* a `target/` prune (26 G -> ~32 G free) *and* NVMe-streamed experts,
because 17.37 > 11.5 GB RAM > 8 GB VRAM. That is precisely P6.4, in one file.
Note Q3_K_M (13.87 GB, still over RAM) would be the cheaper target if a Q3_K
kernel existed; it does not, and `oxillama`'s f32 fallback would need 108 GB of
RAM for 27 B, so Q4_K_M is the honest choice.

**Rejected alternatives:** Qwen1.5-MoE-A2.7B and OLMoE (Qwen2 / llama+MoE
families). Both would need a *new* dense-transformer engine and inherit **none**
of the P5 speed features -- strictly more work for a strictly worse demo.

### 14.5 P7 -- speed on what already runs (Tier 2, order unchanged)

- **E. Batched m>1 attention** (~13% of m=4 / prefill). Needs per-row
  `seq_k`/shard-count in the split kernel -- the shard **combine order is the
  summation order**, so bit-identity *requires* it. Gate: `verify_m` 1,2,4.
- **F. Chunked parallel GDN scan** (~10% of a chunk), replacing the host-side
  per-row loop in the 18 DeltaNet layers. Gate: associativity argument + bit
  identity.
- **G. Prefix caching** (`GpuState` snapshot/restore *between* calls; distinct
  from spec decode's rewind *within* a pass). Serving lever; no single-stream win.

### 14.6 P8 -- hygiene, and unblocking the standing blockers

- **H.** `alloc_storage_buffer` -> `PooledBuffer` at ~30 call sites (kills the
  forgotten-recycle bug class that caused the m>1 decay).
- **I.** Measure the genuine **no-fire floor** on a low-fire workload (the Q&A
  prompt emits EOS immediately, so the earlier ratio was meaningless).
- **J.** Clear the pre-existing failures: `kore-serve` `layered_gguf_test` hang,
  workspace `clippy -- -D warnings` (`kore-core/src/device.rs`, `kore-vulkan`,
  `kore-nn`), `cargo fmt --check` drift (`qwen35.rs`, `backend.rs`,
  `embedding.rs`, `probe_meta.rs`).

### 14.7 P9 -- the reassociation levers (UNBLOCKED by the user, 2026-10-17)

`gdn_step` thread count (K, 10--17% of m=1) and the `dot_wide` FMA chain
(L, ~27% fewer instructions). The §0 gate is **unchanged** and is the whole
point of unblocking them: a reassociated summation does **not** land on a
token-id diff. Each change must carry, in its own commit message:

1. the **explicit error bound / tolerance argument** (why the reassociated order
   is within tolerance for the reduction actually performed), and
2. `bench_chain` showing **fewer exposed µs** under cool-die interleaved A/B.

Order: **L first** (pure ALU, larger instruction win, isolated in `dot_wide`),
then **K**. If L's bound cannot be argued convincingly, it stops there and K is
re-examined on its own.

### 14.8 Order, and the first action

Order: **P6.1 -> P6.2 -> P6.3 -> P6.4 -> P6.5** (coverage), with P7/P8 items
interleaved as palate-cleansers; **P9 last** (it touches the kernels the others
measure, so measuring it against a moving baseline would be worthless).

**First action (cheap, and it prevents building to a guessed schema):** dump the
header of a small real `qwen35`/`qwen35moe` GGUF -- `general.architecture`, all
metadata keys, the tensor list with quant types and dims, and
`expert_count`/`expert_used_count`. Confirm the `*_exps` layout and the FFN
seam against reality **before** writing any loader.

### 14.10 P6.1b -- LANDED, and it exposed two whole-model coverage bugs

**What landed**

- `QKind { Q2_0, Q4K, Q5K, Q6K }` + `QW { name, m, n, kind, shared }` with
  `gemv`/`row_f32`, replacing the Q2_0-only `Q2W` everywhere (load, CPU lane, GPU
  materialise, head, embedding).
- Load-time validation: a tensor's byte count must match what its quant type
  requires before it is allowed to become a weight.
- **Hadamard is now optional**: the transform is applied iff the file carries
  `prism.hadamard.*`. Ornith has none, so `had_fwd` is the identity and `had_pre`
  grew a `had: bool` flag (reusing `HadPreUniforms._pad0`) so the unfolded case
  still costs exactly one dispatch. `KORE_HAD_OFF` remains a *different* knob
  (an ablation of a folded file's transform).
- **Untied head**: `output.weight` when the file has one, else the tied embedding.
- CPU K-quant lane via `oxillama-quant`, **zero copy**: a
  `TensorBytes { gg: Arc<MiniGguf>, name }` adapter with `unsafe impl ByteOwner`
  feeds `SharedBytes::from_owner`, so no K-quant tensor is ever expanded to f32.
- GPU `GpuW`/`wgemv` dispatch to `gemv_q{2_0,4_k,5_k,6_k}`.

**14.10.1 Bug 1 -- the GDN key/value head grouping was wrong**

`n_v != n_k` is the GQA case inside the *linear* attention (Ornith: 32 value
heads, 16 key heads). llama.cpp pairs them as, from
`ggml/src/ggml-cpu/ops.cpp:11200` (`ggml_compute_forward_gated_delta_net_one_chunk`):

```cpp
const int64_t iq1 = iv1 % neq1;   // value head -> key head
```

i.e. the key heads are **tiled** up to the value head count. Its *unfused* path
reaches the same mapping through `ggml_repeat_4d` (`src/models/qwen35.cpp:448`),
so both upstream paths agree. HF's `repeat_interleave`
(`transformers/models/qwen3_5/modeling_qwen3_5.py:588`) would instead be
**interleaved**, and `ggml/include/ggml.h:2644` carries
`[TAG_GGML_GDN_BCAST]` -- a TODO to make the fused kernel's broadcast type
configurable. So the ambiguity is real, documented upstream, and **the GGUF
follows llama.cpp**. KORE now does too, in both the CPU `delta_step` and the
`gdn_step` shader.

Symptom: Ornith-9B emitted `" a"` where llama.cpp emits `" Paris."`, and the
correct continuation was **absent from the top-10 entirely** -- so no tolerance
argument could have rescued it. Bonsai has `n_k == n_v`, which is why a dense
delta model that already worked never surfaced this.

**14.10.2 Bug 2 -- a 1-D activation read as `m = k`**

`gemv_q4_k` / `gemv_q5_k` / `gemv_q6_k` / `gemv_q4_k_split3` computed

```rust
let m = if x_shape.is_empty() { 1 } else { x_shape[0] };
let k_in = if x_shape.len() > 1 { x_shape[1] } else { x_shape[0] };
```

so a 1-D `[4096]` activation became `m = 4096` and the GEMV produced a
`[4096, 4096]` output. `gemv_q2_0` had always had the right rule
(`m = if x_shape.len() == 2 { x_shape[0] } else { 1 }`); all four K-quant kernels
now match it, with `k_in = x.numel() / m`.

Reachability: `gdn_step` returns a 1-D `[n_v*d_v]`, `had_fwd` passes it through
unchanged when the model is *not* folded, and only then does a K-quant GEMV see a
1-D input. Bonsai is folded *and* Q2_0, so the bug was doubly invisible. The
failure was loud only because the shape happened to be rejected downstream -- a
`[k, k]` that happened to match would have been silent.

**14.10.3 Also found: `gemv_q5_k` was never wired into the Vulkan trait impl**

The kernel landed at `c630b72`, but `DeviceBackend::gemv_q5_k` for Vulkan was
still on the trait's "not implemented" default, so the GPU lane died on Ornith's
24 `blk.N.ssm_out.weight` tensors. Trivial to fix, and exactly the
"kernel exists" vs "kernel is reachable" gap that P6 is about.

**14.10.4 Gates (measured 2026-10-17; cool-die discipline as in §8)**

| gate | result |
| --- | --- |
| Ornith token stream vs llama.cpp | **byte-identical**, CPU *and* GPU |
| Bonsai token stream vs the witness | byte-identical (CPU and GPU) |
| `verify_m` Bonsai, m = 1 / 2 / 4 | 11 positions bit-identical, state probe agrees, rewind legs bit-identical |
| `gpu_ops_tests` | 26/26 (2 new) |
| `k_quant_fixtures` | 3/3 |
| `cargo test --workspace --exclude kore-serve` | 562 passed, 0 failed |

Two regression tests were added, and the GDN one was **verified to fail** under
the old (interleaved) mapping -- divergence began exactly at index 128, the first
column of value head 1 -- so it cannot pass vacuously:

- `test_vulkan_gdn_step_groups_key_heads_by_tiling` -- oracle-free
  self-consistency: explicitly tile q/k up to `n_v` heads and run with
  `n_k == n_v`, where the mapping is the identity; output *and* recurrent state
  must then match bit-for-bit.
- `test_vulkan_k_quant_gemv_treats_1d_activation_as_one_row` -- 1-D and `[1, k]`
  inputs must agree bit-for-bit and both yield `[1, n]`.

### 14.11 The runtime oracle (new, and it should gate every arch/quant from now on)

Plausibility is not evidence. `llama.cpp` is the authority for **GGUF semantics**
(the file format's contract), exactly as `gguf-py` is for **quant layout**. Both
are now wired in:

- **Layout oracle** -- `crates/kore-kernels/tests/k_quant_fixtures.rs`, `k_quant.rs`
  as the written spec (P6.1b-1).
- **Semantics oracle** -- a 100-line greedy token-level reference built against a
  CPU-only llama.cpp checkout at `/tmp/opencode/llc`:

  ```bash
  # build once (needs /tmp/opencode/llc/src to be a built llama.cpp tree)
  /tmp/opencode/llc/build_ref.sh
  # then, for any GGUF + prompt:
  /tmp/opencode/llc/ref_ids model.gguf "prompt" 16        # -> prompt_ids / gen_ids / gen_text
  ```

  It prints `prompt_ids`, `gen_ids` and `gen_text` for a greedy decode, which is a
  deterministic function of the model. Compare against
  `./target/release/examples/bonsai <gguf> <tok.json> "prompt" 16 cpu` (and `gpu`).

**Discipline note learned the hard way:** read a probe's *entire* output. A
top-10 logits print inspected through `tail -14` silently hid rank 1, which sent
a real investigation down a false trail -- the engine had been correct at that
point all along.

### 14.12 P6.2 -- `qwen35moe` runs, byte-identical to llama.cpp (LANDED)

The MoE leg had no fixture: every checkpoint on disk was either dense or too big
for the remaining 6.9 G. The way through was to *manufacture* one. HF
`tiny-random/qwen3.5-moe` is a 10 MB random-weight build of the real arch --
`num_experts=128`, `num_experts_per_tok=10`, 4 layers (3 DeltaNet + 1 full),
`hidden_size=8`, shared expert, untied head, `n_k=4 != n_v=8` -- and
`convert_hf_to_gguf.py` supports `Qwen3_5MoeForConditionalGeneration`
(`conversion/qwen.py:740`). 96 tensors, 9.07 MB payload, 20 MB with the vocab.

Reproduce:

```bash
huggingface-cli download tiny-random/qwen3.5-moe --local-dir /tmp/tiny-moe/m \
    --include "*.json" "*.safetensors" "*.jinja"
python /tmp/opencode/llc/src/convert_hf_to_gguf.py /tmp/tiny-moe/m \
    --outtype f16 --outfile /tmp/tiny-moe/qwen35moe-tiny-f16.gguf
/tmp/opencode/llc/ref_ids /tmp/tiny-moe/qwen35moe-tiny-f16.gguf "The capital of France is" 8
# prompt_ids: 760 6511 314 9338 369
# gen_ids:    132029 192124 633 236180 112090 209208 225193 143183
```

**Gate: passed.** `bonsai ... cpu` reproduces both streams exactly. The prompt
is 5 tokens, so the chunked prefill runs an **m=4** MoE batch followed by m=1 --
so this is a batch-invariance check too, not just a single-row one. Dense
`qwen35` (Ornith, CPU *and* GPU) is unchanged and still oracle-identical.

**The authoritative schema the probe produced** (worth keeping, it is the whole
contract):

| key | value |
|---|---|
| `general.architecture` | `qwen35moe` (HF tag `qwen3_5_moe` appears nowhere) |
| `block_count` / `nextn_predict_layers` | 5 / 1 -> **4 real layers** |
| `expert_count` / `expert_used_count` | 128 / 10 |
| `expert_feed_forward_length` / `_shared_` | 32 / 32 |
| `ssm.inner_size/group_count/time_step_rank/state_size/conv_kernel` | 256 / 4 / 8 / 32 / 4 |
| `attention.key_length` / `value_length` | 32 / 32 |
| `rope.dimension_sections` / `dimension_count` | `[1,1,2,0]` / 8 (1/4 partial) |

MoE tensor layout (all ne[0]-contiguous):

```
ffn_gate_inp.weight      [hidden, n_experts]              F32  <- router
ffn_gate_inp_shexp.weight[hidden, 1]                      F32  <- shared-expert gate
ffn_{gate,up}_exps.weight[hidden, moe_ffn, n_experts]
ffn_down_exps.weight     [moe_ffn, hidden, n_experts]
ffn_{gate,up}_shexp.weight[hidden, sshexp]
ffn_down_shexp.weight    [sshexp, hidden]
```

**What landed**

- **Arch accept + prefix-driven metadata.** `qwen35` and `qwen35moe`; the arch
  key *is* the metadata prefix, so every `qwen35.*` lookup became
  `{arch}.*`. This is the core of P6.5, done here because P6.2 could not be
  written without it.
- **nextn blocks are skipped**, not loaded: `n_layers = block_count -
  nextn_predict_layers`. KORE's drafter is prompt-lookup, so an MTP head is
  dead weight.
- **`FfnW::{Dense, Moe}`** replacing the three `ffn_*` weights on both layer
  kinds, so the delta and full layers share one seam.
- **`MoeW` addresses experts as byte windows.** `QW` gained `off` plus the
  `Arc<dyn ByteOwner>`, so `gate.expert(e, stride)` is an offset change --
  nothing is copied, nothing is expanded, and an expert outside the top-k is
  never touched. This is the same struct that the streaming engine will use, so
  P6.3/P6.4 do not need a second weight abstraction.
- **Routing exactly as llama.cpp**: softmax over all experts -> top-k
  (`total_cmp`, NaN can neither panic nor win) -> renormalise -> weighted sum;
  then `sigmoid(gate_inp_shexp . h) * shexp(h)`.
- **F32 and F16 weights are now loadable** (`QKind::F32`, `QKind::F16`, plus
  f16 in `MiniGguf::tensor_f32`). This closed a standing coverage gap and it was
  not optional: the router of a real MoE file is F32, and an `--outtype f16`
  export is otherwise all-F16.

**Two whole-model bugs this fixture exposed** (both invisible on every file we
had):

1. **`attn_gate` output width was hard-coded to `hidden`.** The gate is
   `[hidden, n_v * d_v]`, i.e. one scalar per *value channel*. Bonsai and Ornith
   both have `n_v * d_v == hidden` (4096), so a `hidden`-sized buffer was
   accidentally right; on the tiny model `n_v * d_v = 256 != 8` and the gemv
   rejected the shape. This is the third bug in this family of "an identity that
   happened to hold for Bonsai" (after the tied head and the GDN tile mapping).
2. **`qw` built a one-expert view for a stacked tensor.** `bytes()` clamps to
   the view, so every expert after the first read nothing (`f16 gemv: 0 bytes`).
   The view now spans the whole expert axis and `off` selects the window;
   `view()` keeps the cached narrow view for plain 2-D weights so the dense path
   pays nothing.

**Lesson:** a fixture that shares every dimension with the models you already
run cannot test the dimensions. `hidden == n_v*d_v` and `n_k == n_v` each hid a
bug for weeks. The tiny-random conversion is cheap enough that every new arch
should get one *before* it gets a real checkpoint.

**Not done here, deliberately:** the device lane refuses `qwen35moe` with a
clear message rather than building a dense FFN that would run and produce wrong
tokens. Resident GPU MoE is P6.3's job (the experts must stream).

### 14.13 P8/J -- the lint gate is green (LANDED), and what it was hiding

`cargo clippy --workspace -- -D warnings` and `cargo fmt --all -- --check` both
pass for the first time in this branch's history. Two commits, deliberately
separated so the reformat cannot hide a semantic change:

1. `style: apply rustfmt across the workspace` -- ~30 files, 2000 lines, pure
   formatting. The suite was re-run after it: 562 passed.
2. `chore: clear cargo clippy --workspace -- -D warnings` -- mechanical fixes
   (`is_multiple_of`, `div_ceil`, `sort_by_key`, range patterns, one RoPE loop,
   two GEMV inner loops, one decode loop that counted positions by hand) plus
   three justified `#[allow]s` (`too_many_arguments` on the `DeviceBackend`
   binding tables, `needless_range_loop` on fixed-size accumulators,
   `large_enum_variant` on the per-layer enums) and one type alias
   (`Q6kPadCache`).

**Gate:** after the clippy commit, all three oracle arms were re-run --
tiny MoE, Ornith CPU, Ornith GPU -- and every token id is unchanged. The
formatting commit is justified by the same evidence.

**Not fixed, out of scope on purpose:** `--all-targets` still reports ~40 lints
in *test* code of `kore-btes`, `kore-data`, `kore-clifford` and `kore-kernels`
(`kore-data/src/multipack.rs` index arithmetic like `0 * 5` in a mask test is
the noisy one; it is correct, just arithmetic clippy dislikes). The documented
gate does not include test targets. Worth a single follow-up commit before the
big push, but it changes no behaviour.

### 14.14 P6.3a -- the residency layer (LANDED, `8f04d02`)

`DeviceBank` (in `qwen35.rs`, next to `GpuW`) is the piece that makes a 27 B
expert bank fit an 8 GB card:

- the host side is the file's own mapping, so a miss uploads exactly the window
  that was asked for; experts outside the top-k are never touched (the property
  `MoeW` already relies on);
- replacement is `moe::ExpertCache`'s LRU -- the same policy as the CPU-side
  prototype -- and the slot count is floored at `n_expert_used`, so one token
  cannot evict an expert it is about to need;
- a slot goes to the same `gemv_q*` kernel a resident weight uses, over the same
  bytes, so **offloaded == resident by construction**. Q2_0 is the only format
  needing the padded repack, and it is done on the miss, not on every read.

**Gate (green).** `streamed_experts_match_resident_by_bit`: four Q4_K experts
through a **one-slot** pool against four resident copies, comparing `to_bits`,
on the real adapter. Non-vacuity is enforced inside the test (reference must be
non-zero; the four experts must decode differently), and a deliberate off-by-one
in the expert window fails it (`expert 2 lane 0: -75.171875 != -165.625`).

**Next slice (design settled, not built):** the routed device FFN. The dispatch
arithmetic says *do not* loop experts one gemv at a time:

| approach | dispatches / layer | x64 layers |
|---|---|---|
| per-expert loop (3 gemv + swiglu + combine) | 5 x 16 = 80 | 5120 |
| grouped: stage k experts, one wide gemv | 4 + 1 | 320 |

The grouped form works because a GEMV is row-independent: uploading the k
selected experts **contiguously** into a `k*stride` staging buffer turns
`k x [m, n]` into one `[m, k*n]` matvec. SwiGLU is elementwise over the whole
`k*n` vector; `down` likewise becomes `[m, k*n]`; only the weighted sum of the k
`hidden`-blocks needs a new kernel (`moe_combine`, 4096 outputs each summing k
terms in fixed order, so it stays deterministic). Residency then pays off as
**VRAM->VRAM copies** (`copy_buffer` from a hot slot) instead of PCIe uploads --
which needs a ranged copy, the one primitive still missing.

### 14.15 P6.3b -- the real MoE's quant mix (LANDED, this commit)

The 17.37 GB `Whittle-27B-Q4_K_M` header, read over HTTP range (11 MB of
metadata + tensor table, 1171 tensors) rather than downloaded:

```
Q4_K 561   F32 481   Q6_K 65   Q8_0 32   Q5_0 32
expert_count 64, expert_used_count 16, expert_ffn 192, hidden 5120,
block_count 64, ssm.inner 6144, ssm.state 128, head_count 24, head_dim 256
```

Two things fall out, and both would have blocked the run:

1. **`ffn_down_exps` is `[192, 5120, 64]` Q8_0/Q5_0.** `ne[0] = 192` is not a
   multiple of 256, so a K-quant is illegal there and the quantizer falls back to
   the 32-weight legacy formats -- in *every* layer. KORE had **no** Q5_0 and no
   Q8_0 anywhere. Downloading 17 GB first would have been wasted.
2. **`ffn_gate_inp_shexp.weight` is 1-D `[5120]`.** The GGUF writer trims
   trailing unit dims, so a single-output projection comes back as a vector.
   `qw` required 2-D and refused every file that had been through
   `llama-quantize`. 369 of this file's tensors are 1-D.

Landed: `QKind::Q5_0` (type 6, 32 weights / 22 B) and `QKind::Q8_0` (type 8,
32 / 34 B), the 1-D-as-`[n, 1]` rule, dequant via `oxillama_quant` (AVX2 kernels
exist for both), and a **load-time** refusal in the GPU lane so a missing device
kernel reads as a lane limitation instead of a corrupt model.

**Gates (green).**

- `tiny-random/qwen3.5-moe` re-quantized with `llama-quantize` reproduces the
  real file's fallback structure at 19.8 MB: F16 41, F32 40, **Q5_0 6, Q4_K 5,
  Q8_0 4**, with `ffn_down_exps` Q5_0/Q8_0 exactly as the 27 B file has it.
  This fixture is the coverage test for the whole quant mix.
- Pure `Q5_0` and pure `Q8_0` exports: **12/12 greedy tokens byte-identical to
  llama.cpp** (and Q8_0 reproduces the F16 ids, as it should).
- The F16 P6.2 gate is still byte-identical.

**Recorded negative (important for future gates).** On the *mixed* Q4_K_M file
KORE matches 4 tokens and then splits. It is not a bug: at that step the two
candidates sit at 1.2932698 / 1.2925190 in KORE and 1.29233193 / 1.29206085 in
llama.cpp -- a **7.5e-4 near-tie, ranked oppositely**. Cause: llama.cpp's CPU
`q4_K_8x8` **repack**, which reassociates the dot; KORE deliberately does not
repack. So *token-id parity against llama.cpp is not a bit-gate for K-quant
files* -- it happens to hold on Ornith-9B over 8 tokens, and that was luck.
Byte-identity is a KORE-to-KORE contract (batch invariance); llama.cpp is an
oracle for **semantics**. New harness `examples/logit_probe.rs` classifies a
split as near-tie vs real divergence in one command, and
`q4k_one_block_per_row_matches_f64` pins the one-block-per-row dot against an
f64 reference (the shape the MoE's `down` projection has and Ornith never
exercised).

**Sizing rule found by the same header read -- and it changes the demo.**

```
routed experts        : 8.05 GB
everything else       : 9.31 GB   (shared expert 1.89, embeddings/head 1.76)
total                 : 17.36 GB
```

The **resident tier alone (9.31 GB) exceeds 8 GB VRAM**. Even with a perfect
expert cache, every token must read the non-expert weights, so the ceiling is
NVMe speed over ~9 GB (~4.7 s/token), not PCIe over the expert working set
(16/64 = 25% of 8.05 GB ~ 2 GB ~ 1 s/token). With ~1.3 GB of head pinned, the
attention/GDN/shared-expert tier is ~8 GB and *just* fits; a hit-rate-weighted
estimate is **1-3 tok/s**, i.e. it runs, and it is not fast.

Therefore P6.4's demo checkpoint must satisfy **resident tier <= ~7.5 GB**, not
"total <= RAM". `Whittle-27B` violates it by 1.8 GB; the smaller `POCKET-35B`
mixes are the candidates, and they need Q2_K/Q3_K (no kernel yet). Record this
before spending another byte of bandwidth.

### 14.16 The checkpoint that actually fits (2026-10-18) -- P6.4's target changes

Reading the second candidate's header settled a question that a download could
not have: on this box `Whittle-27B` is not runnable at a useful speed, and
`POCKET-35B-Q2_K` is. Same family, opposite tier split:

| checkpoint | size | routed experts | everything else | experts/token |
|---|---|---|---|---|
| `Whittle-27B-Q4_K_M` | 17.36 GB | 8.05 GB | **9.31 GB** | 16/64 = 25% (~2.0 GB) |
| `POCKET-35B-Q2_K` | 12.93 GB | 11.66 GB | **1.27 GB** | 8/256 = 3.1% (**365 MB**) |

The rule is not "does the file fit in RAM" but **"does the resident tier fit
VRAM"**, because the resident tier is read on *every* token while each expert is
read only when routed. Whittle's 9.31 GB of attention/GDN/shared-expert weights
(two thirds of the file) cannot be cached in 8 GB, so its ceiling is NVMe
bandwidth over ~9 GB/token (~4.7 s/token) no matter how good the expert cache
is. POCKET's 1.27 GB resident tier fits with room for a ~5 GB expert cache, and
a token only needs 365 MB of the 11.66 GB bank.

POCKET is also structurally *easier*: `expert_feed_forward_length = 512` is a
multiple of 256, so `ffn_down_exps` is a proper K-quant (Q3_K) rather than
Whittle's 192-wide row that forces the Q5_0/Q8_0 fallback.

Its quant mix is `Q2_K 301, Q3_K 90, Q4_K 40, F32 301, Q6_K 1` -- i.e. **Q2_K and
Q3_K are the gate**, which is why the next slice went there rather than to GPU
kernels for the formats Whittle needed. 12.93 GB also fits the 34 GB of free
disk, unlike the 17.37 GB file plus its own KV/scratch.

### 14.17 P6.3c -- Q2_K/Q3_K/Q4_0, and a bug in the third-party kernel (LANDED)

`QKind` now covers `Q4_0` (2), `Q5_0` (6), `Q8_0` (8), `Q2_K` (10), `Q3_K` (11)
alongside `F32/F16/Q2_0/Q4_K/Q5_K/Q6_K` -- the full set a real K-quant file
mixes. Q2_K and Q3_K are **implemented in `k_quant.rs`**, not delegated.

That was forced by a finding worth recording: **`oxillama-quant` 0.1.4's Q3_K is
wrong.** Its `decode_scales` reads the twelve scale bytes as a Q4_K-style
"first 8 bytes low, next 4 bytes high" layout; llama.cpp's Q3_K is
`IIIIAAAA / JJJJBBBB / ...` -- scales 0..8 are the *low* nibbles of bytes 0..8
and scales 8..16 the *high* nibbles of those same bytes, with the two-bit high
parts packed four-per-byte in bytes 8..12 (`byte = 8 + i % 4`, `shift = 2*(i/4)`).
The first sub-block happens to agree, so it fails late: `rand_q3_k` caught it at
element 16. KORE had never used Q3_K, so this was latent.

Its Q4_0/Q5_0/Q8_0 kernels *are* correct (`dispatched_formats_match_the_reference`
+ `..._gemv_equals_dequant_then_dot`), so those still go through the dispatcher,
and Q2_K does too on the CPU lane where it is used outside our own gemv.

Three other things this slice fixed:

- **The fixture generator never actually forced finite scales.** `finite_random_blocks`
  wrote the f16 exponent into the *low* byte; an f16 is little-endian, so the
  exponent lives in the high byte. Every "forced finite" scale was random, and a
  random inf/NaN scale makes any *arithmetic* test vacuous (the dequantization
  comparisons passed anyway -- both sides read the same bytes). Now the exponent
  is written where it belongs (`exp << 2` into the high byte, mantissa split),
  and `gen.py` asserts nothing non-finite remains. All fixtures were regenerated;
  21 files, 105 KB.
- `k_quant::dequant_block(type_id, block, out)` exposes one block at a time, so the
  CPU GEMV reuses a single 256-element scratch instead of allocating per row
  (`QW::gemv` for Q2_K/Q3_K -> `gemv_k_quant`).
- A test asserting "Q3_K has no dequantizer" was the one honest signal that the
  coverage matrix had changed; it now uses Q8_K (type 15).

**Gates (green).** `k_quant` matches gguf-py bit-for-bit for Q2_K/Q3_K on both
random and zero blocks; `dequant_matches_llama_cpp_reference` covers 15 fixtures.
End to end on the tiny MoE: **`Q3_K_M` is 10/10 greedy tokens byte-identical to
llama.cpp**, and `Q2_K` reproduces the same benign near-tie already recorded for
`Q4_K_M` (KORE 110997/180412 at 1.2882318/1.2882212 -- **1.1e-5** apart -- where
llama.cpp has them 4.7e-4 apart and ranked the other way, because it repacks
`q4_0_8x8`). F16, Q8_0 and Q5_0 stay byte-identical, so nothing regressed.

### 14.18 The real 35B MoE runs (2026-10-18) -- P6.4 semantics gate

`POCKET-35B-Q2_K` (12.94 GB, 40 layers, 256 experts, 8 routed, hidden 2048,
`ffn=0`) now loads and generates on the **CPU lane**. Two blockers were found
before any kernel work, both invisible in the tiny fixture because the real file
differs from it in ways a synthetic file cannot:

- **`feed_forward_length` is absent.** A pure-MoE file has no dense FFN, so
  llama.cpp never writes the key; `POCKET` declares only `feed_forward_length`
  for experts (512) and the shared expert. The loader required it. It is now
  optional for `qwen35moe` and reads as 0, because only the dense path uses it.
  (`n_layers = block_count - nextn_predict_layers` with `nextn = 0` here, so all
  40 blocks are real.)
- **`ssm_alpha` / `ssm_beta` are Q2_K**, not F32 -- the tiny fixture had them
  unquantized. They were read through an f32-only accessor. All packing-to-f32
  decoding now lives in one module (`kore_nn::dequant::tensor_to_f32`), which
  both `MiniGguf::tensor_f32` and `QW::row_f32` call, so a small auxiliary
  tensor may be packed exactly like a big one.

The second one is the general lesson: **the f32-ness of an auxiliary tensor is a
property of the file, not of the tensor's role.** Attention/GDN gate matrices
are a favourite target of quantizers because they are small.

**Gate (CPU lane, 4 tokens, `"The capital of France is"`).** First two tokens are
byte-identical to llama.cpp (`11751 11`); the third splits on a near-tie --
llama.cpp has `264=19.5385` vs `864=19.5025` (gap 0.036) and picks 264, KORE has
`864=19.5038` vs `264=19.4750` (gap 0.029) and picks 864. Every other top-8
logit agrees to ~0.06 out of ~19.5, i.e. 0.3%, which is the accumulation of
llama.cpp's repacked Q2_K/Q3_K dots over 40 layers. Recorded as near-tie #3; the
routing, shared expert, GDN scan and attention are all confirmed correct on a
35B-A3B, which is what this gate was for.

**Speed** is the honest CPU-lane number: **0.10 tok/s** (~10 s/token), because
`gemv_k_quant` unpacks a block into a 256-element scratch and dots it in scalar
f32. That is the semantics oracle, not the engine. The GPU plan is unchanged and
is now the only thing between here and a useful number: 365 MB of experts per
token at ~6 GB/s is a ~60 ms floor, so the routed-FFN dispatch work (P6.3c) plus
`gemv_q2_k`/`gemv_q3_k` is where the 35B becomes interactive.

### 14.9 Decisions

- **Resolved:** priority = **coverage (P6) before speed (P7)** (user,
  2026-10-17); reassociation **unblocked** with the §0 discipline (per-lever
  written tolerance, not a pre-committed number, not a token-id diff);
  hardware-agnostic base stays **wgpu**, CUDA still deferred.
- **Resolved (2026-10-17, §14.10):** GGUF GDN q/k broadcast is **tile**
  (`hv % n_k`), per llama.cpp -- *not* HF's `repeat_interleave`. **When HF and
  llama.cpp disagree, the GGUF follows llama.cpp**; verify against the oracle.
- **Resolved (2026-10-17, §14.10):** a 1-D activation is `m = 1` for **every**
  quantized kernel (they all share one rule now).
- **Resolved (2026-10-17):** the staging fixture (Ornith-9B-Q4_K_M) is
  llama.cpp-parity-clean, so P6.1b is closed and P6.2 is unblocked.
- **Resolved (2026-10-17, §14.12):** `qwen35moe` is **accepted and correct** --
  byte-identical to llama.cpp on a manufactured 128-expert fixture, including an
  m=4 prefill chunk. The arch key is the metadata prefix (P6.5's core, done
  early because P6.2 needed it). **`quant_coverage` must be run first on any new
  arch**: the tiny-random conversion is now the standard first step, and the
  first real checkpoint is only worth downloading after that gate.
- **Open:** which offload-demo checkpoint (needs the P6.4 sizing probe first);
  authorization to prune `target/` for disk; the exact P9 tolerance (argued
  per-lever at the time).

### 14.19 GPU K-quant kernels: Q2_K and Q3_K (`8a29532`)

The two formats POCKET's resident tier is made of now have device kernels, so
**the GPU lane no longer refuses the checkpoint**: it loads the header, builds
the resident plan, and stops at

    the device lane does not build the routed FFN yet
    (it needs the streaming expert engine); use the CPU lane

That is the correct boundary. Every quant in the resident tier is now accepted,
and the only missing piece is **P6.3c** (routed-expert GPU FFN, §14.14/§14.17).

**Q2_K needs no adaptation.** 84 bytes is 21 words per block, so blocks are
already word-aligned and the file's payload goes up untouched. Its payload packs
four sub-blocks into one 16-byte window, one per 2-bit field, and the window
alternates 0/16 inside each 32-byte bank:

    byte = (i / 8) * 32 + (i % 2) * 16 + l        shift = 2 * ((i % 8) / 2)

Lane `t`'s two weight quads `{4t, 128+4t}` are each **one u32 at one shift**, so
the kernel's lane map and x access pattern are identical to Q4_K's, and each quad
carries exactly one scale byte (the low nibble) and one min nibble (the high
nibble). The reduction is the same `vec4` tree, so a row's bits do not depend on
the batch.

**Q3_K is 110 bytes**, which is *not* a multiple of four: consecutive blocks
start at alternating half-word offsets, so a block's scale bytes or `d` can
straddle two words. `k_quant::repack_q3_k` pads each block to 112 bytes (1.8%)
**at upload**, in both the resident `up_w` path and the expert-bank upload path,
mirroring `repack_q2_0`. The padded layout the shader is written against:

    words  0..7   hmask[32]
    words  8..23  qs[64]
    words 24..26  scales[12]
    word  27      d as f16 in the low half, two pad bytes in the high half

The third bit comes from `hmask`, where a **set** bit means the high bit is
*zero* (`qh = bit ^ 1`, `q = ql - 4*qh`). A quad's four mask bits are four
consecutive bits of one `hmask` word, so one right shift plus `& 0x01010101`
yields them in weight order; quad1 reads bit `t/8`, quad2 bit `t/8 + 4` of the
same word.

**The bug worth recording.** The first version read the scale nibbles from
`w[bo + i/4]` -- word 0, the *hmask* -- instead of word 24. It produced ratios of
exactly 29/30 and 17/32 against the spec. The fix did not come from staring at
the shader: a one-hot probe (x = e_j) showed a *constant* ratio per sub-block,
which localised the error to the scale and not the payload, and then a probe that
printed **both formulas evaluated on the same bytes** named the wrong word
outright. Lesson: when a kernel is wrong by a factor, dump both computations on
one block instead of reasoning about which line is suspect.

**Gates.** `crates/kore-nn/tests/gpu_k_quants.rs` compares each kernel against
`k_quant`'s specification (bit-exact against gguf-py) within f32 reassociation,
and asserts a row's bits are **identical** at m=1 and at m=4. A negative control
(forcing one 2-bit field to be read wrong) confirmed the tolerance test bites;
the batch test correctly did *not* fire, since it does not care about correctness.
`k_quant::repack_q3_k` has a unit test that the padding is zero and that
stripping it returns the file's own bytes.

Full gate after this slice: `fmt` clean, `clippy --workspace -D warnings` clean,
**571 tests pass** (566 + 4 GPU + 1 repack), 0 failed.

**Next (P6.3c), unchanged in shape:** stage `k` experts contiguously with
`upload_into(dst, offset, bytes)` plus a ranged device-to-device copy, one wide
`gemv_q*` over `k*n_expert`, SwiGLU via a single `had_pre(2)`, `down` as one
`k*hidden` gemv, and a small `moe_combine` reduce -- five dispatches per layer.

### 14.20 P6.3c-lite: the routed FFN on the device (POCKET-35B runs on GPU)

`POCKET-35B-Q2_K` now generates on the Vulkan lane. Twelve tokens of "The
capital of France is" are the CPU lane's own sequence, near-tie included:

    11751 11 864 369 1048 279 7526 3177 303 279 3046 13

Token 3 (`864`) is the known llama.cpp divergence (`264`), not a new one --
llama.cpp's repacked K-quant dot reassociates.

**Shape of the slice.** `GpuFfn` became an enum (`Dense` / `Moe(GpuMoe)`).
Routing runs on the host, deliberately: the expert windows must be named before
anything can be uploaded, so the decision is needed host-side regardless and a
device router would only add a readback to the one that is already unavoidable.
`router_probs` / `top_k_experts` are now shared free functions used by *both*
lanes, so the two lanes take the identical decision on the identical activation
rather than two implementations that are supposed to agree.

Everything after the decision is on the device: gate/up matvecs, fused SwiGLU
[+Hadamard] (`had_pre(2)`), the router weight folded into the activation (`down`
is linear -- same arithmetic, a quarter of the dispatch), the down matvec, and a
weighted accumulation, plus the always-on shared expert gated by `sigmoid` of
its own row. **No new shader, no new backend API** -- the banks feed the same
`gemv_q*` kernels an all-resident bank would, which is why an offloaded run stays
bit-identical to a resident one by construction.

The router and `shexp_router` stay host-side: both F32, both small, neither ever
uploaded.

Batching is refused, not faked: four columns route to up to four *different*
expert sets, so a wider pass multiplies expert traffic instead of amortising it.
Prefill chunks on a routed arch are one token wide (`width = 1` when
`cfg.n_experts > 0`).

**Measured (8 GB RTX 2070 Max-Q, 40 layers x 8/256 experts):**

| state | rate | ms/token |
|---|---|---|
| warm page cache | **5.6 tok/s** | 175 |
| cold page cache | 1.2 tok/s | 850 |

CPU lane on the same file: 0.10 tok/s. So ~56x, and the first time a 35B has run
on this card at all.

Expert bytes: ~198 MB/token (6303 misses / 12 tokens, 2409 MB, 32 slots).

**Two corrections to earlier reasoning, both from measurement.**

1. *Dispatch is not the wall.* The expert loop spends **95%** of its time in
   uploads (4698 of 4927 ms) and ~13 us per dispatch across ~55 dispatches per
   layer. P6.3c's contiguous staging is therefore a *latency* optimisation for
   the upload path, not for dispatch, and is not urgent.
2. *The "slot cliff" was a page-cache artefact.* A first pass showed 32 slots at
   1.2 tok/s and 96 at 5.6, which read as eviction churn. Interleaved A/B with a
   warm cache gives 32/96/32/96 = 4.96/5.57/5.83/5.63 -- **inside the noise**,
   with identical hit/miss counters (10017/6303/2409.6 MB at both 48 and 96).
   The 1.2 tok/s runs were the ones that followed a long test run and had to page
   the 13 GB file back in. What sets the rate is whether the file is resident,
   not the pool size. Default stays at 32 slots (~1.5 GB VRAM).

**The honest remaining costs per token (warm):** routing readbacks ~53 ms (40
syncs, one per layer), expert uploads ~50 ms (~198 MB from page cache to VRAM),
the rest compute. Nothing here needs VRAM we don't have, which is the property
that matters for bigger models.

**Next:** coverage, not speed -- P6.5 (prefix-driven loader so `llama.*` /
qwen2 / qwen3 / gemma files load at all, plus the optional q/k/v attention bias
those archs need). `llama.rs` still hard-codes `llama.*` at eight sites and has
no bias path.

## 14.21 Clifford algebra as an inference lever: falsified

The proposal was that geometric/Clifford algebra, which `kore-clifford` already
implements, could serve as a better adapter for tensor algebra and raise tok/s.
It was tested rather than argued, with predictions written down first.

**Arm A -- the same operator, written in GA form** (`kore-kernels` example
`clifford_probe`):

| form | ns / 256-wide f32 block | vs butterfly |
|---|---|---|
| butterfly (FWHT) | 383-424 | 1.0x |
| reflections (Cartan-Dieudonne) | 45,486-48,013 | **113-119x slower** |
| rotor8 (quaternion sandwich) | 392-428 | 0.9-1.1x |

The reflection form reproduces the butterfly to 6.0e-5 and the exact matrix to
2.1e-7, so this is the same operator, not a strawman: it is simply O(n^2)
multiplies where the butterfly is O(n log n) adds. The algebra you *write* an
operator in does not change its cost; the factorization does.

`rotor8` is as fast as the butterfly but computes a **different** operator -- a
relative L2 distance of 1.43 from the fold the weights were trained through. It
is an 8-DoF-per-block family, not the full orthogonal map, so it cannot be
substituted post-hoc.

**Arm B -- rotor-structured quantization** (`kore-tools/clifford-quant-probe.py`;
a rotor is a point on S^3, so the GA-native quantizer is a spherical one:
three quantized angles plus a norm):

| rate | scheme | router (f32, kurt 4.8) | ffn_down (Q6_K, kurt 1.8) |
|---|---|---|---|
| 4.0-4.25 b/w | affine-32 | **20.20%** | **16.94%** |
| 4.25 b/w | rotor-4 | 25.48% | 25.12% |
| 5.0 b/w | affine-32 | **9.41%** | **7.91%** |
| 5.0 b/w | rotor-4 | 11.85% | 12.12% |

The rotor never wins at a matched rate: 26% worse on the heavy-tailed router
weights, 53% worse on ffn_down. Scalar affine quantization of a near-Gaussian
source is close to the rate-distortion bound; a spherical chart spends code
points on a non-uniform metric.

**The constructive half.** The same table shows the algebra that *does* pay: a
32-wide normalized Hadamard applied before affine quantization lowers the error
from 20.20% to **16.16%** on the router weights and from 16.94% to 16.75% on
ffn_down -- at **zero bits and zero extra bytes** (the transform is free). That
is the entire win of orthogonal transforms, and it is the Hadamard's sign
pattern, not the Clifford structure, that delivers it. The gain tracks block
kurtosis: 20% where blocks are heavy-tailed, a tie where they are Gaussian. It
is already why the qwen35 files ship a fold.

**Verdict.** Clifford algebra has no role as an inference adapter: it cannot
beat a fast factorization of the same operator, its parameterized form is not the
model's operator, and its quantizer loses at equal rate. `kore-clifford` stays as
what it is -- the right language for geometric *modeling* (equivariant nets,
rotor-parameterized layers trained from scratch, new quantizer design), not an
execution optimization. Any future claim in this area should be a measurement of
this shape.

**Incidental finding, worth knowing:** `kore_kernels::hadamard::fwht_block_size`
caps the butterfly at 256 (a GPU shared-memory limit), so
`fwht_inplace(x, 512)` is *two* 256-wide transforms. The fold does not go through
it -- `qwen35::hadamard_t` uses its own `wht_inplace`, which handles any
power-of-two block -- so `hadamard_block = 512` really is a 512-wide transform,
and there is no bug here. The probe uses 256-wide blocks to match the shared
helper; the ratio scales as n log n.

---

## 14.22 Clifford applications after the falsification: the survey (2026-10-18)

§14.21 closed Clifford algebra as a *kernel* and as a *weight quantizer*. The
user's follow-up was explicit: it does not have to live in our engine, so survey
**every** application that could raise tok/s on models discoverable on
Hugging Face. Full write-up: `docs/research/clifford-applications-toks.md`.

**16 application families found; 4 apply post-hoc to an existing checkpoint; 2
have a direct tok/s mechanism.** Everything else is an architecture (needs
training) or a variant that loses at matched bits.

**The two survivors are the same object** -- an orthogonal transform used to make
low-bit quantization survivable:
- weights: rotate offline, fold into `W`, rotate the activation once per GEMV
  (~0.3% against an O(d^2) consumer) -> 4 bit becomes 3 bit at equal quality;
- KV: rotate K and V, per-token scale, 4 bit both -> ~2x KV bytes vs Q8.

**The decisive evidence is post-selection, not a paper.** llama.cpp mainline
ships Hadamard KV rotation **by default** (PR #21038, Apr 2026). KVarN (Hadamard
+ dual-axis variance normalization, k4v2) reports *above*-fp16 throughput at 3-5x
capacity and supports **MLA, hybrid linear-attention, and MTP spec decode** --
our three target mechanisms. Meanwhile RotorQuant/PlanarQuant/IsoQuant live in an
**unmaintained fork** and the TurboQuant PR was **closed unmerged**. The Clifford
rotor was an intermediate that got simplified to 2D/4D, then mainline shipped the
plain Hadamard.

**Arm C added** (`kore-tools/clifford-quant-probe.py`, `report_rotations`), 1 Mi
values per tensor, one rate convention (`bits + 0.125` for an f16 scale per
128-wide block), every transform orthonormal and free:

| rate | scheme | router (kurt +4.77) | ffn_down (kurt +1.76) |
|---|---:|---:|---:|
| 4.125 | AFF128s (none) | 19.26% | 12.16% |
| 4.125 | AFF2p (2-wide) | 17.86% | 12.06% |
| 4.125 | AFF4q (SO(4) quaternion) | 15.65% | 11.93% |
| 4.125 | HQMQ 24x96 (Hurwitz) | 12.82% | 12.55% |
| 4.125 | **AFF128h (128-wide Hadamard)** | **11.43%** | **11.70%** |

Three results:

1. **Rotation width is monotonic, and wide wins on weights.** 41% error cut on
   heavy tails (no rotation -> 128-wide), 19% at 4-wide, 7% at 2-wide; only 4%
   total on Gaussian data. Confirms rotation pays exactly in the kurtosis.
2. **HQMQ's multiplicative codebook is a real win over a *narrow* rotation**:
   at equal locality and rate it beats AFF4q by **18%** on heavy tails (12.82 vs
   15.65). The 24-cell is a good 4-chunk quantizer. This closes the gap arm B
   left open.
3. **...and loses to a *wide* rotation at every rate above 3.4 b/w**:
   rate-matched (AFF128h interpolated log-linearly) it loses 11% at 4.125 and
   18% at 4.875 on routers, 7% and 15% on ffn_down. That is a *covering-radius*
   limit -- the product set is S isometric copies of one 24-point code, so S buys
   packing but the covering constant stays fixed by the 24-cell.

**HQMQ verdict: FALSIFIED as a general inference quantizer**, on two axes --
error (above) and cost (`ceil(log2 24S) ~ 12-14` bits of dependent table lookup
per 4 values, 3-3.5 lookups/element vs one shift+FMA; its own kernel is 2.5x
slower than dense fp16 SDPA).

**Two design rules derived:**

1. **Rotation width is set by the cost of the consumer, not by quality.**
   Weight GEMV is O(d^2) -> full-width costs `log2 d / d ≈ 0.3%`. KV attention
   dot is O(d) per pair -> in-kernel full-width costs `log2 d ≈ 7x` -> use <=4
   (hence quaternion/Givens). **But if the rotation is at *append* time it
   commutes with the attention weighted sum** -- `sum_j a_j (R v_j) = R sum_j a_j v_j`
   -- so rotating V at append and un-rotating the single output, plus rotating q
   once per query, is **free at full width**. Append-time is strictly better than
   the in-kernel compromise the literature is stuck with.
2. **Rotation and per-token scaling are not substitutes.** Rotation fixes
   channel-aligned static outliers; a per-token absmax scale fixes per-token
   dynamic ones. Modern outlier-heavy models need both (SAW-INT4, KVarN k4v2).

**New workstreams (priority stated; nothing renumbered):**

- **P6.6 -- KV: rotate K *and* V, full width, at append; INT4 both.**
  Runs **before P7**. `kore-attention`'s `K8V4` already rotates V at append but
  (a) only V, (b) INT8 K, (c) is not wired into the `qwen35` engine, whose
  `kv_cache_append_q8` is Q8. Needed: rotate K on append + rotate q once, rotate
  V on append + un-rotate the output once, per-token and per-32 scales, and keep
  the **deferred** insert (f16 during prefill, quantize on decode) which removes
  dequant cost from prefill. **Correctness trap: the V inverse rotation is
  mandatory** -- RotorQuant read PPL 15,369 -> 7.05 only after adding it.
  Expected: ~2x KV bytes vs Q8, up to ~1.5x tok/s at 8k on dense models, 2x max
  context; POCKET-35B is expert-bound so capacity-only there.
- **P12 -- offline rotation-assisted re-quantization Q4_K -> Q3_K** (dense models
  only; POCKET is already at 2.6 bits so there is nothing left to take).
  Dequantize -> 128-wide/full-row rotate -> requantize with error feedback ->
  standard K-quant blocks. Gate on **perplexity**, never token ids. ~25% fewer
  weight bytes at ~0.3% runtime cost.
- **P6.6a -- the measurement that decides the width.** Dump real `Kcur`/`Vcur`
  for a 2k prompt and run arm C on them: does append-time full-width K rotation
  beat the in-kernel 4-wide scheme on *our* models? Do this before writing the
  kernel.
- **Not doing:** HQMQ; a wide transform inside the attention kernel; anything
  expecting a Clifford win on MoE bytes (FORGE / ButterflyMoE need training);
  GA kernels (arm A's 113-119x penalty stands).

## 14.23 The MoE step, measured properly (2026-10-18) -- the upload deficit is not uploads

§14.20 claimed uploads were **95%** of the expert loop and that dispatch was not
the wall. Both were artefacts of how it was measured: the "95%" came from a cold
page cache (the 13 GB file was being paged back in), and the dispatch estimate
came from dividing a number that was mostly disk wait.

This section replaces both with an instrumented step. `KORE_MOE_PROFILE=1` now
times four regions per routed FFN -- the `ht` download (**which drains the
queue**, so it contains the layer's GPU work, not just a copy), the host router,
the expert uploads, and the submission of the rest -- and
`KORE_MOE_FORCE_EXPERT=<e>` pins every pick to one expert, which makes the banks
resident after the first layer and so ablates uploads entirely while keeping the
dispatch count identical.

### What the step is actually made of

POCKET-35B-Q2_K, 8 tokens, 32 slots, cool die, interleaved. `disk=0 MB` is read
from `/sys/block/nvme0n1/stat` around the run, so these are all page-cache hits.
GPU clocks during the run: **1530-1575 MHz at ~25 W** -- the card is not
throttled and not busy.

| region | ms/token | share | forced-expert | what it is |
|---|---|---|---|---|
| download | 231 | 46% | 154 | drain of the layer's GPU work + 8 KB readback |
| upload | 159 | 32% | ~2 | 374 MB/token host -> VRAM |
| submit | 71 | 14% | 57 | host command encoding |
| router | 43 | 8% | 41 | F32 router matvec on the host |

`upload` is **not** disk: 374 MB/token at 159 ms is 2.3 GB/s, and the raw path
sustains 3.5-4.0 GB/s back to back (`readback_probe`). The gap is per-call
overhead across ~730 uploads/token, not I/O.

### The router was a real bug, and it is fixed

`gemv_bytes_f32` decoded each element with four bounds-checked byte loads and
folded the result into **one** accumulator. On Comet Lake an f32 add has 4-cycle
latency, so a single chain runs at one MAC per 4 cycles: 3.2 ms per layer for a
2048x256 matvec, i.e. 135 ms/token with the GPU idle.

The rewrite keeps one sequential accumulator **per row** and runs several rows at
once, so no row's sum is reassociated -- which matters because the router output
is a *discrete* decision; a reassociated sum there is not a rounding difference,
it is a different model. `chunks_exact` gives the compiler a statically-sized
window, which removes the four bounds checks per MAC.

    2048x256, warm:  823 us -> 212 us  (3.35x, eight rows interleaved)
    bit-identical to the naive loop, exact `to_bits` equality, tested at
    n = 256 (interleaved path) and n = 7, 6 (remainder path)

Whole step, same conditions:

| | before | after |
|---|---|---|
| base | 607 ms/token (1.54 tok/s) | 505 ms/token (1.81 tok/s) |
| forced expert (no uploads) | 400 ms/token (2.4 tok/s) | 258 ms/token (3.4 tok/s) |
| router region | 135 ms/token | 43 ms/token |

The remaining 43 ms is memory, not arithmetic: it is 80 MB/token of F32 router
weights (2 MB x 40 layers), and the isolated benchmark reads them from a 2 MB
buffer that stays in L2.

### Where the time really goes: dispatches, and a starved GEMV

`submit` is 71 ms/token = 40 layers x ~55 dispatches x **32 us of host
command encoding**. That is the whole region: `record_dispatch` builds a fresh
bind group and writes uniforms per dispatch, and the expert loop issues ~48 of
the 55.

The expert-shaped GEMV is equally far from the metal. Same kernel, same bytes,
only the row count changes (`readback_probe`):

| rows | dispatch | GB/s |
|---|---|---|
| 512 (one expert) | 37.5 us | 9.2 |
| 2048 (four experts) | 48.5 us | 28.4 |
| 4096 (eight experts) | **67.3 us** | **40.9** |
| 16384 | 397.4 us | 27.7 |

Eight experts stacked into *one* dispatch costs 67 us where eight dispatches
cost ~300 us. The 512-row number is bounded below by the ~32 us host encode; the
kernel's own ceiling is the 40 GB/s at n=4096, against 448 GB/s of VRAM. The
shader is `@workgroup_size(128)` with four rows per workgroup, so n=512 is 128
workgroups -- occupancy-starved, one serial 8-block loop per thread, no
split-K.

**So the next lever is dispatch *count*, and it attacks two regions at once.**
Stacking the eight experts' gate and up matrices into one 4096-row dispatch (the
bytes are already contiguous per expert in the file) removes 14 dispatches per
layer: ~450 us of host encoding and most of the kernel latency, per layer. The
`down` matrix cannot be stacked the same way -- each expert needs its own input
-- so it needs either a block-diagonal GEMV or a batched-input variant. That,
plus a split-K reduction for small n, is the P6.7 milestone.

**Corrections to §14.20 to carry forward:**

1. "Uploads are 95%" -- no. Uploads are 32% at 2.3 GB/s with a hot page cache;
   the 95% run was cold.
2. "Dispatch is not the wall" -- it is 14% directly and gates the kernel's
   occupancy. What was measured then was disk wait.
3. "5.6 tok/s warm" -- not reproducible on this build. Interleaved, disk-verified
   runs give 1.8 tok/s base and a 3.4 tok/s no-upload ceiling. The 5.6 figure
   should be treated as withdrawn until it is re-measured.
4. The slot cliff finding stands: 32 -> 64 -> 96 slots moves uploaded bytes from
   2992 to 2673 MB (-11%) and rate not at all. Default stays at 32.

**Also landed here:** `bonsai` accepts `-` for the tokenizer path and reads the
prompt ids from `KORE_IDS`, because this GGUF carries `tokenizer.ggml.tokens`
but not the merges a `tokenizer.json` needs; and `readback_probe` pins the two
host-side ceilings (45-100 us idle readback, 3.5 GB/s raw upload) so future
claims about either can be checked in seconds.

## 14.24 P6.7: the whole top-k in one matvec (2026-10-18, `06b22bc`)

§14.23 ended by naming dispatch count as the next lever and predicting that
stacking the eight experts' gate and up matrices would remove 14 dispatches a
layer. That is what landed -- but not by making the bytes contiguous in the file,
because they are not: the LRU hands back each expert as its **own** slot buffer,
so eight experts are eight allocations.

### The mechanism: gather, then one matvec

Per layer, per token:

```
8 x gemv(gate) + 8 x gemv(up) + 8 x had_pre + 8 x mul_scalar + 8 x gemv(down) + 7 x add
```

becomes

```
16 x copy(344 KB) -> stage, 1 x gemv(4096 rows) x2, 1 x had_pre([8,512]),
1 x mul_rows, 8 x copy(2 KB) -> act row, 8 x gemv(down), 7 x add
```

The copies are the new primitive. `DeviceBackend::copy_buffer_range` records a
real `copy_buffer_to_buffer` into the *same* batch as the dispatch that reads it
-- alternate compute pass / copy commands inside one `CommandEncoder` -- because
`Queue::copy_buffer_to_buffer` would only order against the next submit, which is
too late. `mul_rows` folds the router's eight per-expert weights into one
elementwise dispatch instead of eight `mul_scalar`s.

### Why it is bit-identical, not merely close

The gather is not a reinterpretation of the data, it is the same rows placed
adjacently: for the K-quant families a matrix is `n` rows of whole blocks, so
`k` of them concatenated *is* one `[k*n, blocks]` matrix, and the row a kernel
reads is the row it would have read from the expert's own buffer. Nothing in a
row's reduction depends on `n`. The 4096-row matvec therefore computes exactly
the twelve-hundred-and-eighty dots that eight 512-row matvecs computed, in the
same order, in the same lanes -- it is a **layout** change, so it needs no
tolerance argument and no token-id gate.

Pinned by test rather than argument:
`batched_expert_gather_matches_one_at_a_time` builds four Q4_K experts with a
random payload, GEMVs them separately, gathers them into a stage with
`copy_buffer_range`, GEMVs the stage once, and compares all 32 outputs by
`to_bits`. `mul_rows_matches_per_row_scalar_multiply` does the same for the
per-row scale. POCKET token ids are unchanged across the two paths:
`[561, 3177, 369, 29099, 383, 279, 181474, 10629]`.

`can_batch` is deliberately narrow: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, and
`n_used > 1`. Q2_0 is excluded because `repack_q2_0` pads each expert to a
multiple of four rows, so the concatenation would only line up by luck.
`KORE_MOE_BATCH=0` keeps the per-expert path for A/B.

### Measured

POCKET-35B-Q2_K, 32 slots, cooled die, 8 tokens, interleaved, paired:

| | batched | per-expert |
|---|---|---|
| run set A (3 pairs) | 227.0, 229.9, 230.4 | 251.0, 256.0, 257.0 |
| run set B (2 pairs) | 230.4, 232.9 | 254.2, 250.3 |
| mean | 231.7 | 252.3 |

**1.09x**, and the whole gain is where it was predicted to be:

| region | batched | per-expert |
|---|---|---|
| submit | 88 ms / 8 tok | 300 ms / 8 tok |
| download | 751 ms | 710 ms |
| upload | 800 ms | 806 ms |
| router | 205 ms | 210 ms |

`submit` is 3.4x lower and everything else is flat. Bytes moved are identical --
this is not a bandwidth optimisation, and it is worth saying so because the
absolute rate now sits at ~230-250 ms/token in these runs, well below the 505
ms/token recorded in §14.23. That drop is **not** this change; it is the box
(page cache and thermals) drifting between sessions, which is exactly why every
claim here is an interleaved ratio and never a cross-session absolute.

### Two variants that lost, recorded so they are not rebuilt

1. **Staging the `down` outputs to fold them with one kernel.** Eight 8 KB
   copies plus one block-sum, replacing seven adds -- on paper seven dispatches
   cheaper. Measured: `submit` 88 -> 275 ms / 8 tokens, end to end ~23 ms/token
   *worse*. The copies each force a compute-pass boundary, and that costs more
   than the adds they remove.
2. **Raising the encoder's flush threshold 48 -> 512**, to let the host run
   ahead of the GPU. Measured interleaved: `submit` 280 -> 69 ms / 8 tokens, but
   `download` +210 ms. The sum is flat. The flush wait is **GPU work surfacing
   earlier**, not time the host was wasting -- which also retires the
   "host and GPU do not overlap" framing of §14.23 as a description of a *bug*
   rather than of an accounting boundary.

### What is left in the step, and it is no longer dispatch

`upload` is now the largest host region at ~100 ms/token (44%), then `download`
(the layer's GPU work) at ~90, `router` at ~26, `submit` at ~11.

`down` is deliberately still eight dispatches. One expert's `down` is already
2048 rows, which is past the row count where this kernel reaches its bandwidth
ceiling (28.4 GB/s at 2048 vs 40.9 at 4096, per `readback_probe`), so stacking
would buy dispatch count and nothing else -- and dispatch is no longer the wall.

The next lever is **overlap**, not count: the layer still ends in a routing
readback that drains the queue, so ~100 ms/token of host memcpy happens while the
card is idle. Prefetching layer L+1's experts from token t-1 during layer L's
GPU run is the P6.8 target. `down`'s grouped-input shader stays parked: it would
let one dispatch read a different activation row per expert, which is the last
piece needed if `down` ever becomes worth stacking for a reason other than count.

## 14.25 The profile was measuring the prompt (`8856b83`)

`moe host` divided the run's four counters by the number of tokens *generated*.
On a 12-token prompt against 8 new tokens, that is 20 forwards reported as 8
steps: every region inflated 2.5x, and the whole prefill attributed to the first
generated token. `generate_greedy` now zeroes the counters after the prompt.

The same POCKET run that read **226 ms/token** reads **64 ms/forward**. The
decode rate was never 4 tok/s; it is ~15 tok/s, and the wall of a short run is
mostly prompt processing.

### What a decode step costs now (POCKET-35B-Q2_K, 32 slots, batched path)

| region | ms/forward | share | what it is |
|---|---|---|---|
| download | 23.6 | 37% | the layer's GPU work + 8 KB readback |
| upload | 26.0 | 41% | expert misses, host -> VRAM |
| router | 10.0 | 16% | F32 router matvec on the host |
| submit | 4.2 | 7% | host command encoding |
| **step** | **63.8** | | |

### The uploads are at the ceiling, and always were

`MoeTimes::upload_bytes` now counts the bytes the banks actually moved:
**106 MB/token** (prefill included, 150 MB/forward) against 26 ms -> **4.1 GB/s**.
`readback_probe` measures the raw path at 3.5-4.0 GB/s. So the upload region is
bandwidth-bound with nothing left in it -- the "per-call overhead across ~730
uploads" hypothesis in §14.23 was an artefact of dividing per-token bytes by a
per-forward time. The only lever on this region is **fewer bytes**, and the slot
sweep already showed 32 -> 96 slots buys 11%.

### P6.7, re-measured on the decode step

Interleaved, N=8, two pairs:

| | batched | per-expert |
|---|---|---|
| ms/forward | 64.9, 63.8 | 76.3, 76.6 |
| submit | 4.3 | 14.8 |
| upload | 209 ms/8 tok | 210 ms/8 tok |
| router | 80 ms/8 tok | 81 ms/8 tok |

**1.19x**, not the 1.09x the prompt-diluted wall showed. The regions the change
does not touch are identical to the millisecond, which is the check that the
measurement is now seeing the thing it claims to.

### Consequence for the milestone order

The step is now **host 40 ms vs GPU 24 ms**, and the two do not overlap, because
the chain per layer is `dense GPU -> routing readback -> upload -> FFN GPU` and
each link needs the previous one. Uploads (26 ms) are the largest single host
region and the only one that is pure bandwidth.

That makes **expert prefetch** -- upload layer L+1's predicted experts while
layer L's FFN runs -- the highest-value remaining change: it is the only way to
move the upload bytes off the critical path, and it is worth up to 26 of 64 ms.
It must be measured for prediction accuracy before it is built, since a wrong
prediction spends the same PCIe bandwidth for nothing.

`router` (10 ms) is the second: an F32 matvec whose only alternative is the
device, where the row reduction would have to stay sequential to keep the
discrete top-k unchanged. Parked until prefetch is priced.

## 14.26 Prefetch is refuted before it is built (2026-10-18, `0e4ea1b`)

§14.25 named expert prefetch -- upload layer L+1's predicted experts while layer
L's FFN runs -- as the highest-value remaining change, because the uploads are
26 of 64 ms and sit on the critical path. It also said the prediction rate has to
be measured first. It was.

### The measurement

`PrefetchStats` keeps, per layer, the previous token's pick set, and counts three
things per step: how many of this token's picks repeat it, how many picks are
misses, and how many of those misses the previous set named. `KORE_MOE_PROFILE=1`
prints it (`moe locality:`).

POCKET-35B-Q2_K, 32 slots, 24 tokens, steady state, 960 layer-steps:

| | |
|---|---|
| picks repeating the previous token | **43%** (3.4 of 8) |
| misses per step | **2.83** of 8 |
| misses the previous token's set named | **0%** |

### Why the third number is zero by construction, not by luck

The LRU holds 32 experts per layer and a step loads 2.83. At that fill rate an
expert picked on token *t* is not evicted until roughly `32 / 2.83 ≈ 11` tokens
later. So every expert in token *t*'s pick set is still resident when token
*t+1* routes -- and a *miss* is by definition an expert that is **not** resident.
The two sets cannot intersect. The 43% that do repeat are already hits.

That is a property of LRU-plus-32-slots, not of POCKET's routing, so it does not
change with a different checkpoint: **the LRU already subsumes the previous
token as a predictor.** Prefetching its set would upload bytes that are either
already resident or about to be discarded, spending the same PCIe bandwidth to
duplicate a decision the eviction policy already made.

A predictor that beats the LRU would have to name experts *outside* the recent
window, i.e. predict on content rather than recency. There is no evidence for
that being learnable here and no cheap way to try it, so the milestone is
closed. The upload region's only remaining lever is fewer bytes (P12).

### What replaced it: the router, blocked across threads

With prefetch gone, `router` (10 ms/token, 16%) was the largest removable host
region. It is a 2048x256 f32 matvec: one sequential accumulator chain per row,
12 GB/s, i.e. one core's memory bandwidth.

Rows are independent, so they are cut into blocks across rayon. Each row keeps
its own sequential chain, so *which* thread computes a row cannot enter its
arithmetic -- a scheduling change, and therefore not something that needs a
tolerance argument even for a router whose output is a discrete top-k.

| | isolated (`router_gemv_throughput`) | in situ, ms per 8 tokens |
|---|---|---|
| serial | 172.5 us | 79.9 |
| 4 blocks | | 62.5 |
| 8 / 12 / 24 / 48 | 61.7 us at 12 | 62.6 / 67.8 / 68.3 / 66.4 |

2.8x isolated, **1.28x in situ** (62.55 vs 65.35 ms/forward end to end, 2 pairs).
The gap is the fork/join and the parked-worker wake-up, which the isolated test
does not pay because it runs the same call 200 times back to back and keeps the
threads hot. The block count is therefore sized to *stop* adding threads rather
than to fill them, and `KORE_GEMV_BLOCKS=1` reaches the serial path for an A/B.

Token ids are identical in every arm:
`[561, 3177, 369, 29099, 383, 279, 181474, 10629]`.

## 14.27 Where the decode step stands (2026-10-18)

| region | ms/forward | share | lever |
|---|---|---|---|
| upload | 26.0 | 41% | bytes only -- at 4.1 GB/s, the PCIe ceiling |
| download | 23.6 | 37% | the layer's GPU work |
| router | 8.3 | 13% | done (was 10.0) |
| submit | 4.2 | 7% | done (was 14.8 before P6.7) |
| **step** | **62.2** | | ~16 tok/s decode |

All four regions are now at a floor: uploads at the PCIe ceiling with
unpredictable misses, the router at the shared memory bandwidth with its rows
blocked, submit at the dispatch count after P6.7, and download is the GPU's own
work. The host total (38.5 ms) still exceeds the GPU total (23.6 ms) and the two
do not overlap, because the chain per layer is
`dense GPU -> routing readback -> upload -> FFN GPU` and every link needs the one
before it. Prefetch would have been the way to overlap them; §14.26 closes it.

**There is no remaining internal lever of more than a few percent.** Further
speed has to come from outside the step: fewer expert bytes per token (P12),
a faster dense path (P9/L -- reassociating `dot_wide` for FMA, which needs the
tolerance argument), or a wider step (prefill batching on a routed arch, which
is the real fix for prompt processing and is not a decode-rate change at all).

### The prompt is the real remaining cost

At ~64 ms per forward and one forward per prompt token -- batching is refused on
a routed arch -- prompt processing is ~16 tok/s too. A 1000-token prompt is over
a minute of prefill, and the harness reports ~4 tok/s because a 12-token prompt
is amortised over 8 generated tokens. Batching the prefill is the change that
would matter to a user; it needs the grouped-input GEMV (`base_row`) that the
`down` path did not justify.

---

## 14.28 Coverage: a released Qwen3.6-35B-A3B runs byte-identically

### The file

`bartowski/Qwen_Qwen3.6-35B-A3B-GGUF` / `Qwen_Qwen3.6-35B-A3B-Q2_K_L.gguf`,
**14 003 418 752 B (13.04 GiB)**, sha256 `bcecb058c4b3…34ef` (verified).
`Q2_K_L` over `Q3_K_S` (15.28 GiB) purely on disk: reclaiming `target/debug`
(11 GB of debug-profile artifacts, unused — everything is built `--release`)
took the box from 9.5 GB to 20 GB free, and Q3_K_S would have left ~3.6 GB on a
99%-full root. `Q2_K_L` is also the better coverage pick: same expert width as
`Q2_K` but with the resident tier upgraded, which is where the new code is.

Per-tensor census: `{F32 368, Q8_0 107, Q2_K 152, Q4_K 100, Q3_K 20, Q6_K 4, BF16 2}`.
The types are not decoration:

| tensor group | type | role |
|---|---|---|
| `ffn_gate/up_shexp` (64), `ffn_down_shexp` (21) | **Q8_0** | shared expert, every layer |
| `ssm_out` (16) | **Q8_0** | GDN output projection |
| `attn_k` / `attn_v` | **Q8_0** | full-attention KV projections |
| `output.weight`, `token_embd.weight` | **Q8_0** | LM head + embedding, 540 MB each |
| `attn_qkv` (30), `attn_output` (10) | Q4_K | full-attention |
| `attn_q` | Q6_K (4) / Q2_K (6) | full-attention (gated) |
| `ffn_gate/up_exps` | Q2_K | routed experts |
| `ffn_down_exps` | Q3_K | routed experts |
| `blk.40.*` (2 tensors) | BF16 | MTP block, skipped |

### What had to be built: Q8_0 on the device lane (237f38b)

Q8_0 was the *only* missing piece, and it was not a corner case — it is the
shared-expert FFN, `ssm_out`, the KV projections and the LM head, i.e. the
per-token critical path. `up_w` and `wgemv_at` refused the type outright, so no
released file of this family could load at all. A host-side fallback for
`ssm_out` alone would have cost ~2.8 ms/layer.

The kernel mirrors the K-quant family exactly (`@workgroup_size(128)`, four rows
per workgroup, per-lane accumulators, one 32-lane tree reduction) so its result
is independent of the batch width `m`. Two layout facts drove it: a Q8_0 block is
32 quants (an eighth of a K-quant block), so the lane map is per-quad rather than
per-block; and the file's 34-byte block is not a whole number of words, so
`kore_kernels::q8_0::repack_q8_0` adapts it once at load to 36 bytes with the
quants first — the same kind of upload-time adaptation as `repack_q3_k`, and for
the same reason (Q8_0 here is resident-only, ~300 MB, so a host repack at load is
cheaper than a device repack pass).

### Metadata: no new arch code

Compared key by key against POCKET, the *only* differences are
`block_count` 41 vs 40 and `nextn_predict_layers` 1 vs 0. `n_layers = blocks -
nextn` already dropped the MTP block. `rope.dimension_sections = [11,11,10,0]` is
**identical** to POCKET's — an earlier note that the released file omits it was a
parser artifact (the first header script discarded arrays and read them as
`None`), so the "IMROPE collapses to standard rope" reasoning was never under
threat.

### Result: byte-identical to llama.cpp

```
prompt_ids: 760 6511 314 9338 369            ("The capital of France is")
gen_ids:    11751 13 198 760 6511 314 9338 369
gen_text:   "Paris.\n\nThe capital of France is"
```

Both sides greedy, so the streams are a deterministic function of the weights.
Locked in as `crates/kore-nn/tests/qwen36_real.rs` — guarded on the file's
presence (13 GiB is not vendored), asserts `n_layers == 40` so the MTP skip is
covered too, and runs in 5.7 s from a warm cache.

### The rate, and why §14.27's ~16 tok/s is not what a user gets here

Three regimes, same file, same binary, cooled die, `gpu` lane, 24 tokens unless
noted. `File system inputs` is `/usr/bin/time -v`'s count of blocks actually read
from storage (× 512 B).

| regime | ms/forward | tok/s | real disk reads | expert bytes uploaded |
|---|---|---|---|---|
| forced 1 expert/layer (working set ~45 MB, fully cached) | **26.4** | **16.26** | 52 MB | 1.8 MB/token |
| routed, cache warm (repeat runs) | 89.9 / 100.9 | 7.03 / 5.72 | 858 / 962 MB | **170 MB/token** |
| routed, first touch (cold cache, 16 tok) | 288.2 / 335.0 | 1.90 / 1.77 | **4 550 MB** | 189 MB/token |

The forced-expert row is the engine's own floor with expert I/O removed: 26.4
ms/forward, and 17.5 ms of that (66%) is `download` — 437 µs per layer, a
readback that drains the queue. That round trip is intrinsic to the streaming
design: the host must know the routing before it can upload the experts, so it
has to wait for the router each layer. It is a floor unless routing moves
on-device, which is what streaming deliberately gave up.

**The correction.** §14.27's 62 ms/forward and ~16 tok/s for POCKET were measured
with a *warm page cache*: a 12.9 GB file on a 16 GB box, nothing else competing.
Put the 14.0 GB file next to it and both get evicted. Re-measured in one
interleaved cooled A/B, POCKET reads 335.0 / 288.2 ms/forward (1.77 / 1.90 tok/s)
against the bartowski file's 288.2 / 335.0 (1.90 / 1.77) — indistinguishable.
The accounting confirms it: the cold 16-token run uploaded 3 028 MB of experts
and read 4 550 MB from disk, and 4 550 − 3 028 ≈ 1.5 GB is exactly the one-time
resident tier, so **essentially every expert re-read went to the NVMe, with no
readahead inflation**.

So on this box the 13–14 GB class is larger than **RAM**, not just larger than
VRAM, and the streaming design's implicit assumption — that the OS page cache
absorbs expert re-reads — is false in that regime. Nothing in the engine is
broken; the box is. The honest statement of this milestone:

> A released 13 GiB Qwen3.6-35B-A3B runs on an 8 GB card, byte-identically to
> llama.cpp, at **5.7–7.0 tok/s** in steady state and **1.8 tok/s** cold, against
> an engine floor of **16.3 tok/s** when the expert I/O is taken out.

### Levers the measurement actually supports

1. **Fewer bits per routed expert (P12).** 170 MB/token is 60% of the warm step
   and ~42–56 ms of it is PCIe time at the measured ~3 GB/s. This is the only
   lever that touches that term. Rotation-assisted Q4_K→Q3_K requant is already
   the planned route, and it is perplexity-gated, never token-id-gated.
2. **More slots.** 32 → 96 measured 11% fewer uploaded bytes on POCKET and would
   be worth 11% here; VRAM is 2.8 GB of 8 GB used, so it is free. Default stays
   32 until it is measured on this file.
3. **The layer round trip.** 17.5 of the 26.4 ms no-I/O floor. Removing it means
   device-side top-k, which only pays if routing also moves on-device.
4. **Host-side I/O** is *not* a lever: with the one-time resident load accounted
   for, readahead inflation is ~0, so there is no free bytes there.

Prompt processing is unchanged and still the worst of it: ~26–90 ms per forward
at one token wide, so a 1000-token prompt is minutes. That is the grouped-x GEMV
plus batched GDN in the backlog, and it is what a user would feel first.

## 14.29 P6.6a: the KV rotation is append-time, and 4-wide is not close (2026-10-18)

P6.6a is the measurement that was supposed to run *before* any INT4-KV kernel
work: dump the real `Kcur`/`Vcur` and ask arm C which rotation is worth building.
It is done, and the answer is unambiguous enough to also settle a spec change.

### The instrument

`AttnState` already holds exactly what a quantizer is handed -- post-RoPE keys
and raw values -- so nothing inside `forward` had to change:

- `AttnState::kv()/len()` (`crates/kore-nn/src/qwen35.rs`) exposes the cache;
- `cargo run --release -p kore-nn --example kv_dump -- <model.gguf> <n>` prefills
  a real id stream and writes a `KVDU` file (geometry + `folded` flag travel with
  the data, because two files with the same geometry can sit in different bases);
- `kore-tools/clifford-quant-probe.py --kv <file>` scores the rotation arms on it.

### The subject, and why it transfers

2048 tokens of real prose, tokenized by **the model's own** tokenizer (llama.cpp
`ref_ids`; a proxy model's ids would be the wrong embedding rows), prefilled
through `models/ornith-9b/Ornith-1.0-9B-Q4_K_M.gguf`: `qwen35`, dense, 32 layers,
8 of them full attention, `kv_dim 1024`, 16.7 M values per half.

This is the right regime for the files that matter, and that is a correction to
an earlier assumption in this plan: **the released checkpoint carries no
`prism.hadamard.*`, so Qwen3.6-35B-A3B is *unfolded*.** `folded` is
`meta.contains_key("prism.hadamard.sign_widths")`, and POCKET and the real
bartowski file both fail it. `hadamard_block: 512` in the printed config is a
*default*, not evidence of folding. So the unfolded measurement is not a proxy
for the folded one -- it *is* the case that matters, and only `Ternary-Bonsai-2B` is
folded among the local files.

### The verdict

Ratios are the mean over the 8 layers, all values, at 4 adjustable bits with one
f16 scale per block of the stated width (rate = 4 + 16/width b/w).

**K** (excess kurtosis 2.0 -> 3.2 as the block widens):

| width | b/w | AFF | had | hadD | quat4 | givens2 |
|---|---|---|---|---|---|---|
| 32 | 4.500 | 12.0811% | **9.4387%** | 9.4429% | 11.1038% | 11.5975% |
| 64 | 4.250 | 14.2253% | 10.5639% | 10.5709% | 12.9320% | 13.6025% |
| 128 | 4.125 | 16.5811% | **11.5732%** | 11.5970% | 14.9397% | 15.7853% |
| 256 | 4.062 | 19.1083% | 12.5276% | 12.5522% | 17.1165% | 18.1370% |

**V** (kurtosis 8.9 -> 24.8; layer 6 is 25.92% -> 9.87%, a 2.63x layer):

| width | b/w | AFF | had | hadD | quat4 | givens2 |
|---|---|---|---|---|---|---|
| 32 | 4.500 | 12.0952% | **9.2885%** | 9.2624% | 11.0439% | 11.5855% |
| 64 | 4.250 | 14.3719% | 10.3509% | 10.4027% | 13.0473% | 13.7314% |
| 128 | 4.125 | 16.9278% | **11.4218%** | 11.4624% | 15.2148% | 16.0922% |
| 256 | 4.062 | 19.8276% | 12.4677% | 12.2417% | 17.7296% | 18.7642% |

Four things to take from it.

1. **Append-time full width beats the in-kernel scheme at every width, and the
   margin is not close.** Of the log-error reduction a full-width append-time
   rotation buys, the in-kernel SO(4) captures only **24-34%** and the in-kernel
   pair **12-17%** -- at *every* width:

   | width | append | SO(4) | pair |
   |---|---|---|---|
   | 32 | 100% | 34% (K) / 34% (V) | 17% / 16% |
   | 128 | 100% | 29% / 27% | 14% / 13% |
   | 256 | 100% | 26% / 24% | 12% / 12% |

   This is exactly P6.6a's question, answered: do the append-time route, because
   a 4-wide in-kernel rotation is a 3-4x worse use of the same idea. It is also
   the *cheap* route, for the reason in point 4.

2. **The rotated 128-wide INT4 strictly dominates the unrotated 32-wide INT4** --
   lower error *and* lower rate: 11.5732%/11.4218% at 4.125 b/w against
   12.0811%/12.0952% at 4.500 b/w. The rotation pays for its own scale overhead.
   That is the sentence that justifies P6.6 existing at all rather than a plain
   INT4 cache.

3. **Random signs buy nothing** (`hadD` is `had` within +-1%, and at 256-wide it
   even helps V slightly, 12.2417 vs 12.4677 -- one seed, not a trend). Cheap to
   check and worth checking, because it removes a kernel constraint:
   `hadamard_signed` asserts `block_size == 512` (the shader's workgroup buffers
   are sized for it), while the plain `hadamard.wgsl` supports any `block_size <=
   256` behind its `if (lx < bsize)` guard. So the unsigned transform is the one
   to use, and no signed shader has to be resized.

4. **The rotation is free at append**, and the algebra is worth writing down
   because it is what makes point 1's winner also the cheap one. `H` orthonormal
   means `(R k) . (R q) = k . q`, so K and q rotate together and attention is
   unchanged; and `sum_j a_j (R v_j) = R sum_j a_j v_j`, so V's rotation commutes
   with the weighted sum and only the single layer output has to be rotated back.
   Both are one `O(d log d)` pass per token per layer, independent of sequence
   length. In-kernel rotation is `log2(width)` per key-value pair *at every query*.

### What P6.6 now is, on the existing surface

The revised spec is small, because the primitives already exist:

- **Rotation:** `hadamard_transform` is already block-wise along the last
  dimension and already passes `block_size` as a uniform -- it just *derives* it
  as `min(1 << cols.trailing_zeros(), 256)`, which is 256 for every row here
  (`kv_dim` 1024 and 512, `n_heads*head_dim` 4096). So the one host-side change
  is to make that derivation an argument. No shader change, no resize. `H` is
  symmetric and self-inverse, so the *same* call rotates K/V/q at append and
  un-rotates the V output -- there is no forward/inverse pair to get wrong.
- **Block width: 128**, over the plan's "per-token and per-32 scales". The width
  is really a *rate* knob (a narrower block needs more scales), and 32-wide is
  the lowest-error row of the table. But 128 is the only width that beats the
  unrotated *narrower* block on both axes (point 2), and it halves Q8 KV
  (4.125 vs 8.25 b/w). If a longer context matters more than the last 2 points of
  error, take 32-wide; the rotation gain is larger there anyway (1.28x/1.30x).
- **INT4 with a per-128 f16 scale** replaces `quantize_q8_rows` +
  `kv_cache_append_q8` on this path; the dot dequantizes in kernel as today.

Precedent worth noting: `kore-attention`'s `K8V4` already rotates V *at append*
(`kv_cache.rs:218`) -- the design is not speculative -- but it un-rotates by
rotating the **entire cache** at dequant time (`:280`), `O(seq * d)` where the
correct placement is one `O(d log d)` op on the layer output. So the existing
mode is a proof of concept with the expensive half in the wrong place, which is
what point 4 fixes.

### Caveats, and what is not measured

- This is a **reconstruction** metric. It cannot say "INT4 KV is fine"; it says
  "if INT4 KV, rotate at append at full width". The gate for shipping it stays a
  perplexity comparison against the Q8 cache, never a token-id diff.
- `quat4`/`givens2` are the literature's in-kernel arms (SO(4) and Givens per
  pair), scored with the same block/scale layout, so the comparison is at equal
  rate -- not a straw man with a worse scaler.
- **Not measured:** POCKET (unfolded MoE, `kv_dim 512`, 10 full-attn layers) --
  the CPU prefill is ~35 min for 2k tokens and the geometry differs only in
  `n_kv`. Bonsai (the folded case) was started and then **stopped on purpose**:
  it runs at 0.25 tok/s (the CPU path dequantises Q2_0 per gemv), no file we ship
  is folded, so its result cannot change this decision, and an hour of the box
  is a worse price than that.

## 14.30 P6.6's speed premise is refuted: 8k attention is at 3% of the bandwidth ceiling (2026-10-18)

P6.6a settled *which* rotation to build (§14.29). It did not test the reason the
plan wanted P6.6 at all, which was "up to ~1.5x tok/s at 8k on dense models" --
i.e. that long-context decode attention is limited by KV **bytes**, so halving
them buys time. That premise is worth an hour before a risky attention rewrite,
and `bench_attn` was already the instrument: it prints the achieved `GB/s` of
exactly `2 * seq_k * kv_dim` bytes.

Cooled RTX 2070 Max-Q, Bonsai's full-attention geometry (8 q heads, 2 kv heads,
`head_dim 256`, so `kv_dim 512` -- the same `kv_dim` and `head_dim` as the real
35B's full-attention layers), Q8 KV, medians over 20 reps:

| seq_k | best arm | ms | GB/s | vs single |
|---|---|---|---|---|
| 2048 | 64 splits | 0.312 | 6.7 | 5.3x |
| 4096 | 64 splits | 0.422 | 9.9 | 12.3x |
| 8192 | **64 splits** | **0.632** | **13.3** | **23.5x** |
| 8192 | single | 14.839 | 0.6 | 1.0x |

### The kernel is not byte-bound, and not compute-bound either

13.3 GB/s against a 448 GB/s ceiling is **3% of peak bandwidth**. The matching
compute figure rules out the other explanation: 8192 keys x 8 heads x 256 dim x 2
(MACs per QK and per PV) is 67 MFLOP in 0.632 ms, i.e. **106 GFLOPS, ~1.5% of
fp32 peak**. Neither ceiling is anywhere in sight, so the time is set by
*latency and parallelism* -- which is exactly what the 23.5x from 1 -> 64 shards
says, and why the curve is still improving at the cap.

### Consequence 1: P6.6 is a capacity feature, not a tok/s feature

INT4 KV halves the **bytes** and does neither of the two things this kernel is
short of:

- it does not reduce the **MACs** -- every key still needs a 256-wide dot and a
  256-wide accumulate, at 1.5% of the compute ceiling; and
- it does not reduce the **dependent loads**, which are what the 3% bandwidth
  figure says are being paid for -- and it *adds* unpacking work on the same
  latency chain.

So on this hardware P6.6 cannot speed decode attention up, and can plausibly
slow it down slightly. What it does do is real and worth having: **~2x KV bytes
against Q8, so ~2x context at a fixed card.** P6.6 should be planned, measured
and reported as a *context/capacity* item. Selling it as "1.5x tok/s at 8k"
would have been an unmeasured number that this box does not support, and it
would have been the justification for the single riskiest kernel change on the
list.

### Consequence 2: the parallelism win is already harvested

The engine's policy is `attn_shard_count(seq_k) = ceil(seq_k / 64).clamp(1, 64)`
(`ATTN_KEYS_PER_SHARD = 64`, cap 64), so at 8k it already runs the bench's best
arm. There is no policy factor left to take: what remains is the kernel's *inner
loop* (3% of bandwidth is a latency-hiding problem -- vectorisation, ILP, fewer
dependent steps per key), not its footprint.

### Consequence 3: P6.6 must be opt-in, and that is a correctness decision

The real Qwen3.6-35B-A3B currently runs **byte-identically to llama.cpp** with a
Q8 KV cache (§14.28), and that property is the coverage gate. Rotated INT4 KV
changes the attention numerics, so shipping it as a default would trade a
hard-won verification property for a lever that the measurement above says does
not exist. P6.6 therefore lands behind an explicit KV-mode flag with Q8 as the
default: the file that proves the engine correct keeps proving it.

### Where the attention cost actually sits

0.632 ms/layer at 8k is 6.3 ms of the 35B's 89.9 ms routed step (**7%**), against
expert I/O at 60%. On a dense model with 8 full-attention layers over a ~19 ms
step it is ~5 ms, about **26%**. So inner-loop work on this kernel is a real
lever for *dense* long-context and a small one for the MoE -- the opposite of
what the P6.6 plan entry assumed about where the bytes matter.

Net ranking change since §14.28: expert bytes keep the top spot (170 MB/token,
60% of the step, and no cheap lever touches them); prefill at one token wide
stays the worst thing a user feels; **P6.6 drops from "before P7" to a
capacity-only item**; and the attention inner loop becomes the interesting
long-context lever.

## 14.31 Does a wider prefill chunk amortise expert bytes? Measured: not until it beats the cache (2026-10-18)

§14.28 left prefill at one token wide on a routed arch as the worst thing a user
feels, and the source says why in a comment:

> Each column of a routed FFN picks its own experts, so a four-column chunk would
> not amortise the expert read the way it amortises a resident weight -- **it would
> multiply the traffic instead**.

The second half of that is wrong -- per-token traffic is unchanged by chunking,
and a chunk **deduplicates the union** of its tokens' picks. But "there is a win"
and "the win is worth a large build" are different claims, and the second one had
never been measured. It has now, and the answer is not the one the comment (or the
first reading of the data) suggests.

### The instrument

`KORE_MOE_ROUTE_DUMP=<path>` appends one line of picked expert ids per routed-FFN
call, in call order, on both lanes, and is otherwise free. The layer index is not
recorded because it does not need to be: a token calls the routed FFN exactly once
per routed layer in a fixed order, so the period is recoverable, and recovering it
is also a check.

Subject: the real 14.0 GB `bartowski/Qwen_Qwen3.6-35B-A3B-Q2_K_L`, 361 real prompt
tokens + 8 generated = **369 tokens x 40 layers = 14760 calls**. The period comes
out at **k = 40** -- equal to `n_layers`, so every layer is routed -- by maximising
same-set overlap, and the winning overlap is **2.917 of 8 = 36.5%**, which
independently reproduces the 36-43% repeat rate the prefetch probe measured in
§14.26. Two instruments, built for different questions, agree on the locality.

### The engine's own counters, and a model that reproduces them

`expert bank: 226794 hits, 127446 misses, 46557.5 MB uploaded`. The banks are
per projection, so 127446/3 = **42482 expert misses**, i.e. **115.1 misses/token**,
and 46557.5 MB / 42482 = **1.096 MB per expert** (gate+up+down, Q2_K family).
Per token the cache is paying 126.2 MB against a 320-expert (350 MB) need -- a 64%
hit rate. A plain per-layer LRU with **cap 32** reproduces **42482 misses exactly**,
so the model below can be trusted for combinations the engine cannot run yet.

### The naive reading, and why it is wrong

Union of picks over `m` consecutive tokens, per layer, with no cache at all:

| m | union/token | vs m*n_used | bytes/token if nothing were cached |
|---|---|---|---|
| 1 | 1.000 | 1.00x | 350 MB |
| 2 | 0.816 | 1.23x | 286 MB |
| 4 | 0.667 | 1.50x | 233 MB |
| 8 | 0.533 | 1.88x | 187 MB |
| 16 | 0.402 | 2.49x | 141 MB |
| 32 | 0.280 | 3.58x | 98 MB |
| 64 | 0.184 | 5.42x | 64 MB |

"3.58x at m=32" is the number that would justify the build, and it is **not
available**, because it is measured against a baseline with no cache. The engine
does not have no cache: at m=1 it already uploads 115 of 320 needs. A chunk's union
does not *add* to that saving, it **competes** with it -- both are the same
cross-token overlap, banked by different mechanisms.

### The honest comparison (validated model)

| chunk m | cap 32 | cap 64 | cap 128 |
|---|---|---|---|
| 1 | 1.00x | 2.03x | 4.98x |
| 4 | 0.99x | 2.03x | 4.98x |
| 8 | 0.96x | 2.01x | 4.98x |
| 16 | 1.03x | 1.97x | 4.97x |
| 32 | 1.35x | 1.97x | 4.94x |
| 64 | 1.91x | 2.25x | 4.95x |
| 128 | 2.97x | 3.17x | 5.32x |

Three things fall out, and the first is the important one:

1. **The slot pool is the cheap lever and the chunk is the expensive one.** At
   m=1, widening the pool 32 -> 64 is **2.03x** and 32 -> 128 is **4.98x**, where a
   chunk only reaches 1.35x at m=32 and 1.91x at m=64 -- and the pool is a number,
   while the chunk is a new routed FFN, a grouped GEMM, a grouped-x `down`, and
   `m`-wide recurrent state.
2. **Below m=32 with cap 32, chunking is worse than nothing** (0.96-1.03x). The
   union of 4-16 tokens is no bigger than what the 32-slot pool already holds, so
   the chunk evicts exactly what it just fetched. A chunk only pays once it is
   *wider than the cache*, which is a stricter requirement than "wider than 1".
3. The two levers are substitutes, not multipliers: at cap 64 chunking is flat
   (2.01x -> 1.97x) from m=4 to m=32.

### What this does to the plan

The plan currently records *"32 -> 64 -> 96 slots cuts uploaded bytes 11% and does
not move the rate. Default stays 32."* The model above says 32 -> 96 should be
roughly **3.5x fewer bytes**, not 11%. Those cannot both be true, and the answer
decides something much larger than the pool default: **if the bytes really do drop
~3.5x and the rate still does not move, then expert bytes are not the binding term
of the step at all**, and every "fewer expert bytes" proposal on the list --
including extending P12's rotation-assisted requant to the experts -- is dead on
arrival. That is a one-environment-variable, interleaved, cool-start A/B, and it is
running now; its result goes in §14.32.

Regardless of that result, the code comment at the routed FFN should change: a
wider chunk never "multiplies" traffic, it deduplicates the union -- and the honest
reason to refuse one is that below m~32 the dedup buys nothing the slot pool has
not already banked.

## 14.32 Expert bytes are not the binding term: a 2.1x byte cut buys 8% (2026-10-18)

§14.31 modelled the slot pool and predicted that widening it 32 -> 64 should cut
uploaded expert bytes **2.03x**, and 32 -> 96 **3.5x**. The plan records that 32 ->
96 "cuts uploaded bytes 11% and does not move the rate". Both numbers cannot be
right, and which one is decides whether *any* "fewer expert bytes" work -- the
planned rotation-assisted requant of the experts included -- is worth building.

Interleaved, cool-start A/B, `gpu` lane, real `bartowski` 14.0 GB file.
`File system inputs` is `/usr/bin/time -v` (x 512 B), i.e. blocks actually read
from storage.

| slots | expert bytes uploaded | misses | tok/s | disk read |
|---|---|---|---|---|
| 32 | 4078 MB | 11130 | 3.42, 3.70 | 19 MB, 0 |
| 64 | 2799 MB | 7701 | 3.80, 3.89 | 56 KB, 60 KB |
| 96 | 2797 MB | 7695 | 3.78, 3.83 | 0, 0 |

**A 1.46x cut in expert bytes buys ~8% tok/s**, and 64 -> 96 changes *six* fetches
over the whole run: the pool saturates at 64, which is itself informative -- the
misses left are cold experts, not capacity evictions, or adding 1280 slots would
have found something.

The plan's "11%" is simply wrong, and so is the model's "2.03x": the engine's
cache is not worth a true LRU's 2.03x at the same capacity. (The model's
calibration was against a 369-token run; over a 29-token run the cold experts
dominate and no policy can help. Both numbers are recorded rather than one being
quietly dropped.)

### The profile says why, and it is not the bytes

Same file, N=48, `KORE_MOE_PROFILE=1`, two rounds each, cool-start:

| slots | download | router | upload | submit | ms/gen-token |
|---|---|---|---|---|---|
| 32 | 4180 / 4558 ms | 748 / 666 | 5142 / 4338 | 519 / 457 | 220.6 / 208.7 |
| 64 | 3960 / 4020 ms | 670 / 765 | 3249 / 4539 | 455 / 519 | 173.6 / 205.1 |

Two terms, roughly equal, and only one of them is about bytes:

- **`download` is 39-48% of the host step and does not move with the pool at all**
  (4180 -> 3960, 4558 -> 4020). This is the per-layer readback that lets the host
  route, and it is ~2.3 ms/layer -- 5.2x the 0.44 ms/layer it costs when the
  experts are already resident (§14.28's forced-expert floor). The difference is that in
  the routed case the readback queues behind a layer's worth of pending uploads,
  so this region is measuring the *serialization* of the streaming dependency, not
  the copy. That dependency -- h_L must come back before layer L's experts can be
  chosen -- is exactly what streaming gave up.
- **`upload` is 43-49%**, and a 2.1x byte cut moves it 1.22x (4740 -> 3894 ms
  averaged). Sub-proportional, because on this box the region is not a pure PCIe
  transfer: the 14 GB file does not fit the page cache, and even "warm" runs read
  1-2 GB from the NVMe inside it.

So the byte lever can only ever touch half the step, and it touches that half at
about half efficiency. `router` + `submit` are 13% together and neither is a byte
question.

### Consequence: the lever ranking from §14.28 inverts

1. ~~**Fewer bits per routed expert (P12).**~~ **Demoted.** It was justified by
   "170 MB/token is 60% of the warm step", but the measured byte-to-rate transfer
   is 1.46x -> 8% and 2.1x -> 6%: two independent runs agreeing that the rate is
   near-insensitive to expert bytes. A 2x bit reduction would have to beat a 2.1x
   *structure* reduction that bought 6%, while paying a large quality cost on an
   already 2.5-bit file and building a requant tool. Rotation-assisted requant of
   the experts is not worth starting on this evidence.
2. **The per-layer round trip.** Promoted to first. It is the single largest
   identified term (39-48%) and no byte-reduction touches it. Device-side top-k
   plus a batched readback shrinks the payload, and decoupling the readback from
   the upload queue is what would actually stop the serialization being paid
   twice -- but note that the routing dependency is *not* removable without
   giving up streaming, so the win here is real but bounded.
3. **More slots** stands as measured: 32 -> 64 is a consistent +6-8% (three
   uncooled-ish rounds and two profiled rounds, never negative) for 2.8 GB of the
   5.2 GB of VRAM that is currently idle. The code comment claiming 32 vs 96 is
   "inside the run-to-run noise" should be updated: the *bytes* claim in it is
   wrong (it is ~1.5x, not 11%) and +8% is outside the noise. The default stays
   32 in code until someone decides the VRAM trade is worth it.

### This also closes the prefill-chunk question from §14.31

§14.31 was scoping *"is a wider prefill chunk worth building?"* and answered it as
a pure byte question: don't bother below m=32, and above the pool it is 1.9x at
m=64. That scoping inherited the same wrong premise as everything else here. The
chunk's **only** mechanism is the byte win -- it deduplicates the union of a
chunk's picks -- so it is bounded by the same measured transfer: 2.11x of bytes
bought 6% of rate, so 1.9x of bytes at m=64 should be expected to buy roughly
5-10%, in exchange for a new routed FFN, a grouped GEMM, a grouped-x `down`, and
m-wide recurrent state. **Do not build it**; the scoping question is answered, and
the answer is no. The one part of §14.31 that survives is the code-comment
correction (a chunk never "multiplies" traffic -- it deduplicates, and the honest
reason to refuse one is that below m~32 the dedup buys nothing the slot pool has
not already banked), and the instrument itself, which is worth keeping because it
is now the cheapest way to ask routing questions without a GPU.

The honest summary of the 13-14 GB class on this box: **it is larger than RAM, not
just larger than VRAM**, so expert I/O is unavoidable; but making that I/O
*denser* does not make it faster, because what is being paid for is a host round
trip per layer plus reads the page cache cannot absorb. The next real lever is the
round trip.

## 14.33 P5d re-measured on a quiet die: the harness had a cold-start bias, and the 8 GB wall is a VRAM cliff (2026-10-18)

P5d (rewind + prompt-lookup spec decode) was measured end-to-end on the dense 9B
before the coverage work started. Those numbers did not survive a quiet box. Two
things were wrong; both are now fixed or characterised.

### The harness bug: cold-start order bias

`bench_draft` rotates which arm runs first each round (`first = round % n_arms`).
That is only fair if the GPU is warm. Pipelines are compiled lazily on first use,
so the arm that runs first in round 0 pays a one-time cost for the whole run --
and with `rounds < n_arms` the rotation never gives that arm a turn behind a warm
GPU.

The symptom was arithmetically impossible: policy `k=3 n>=16` on a prompt the
drafter cannot match fired **0.0%** of the time and was reported **1.201x faster
than plain greedy**. At 0% fire the spec path does strictly more work per token
than greedy (a match scan plus the same forward), so any ratio above 1.0 is an
artefact -- and so was most of the reported speedup.

Fix (in `crates/kore-cli/examples/bench_draft.rs`): warm *every* arm with a full
`n_new` pass before timing any of them, so the same long-context shard counts and
the same `Head::Last` width are compiled too. The arms themselves are unchanged.

What the fix alone moved, Bonsai-2B Q2_0, 5-token prompt, n=64, 2 rounds:

| | before | after |
|---|---|---|
| greedy | 69.2 tok/s | **109.4 tok/s** |

Greedy alone moved 1.58x. The cold start was ~340 ms once, spread over 64 tokens,
and it landed entirely on the arm that happened to run first.

### The corrected end-to-end numbers

Cooled idle RTX 2070 Max-Q, Vulkan, all arms interleaved in one process, warmed
harness. Every run asserted its arms byte-identical to greedy (the harness exits 1
otherwise).

| subject | prompt | n | greedy | `k=3 n>=4` | `k=3 n>=16` |
|---|---|---|---|---|---|
| Bonsai-2B Q2_0 | 5 tok | 64 | 9.142 ms / 109.4 tok/s | 6.325 ms / 158.1 / **1.445x** / fire 71.4% | 6.972 ms / 143.4 / 1.311x / fire 40.0% |
| Ornith-9B Q4_K_M | 5 tok | 64 | 46.983 ms / 21.3 tok/s | 29.496 ms / 33.9 / **1.593x** / fire 71.4% | 33.242 ms / 30.1 / 1.413x / fire 40.0% |
| Ornith-9B Q4_K_M | `def fibonacci(n):` | 64 | 48.838 ms / 20.5 tok/s | 50.074 ms / 20.0 / **0.975x** / fire 1.6% | 49.378 ms / 20.3 / 0.989x / fire 0.0% |

Acceptance was 100% on every arm that fired; the acceptance on the code prompt is
undefined because essentially nothing fired.

Four readings:

1. **The honest payoff is 1.31-1.59x**, not the 1.91-2.24x the unwarmed harness
   reported (the same Ornith run read 2.241x before the fix -- a 1.41x inflation,
   all of it on the baseline arm).
2. **The no-fire floor is 0.975-0.989x**, which is P8/I answered and is what it
   should be: on a prompt with no exploitable repetition, spec decode costs
   1.1-2.5% and buys nothing. That floor is the number the payoff has to beat.
3. **`3:16` is not a no-fire arm.** On echo-rich prompts it fires 40%, because a
   16-token match against a short degenerate history really is available.
   `min_match` is clamped to `max_match = 16`, so no arm can be made that never
   fires; the floor has to be measured on a prompt with nothing to match, as above.
4. **The degenerate prompt is not a proxy.** Both fast prompts are echo cases
   ("...France is" -> "Paris.\nThe capital of France is" is partly a copy of the
   context). A representative prose continuation **cannot be measured on this box
   at all** -- it OOMs, see below -- which is the single clearest statement of what
   the 8 GB card is holding back.

### The 351 ms/token claim is retracted

An earlier note recorded Ornith-9B Q4_K_M greedy at 351-395 ms/token (2.8 tok/s).
On a quiet cooled box the same binary and the same file give **47-61 ms/token
(16-21 tok/s)** -- a 6-7x difference. The plan's own protocol forbids the
measurement that produced the slow number (cool die, interleaved A/B); those runs
were taken while the box was busy with MoE A/Bs and while 27 GB of GGUF in `/tmp`
was fighting the page cache. The number is retracted, and the wide effective-
bandwidth gap it implied was an artefact of it.

### The 8 GB wall is a VRAM cliff, not a leak

`bench_draft` cannot run a representative prompt on this card. The accounting:

- **4,997 MiB of weights** are uploaded: Ornith's tensors sum to 5,690 MB, and
  `token_embd` (572 MB, Q4_K) stays on the host because embedding is a host-side
  row dequant. `output.weight` (834 MB, Q6_K) is the head and is uploaded.
- the `nvidia-smi`-visible upload ramp tops out near **4,691 MiB**;
- **after** the upload finishes, the footprint rises a further **~2.4 GiB** during
  generation setup, in ~260 MiB and ~772 MiB steps, to a **7,231 MiB plateau /
  7,366 MiB peak**.

Less than 1 GiB of headroom is left, so the cliff is prompt-shaped:

| prompt | n_new | max_ctx | result |
|---|---|---|---|
| 5 tok | 256 | 325 | OK |
| 21 tok | 1 | 86 | OK |
| 21 tok | 32 | 117 | **OOM** |
| 42 tok | 1 | 107 | **OOM** |
| 42 tok | 64 | 170 | **OOM** |

Two hypotheses are already ruled out so the next agent does not redo them:

- **not** allocation granularity: `alloc_storage_buffer` allocates exact sizes
  (4-byte aligned) from a size-keyed pool, and buffers are created
  `STORAGE|COPY_SRC|COPY_DST` with no mapped staging mirror;
- **not** a per-generation leak: on Bonsai the peak is bit-identical (1039 MiB) at
  4 and at 18 generations of the same workload.

So the ~2.4 GiB allocated during generation setup is unidentified and is a
first-order item -- on 16 GB it is a curiosity, on 8 GB it is the wall.

### What this changes

- **P8/I is answered:** the no-fire floor is 0.975-0.989x.
- The spec-decode *design* is sound and byte-identical; the *published payoff* was
  inflated and is now ~1.4x on repetition-rich text, ~1.0x otherwise.
- `k` and `min_match` should be re-tuned on the warmed harness before any further
  spec work: the plan's optimum (`k=3, min_match=4`, from `draft_headroom`) was
  selected against the biased ratio.
