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

**Implication (superseded by the follow-up below):** Phase 1 (paged KV) still
matters for long context, but for short-context tok/s the next lever was
believed to be **fusing the small latency-bound dispatches** (e.g.
rms_norm+hadamard, and epilogue folds into the GEMV), not further Q2_0 GEMV
work. That belief was then tested and **disproved** — see the next section.

### Phase 0.5 follow-up: dispatch fusion is measurably neutral

The prediction above was tested directly and **did not hold**. Shipped:

- **`add_rms_norm` fold** — each residual add is absorbed into the norm that
  consumes it; the FFN's trailing add is carried forward as a *pending*
  residual. Removes 48 dispatches/token.
- **`hadamard_signed` rewrite** — double-buffered, 9 unrolled stages, one
  barrier per stage, no dynamically-indexed register array (that indexing
  spilled to local memory). Bit-identical.
- **`had_pre.wgsl`** — a fused "pre-op + Hadamard" kernel with four modes
  (`add+rms_norm+had`, `rms_norm+had`, `swiglu+had`, `sigmoid_mul+had`). It
  replaces **79 dispatches/token** with 79 fused ones — the entire norm→rotate→
  mat-vec prefix of every block, plus the head. Bit-identical to running the
  standalone kernels in sequence (`test_vulkan_had_pre_matches_standalone_kernels`
  asserts exact equality, not epsilon parity).

Measured with the order-rotating interleaved harness (6 rounds, 80 W limit,
AC):

| build | median tok/s | min tok/s |
|---|---|---|
| baseline (pre-fold) | 56.77 | 56.01 |
| + add_rms_norm fold + hadamard rewrite | 58.20 | 57.12 |
| + `had_pre` fusion (−79 dispatches) | **58.16** | **57.14** |

**The fusion is exactly neutral.** That falsifies the "~300 small dispatches
are launch-bound" reading of the cost model: removing a dispatch while *keeping
its work* saves nothing. A related `bench_chain` measurement agrees — removing
a hadamard dispatch entirely (work *and* launch) is worth ~12.5 µs, but that
12.5 µs is the kernel's own execution latency, not launch overhead. The
per-dispatch launch cost here is ~0 because the decode loop submits 48
dispatches per command buffer and the GPU never drains between them.

So: **the decode loop is bound by the serial execution latency of its kernels,
not by dispatch count.** Fusion is kept anyway — it is verified bit-identical,
cuts CPU-side encoding and VRAM churn by 25%, and pays off whenever the host
becomes the bottleneck (larger batches, weaker CPUs, MoE offload) — but it is
not a tok/s lever and should not be budgeted as one.

### Q2_0 GEMV: re-diagnosed, still ~12% headroom

`bench_gemv` shows Q2_0 pinned at **56–62 GB/s on every shape** (small, mid,
large, big-k) while Q4_K reaches 130–160 GB/s on the *same* shapes. Three
diagnostics (patched shader, timed, reverted):

| variant | emb_head time | conclusion |
|---|---|---|
| as shipped | 2.34 ms | — |
| code load + 4× `i32→f32` extraction removed | 2.29 ms | **not** unpack-ALU-bound |
| k-loop `x` load removed | 2.51 ms | **not** x-traffic-bound |
| *all* k-loop global loads removed | 1.79 ms | 76% is instruction latency |
| k-loop `break` removed (enable unrolling) | 2.66 ms | worse — register pressure |

The right metric is **weights/s, not bytes/s**: Q2_0 achieves ~206 Gweights/s
against Q4_K's ~231 Gweights/s on the same warp-per-row structure. A 2-bit
kernel *should* be cheaper per weight, so the residual gap is real but small —
consistent with the earlier "within ~15% of `mul_mat_vec_q`" finding. With all
memory access removed the kernel still spends 1.79 ms of 2.34 ms, i.e. it is
latency/occupancy-bound on its own dependency chain; that is a deep rewrite
(more accumulator sets + real unrolling + occupancy work) for a ≤12% prize, so
it stays closed.


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

## Phase 1a status: long-context decode FIXED by split-KV sharding (2026-10-17)

The Phase 1 diagnosis below — that the contiguous KV tensor (F1) and the O(seq_k)
read volume (F7) are the binding constraints — is **wrong**, and this section
supersedes it. The KV representation was already fixed by the Q8 path; what was
actually binding was **how few workgroups the decode kernel launched**.

**Instrument.** `cargo run --release -p kore-vulkan --example bench_attn` times
`fused_attention_decode_gqa_q8` against `seq_k` at Bonsai's full-attention shape
(8 query heads, 2 KV heads, head_dim 256). All arms are warmed up and then timed
in the *same* loop with the arm order rotated per repetition, and reported as
medians.

| seq_k | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|
| single-group ms | 0.338 | 0.606 | 1.117 | 1.232 | 2.778 | 6.369 |
| effective GB/s | 0.8 | 0.9 | 0.9 | 1.7 | 1.5 | 1.3 |

That is 1–1.7 GB/s on a memory system that does 130–160 GB/s in the GEMV. The
kernel is **memory-latency-bound**, not bandwidth-bound: the F32 kernel reads 4x
more bytes and is only 1.9x slower, which is the signature of a fixed latency
dominating. The cause is structural: at decode the grid is
`(num_heads x seq_q) = (8 x 1)` — **8 workgroups of 64 threads to cover the whole
GPU**, each walking the entire window with no way to hide a ~500-cycle DRAM
latency. (The K read is also only 8 bytes per cache line per instruction, but the
grid is the dominant term.)

**Fix.** Shard the KV window across `n_splits` workgroups per head and merge the
shards log-sum-exp style:
- `flash_decode_gqa_q8_split.wgsl` — grid `(num_heads, n_splits, 1)`. Each shard
  computes its own (running max, sumexp, V accumulator) over its slice of
  `[0, seq_k)`. Arithmetic per shard is identical to the single-group kernel; the
  only change is that the running max is the shard's own max.
- `flash_decode_gqa_q8_combine.wgsl` — one workgroup of 256 threads per head
  rescales each shard by `exp(shard_max - global_max)` and normalizes.
- `n_splits = ceil(seq_k / 64)`, capped at 64; below `seq_k = 256` the merge pass
  costs more than the sharding saves, so the single-group kernel is kept
  (`KORE_ATTN_SPLIT_MIN` overrides the threshold; set it above the context length
  to disable sharding entirely — that switch is the A/B control). Sharding is an
  optimisation only: `attn_can_shard()` falls back for shapes the shard kernel
  cannot stage.

**Result (bench_attn, same bytes, `max |diff|` vs the single-group kernel):**

| seq_k | single ms | split ms | n_splits | speedup | GB/s | max \|diff\| |
|---|---|---|---|---|---|---|
| 512 | 0.606 | 0.132 | 32 | 4.6x | 4.0 | 4.5e-8 |
| 2048 | 1.232 | 0.139 | 32 | 8.8x | 15.0 | 3.5e-8 |
| 4096 | 2.778 | 0.192 | 64 | 14.5x | 21.9 | 3.3e-8 |
| 8192 | 6.369 | 0.302 | 64 | 21.1x | 27.8 | 3.5e-8 |

*Caveat on how this table was first produced:* the single-group numbers were
originally taken from a separate timing pass earlier in the process, which
under-measured them by 1.8x at small `seq_k` because an idle RTX 2070 sits at
300 MHz and spends the first pass ramping to boost. The example now warms every
arm and interleaves them in one loop.

**End to end.** `bench_context` samples decode *immediately after* the prefill, so
its ctx-5 arm runs on a cold die and its ctx-8192 arm on a heat-soaked one: it
conflates context with heat. `crates/kore-cli/examples/bench_context_cool.rs`
prefills, cools the die to a target temperature, runs 8 untimed forwards to
escape the 300 MHz idle clock, then times a 40-token window:

| decode ctx | 13 | 1032 | 2056 | 4104 | 8200 |
|---|---|---|---|---|---|
| sharded ms | 14.80 | 15.05 | 15.11 | 15.34 | 16.17 |
| sharded tok/s | 67.6 | 66.4 | 66.2 | 65.2 | 61.9 |
| unsharded ms | — | 19.29 | — | 33.48 | 52.63 |
| unsharded tok/s | — | 51.8 | — | 29.9 | 19.0 |
| **sharding speedup** | — | **1.28x** | — | **2.18x** | **3.26x** |

Prefill goes through the same one-token-at-a-time path, so prompt throughput *is*
decode throughput, and it gains the same way:

| prefill ctx | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| sharded tok/s | 64.2 | 63.0 | 57.8 | 40.4 |
| unsharded tok/s | 58.0 | — | 33.1 | 12.8 |
| speedup | 1.11x | — | 1.75x | **3.15x** |

The curve is now flat: 14.8 → 16.2 ms (+9%) from ctx 13 to 8200. On heat alone
the old hot ctx-4104 reading was 1.22x the cool unsharded one (40.8 vs 33.5 ms),
so the earlier "2.08x end to end" was a mix of the sharding win and that heat.

**Token ids are byte-identical** between the sharded and unsharded arms and across
contexts. The shard merge changes the softmax reduction order by ~3e-8, which
does not perturb argmax.

**What this means for the plan:**
- Paged KV (F1) is still wanted, but for **memory management** — a soft `max_ctx`,
  no full-length preallocation, offloadable blocks — not for tok/s. Do not
  budget it as a decode-speed lever.
- F7's "fuse the attention decode into the KV append to make the per-token read
  O(1) per block" is **not** the win either; the reads were never the problem.
- The remaining lever on this kernel is the read pattern itself (8 useful bytes
  per cache line per instruction), worth up to ~6x more (28 GB/s → >100 GB/s) and
  the thing that would matter at 32K–262K context. That is a KV *layout* change,
  not a paging change.
- **Measurement caveat.** All long-context numbers must be cool-die and
  clock-ramped. A 4096-token prefill heat-soaks this GPU for minutes, and the
  first forwards on a cooled die measure the 300 MHz → boost ramp (81 ms/token
  for the first four vs 15.5 ms for the last four). Two earlier A/B attempts were
  garbage because of this: arm readings swung 6.5–20 tok/s inside one run. Use
  `bench_context_cool`.

---

## Phase 1 — paged KV + the information-flow fix — SUPERSEDED, see Phase 1a above

**Kept for the record. Two of its three premises were measured and refuted; do not
implement this as written.** (1) The contiguous F32 KV tensor was already replaced
by the Q8 path and is not the constraint. (2) The read *volume* is not the
constraint — the kernel was latency-bound on 8 workgroups, now fixed by sharding.
(3) F7's "amortize the per-token read to O(1) per block" is not a win: the window
must still be read every step. Paged KV remains worth doing for **memory
management** only.

**Goal (restated after Phase 1a):** decode at long context is no longer the
problem — it is flat to ctx 8192. The remaining problem is that **prompt
processing runs at decode speed** (see Phase 2).

Takeaways from the audit:
- **F1 (stock):** KC/VC as one contiguous F32 tensor is the binding constraint — replace with paged KV blocks (SOAR-style ring of fixed-size blocks, each block owned by one layer). This also makes the "max_ctx" a soft budget, not a hard OOM.
- **F2/F4 (info flow):** GPU decode reads INT8 KV (+ parallel F32 scales) instead of full-resolution floats. `kv_cache_append_q8` writes U8 rows + per-token scales; `fused_attention_decode_gqa_q8` reads U8 + scales and dequantizes per-thread in `flash_decode_gqa_q8.wgsl`. This is the Strata/sudoingX mechanism (1.58-bit ternary / INT8 KV + inline dequant).
- **F7 (rule):** fusing the attention decode into the KV append amortizes the per-token VRAM read to O(1) per block rather than O(seq_k).

1. In `kore-vulkan`: introduce paged KV blocks (fixed-size `[block_ctx, kv_dim]` per layer, free-list allocator; `max_ctx` becomes the number of blocks × block_ctx).
2. Rewrite `fused_attention_decode_gqa_q8` to: for the decode window, read only the current block's KC/VC (U8 + scales), dequantize per-thread, run the softmax + V-accumulate, and **return already-accumulated hidden states** so the next step's attention reads a bounded window (not the whole KV). Reuse the flash fuses so no extra passes.
3. Keep `fused_attention_decode_gqa` (F32) as the slow path only while the transition is verified.
4. **Verification gate (rewritten after Phase 1a):** decode at ctx 4096 must not
   regress against the split-KV path (65.2 tok/s sharded / 29.9 unsharded, cool
   die), and token ids must stay byte-identical to the pre-change reference.

---

## Phase 2 — throughput, only after Phase 1 is green — REVISED

**Goal:** the measured bottleneck. Cold-die ctx-13 decode is **14.55 ms/token**, of
which ~10–11 ms is the Q2_0 GEMVs and ~3.5–4 ms is ~150 small latency-bound
dispatches. And **prefill is one token at a time**, so a 4096-token prompt takes
71 s (57.8 tok/s) — the same speed as decode, which is the real usability limit.

Measured decomposition (see the plan's P1b entry) and the levers, in order of
confidence:

1. **GEMV inner loop (~70% of the token).** `bench_gemv` gives Q2_0 193 Gweights/s
   on the big shapes against Q4_K's 240 on identical shapes *and* identical grids.
   The GB/s gap (54 vs 135) is a mirage: Q2_0 reads half the bytes per weight.
   The split-K/occupancy hypothesis is **refuted** — `emb_head` launches 62080
   warps. The ~20% gap is unpack work per weight (4 weights/byte), so the fix is
   fewer instructions per weight (a `dot()`-based unpack, a wider per-lane tile),
   not more parallelism.
2. **Batched/chunked prefill (the usability lever).** Needs a chunked parallel
   scan for the 18 DeltaNet/GDN layers plus batched GEMVs for the 6 full-attention
   layers — a large project, not a quick win. The GEMV kernel already supports
   `m = 2..4`.
3. **Continuous batching / multi-token decode** — note 2: the GDN recurrence is
   what blocks batching across tokens.
4. `argmax` fusion, KV pre-allocation reuse — small.

Takeaways from the original audit, for the record:
- **F9 (buffers):** continuous/mtp batching so the decode reads the KV in a
  coalesced async pattern (already "in theory" for this workload; but see 2).
- **F6/F7:** if the remaining decode steps are still compute-light, look at argmax
  fusion + KV pre-allocation reuse.

---

## Rollback policy (no parallel systems)

- **Gates are no-regression *ratios* under identical conditions, not absolute
  tok/s.** The old "short-context ≥ 80 tok/s floor" is retired: on this laptop the
  GPU is pinned to 30 W on battery (80 W on AC), boost clocks drift, and a
  heat-soaked die reports the same binary 2x slower. A phase passes only if
  `bench_context_cool` (cool die, clock-ramped, medians) or an interleaved
  order-rotating A/B shows no regression, and tok/s at a fixed context does not
  fall. Revert to the last green commit on any regression.
- The map and tracker (`docs/wayfinder/`) are the source of truth for what is green vs. in-flight; log each gate as `PASSED / BLOCKED (reason)` with the measured numbers.
- Do not commit a half-built representation (dependencies between `kore-nn`, `kore-vulkan`, and the `Qwen35Gpu` struct are tight but not yet verified) — finish Phase 0 first, then Phase 1.

---

## Next decision (gated)

**Where we are.** Phase 0 is PASSED. Phase 0.5's GEMV work is done *and* its
follow-up (dispatch fusion) has now been measured and closed out as neutral:
short context is **58.2 tok/s** (median of 6, rotating harness, AC/80 W) against
56.8 for the pre-fusion baseline, and the `had_pre` fusion that removes 79 of
~310 dispatches per token changes nothing. Both the GEMV and dispatch count are
now measured dead ends for short-context tok/s.

**What that leaves.** Two levers that are actually large:

1. **Phase 1 (paged KV)** — the long-context bottleneck (F1–F4): decode reads
   grow as O(seq_k), which is what takes 4096-context from ~59 to ~18.7 tok/s.
   This is the *product* goal (run big models on small hardware), and it is a
   structural change, not a micro-optimisation.
2. **M2 (MoE expert offload)** — weights ≫ VRAM streamed from system RAM while
   the GPU computes hot experts. This is the only shape that "punches above the
   weight class" on an 8 GB card: a 27B dense ternary (M1) cannot fit, but a
   ~30 B MoE with 2–3 B active params can run at 40–60 tok/s here.

**Note on the retired gate.** The "≥ 80 tok/s" short-context floor appears
throughout the phases below but does **not** reproduce on this machine: HEAD
itself measures 33.7 tok/s @ctx5. It is retired as an absolute gate — gates must
be expressed as no-regression *ratios* measured under identical conditions
(same prompt, 1 warmup + 6 timed runs, clocks recorded), because the pinned
curve was taken under different power/thermal conditions.

**Note on measurement validity.** Micro-optimisation work needs AC power. On
battery this laptop pins the GPU to a 30 W power limit (default 80 W) and
tok/s drifts 57 → 21 across a session; only interleaved, order-rotating A/B is
valid then. The fusion result above was taken while charging at the 80 W limit.

Proposed order:

1. **M2 (MoE expert offload)** — ahead of M1, because it is the only milestone
   that fits the hardware and it reuses the offload/bandwidth machinery the
   engine is being built for. Entry model: Strata Coder (IQ1_M, 29.6 GB) on the
   32 GB RAM box.
2. **Phase 1 (paged KV)** — parallel track; it is what makes long context and
   offloaded-KV MoE viable, and it is the concrete fix for the 4096-context
   collapse.
3. **M1** (27B ternary dense) — only if a leaner representation lands; it is
   INFEASIBLE at ~7.6 GB of weights on this 8 GB card.

Closed out (do not re-open without new evidence):

- **Dispatch fusion** — neutral (this section). Kept in the tree because it is
  bit-identical and cuts CPU encoding/VRAM churn 25%, but not a tok/s lever.
- **Q2_0 GEMV micro-optimisation** — ~206 vs Q4_K's ~231 Gweights/s; the kernel
  spends 76% of its time with *zero* memory access, i.e. it is bound by its own
  instruction latency/occupancy. ≤12% prize for a deep rewrite.


