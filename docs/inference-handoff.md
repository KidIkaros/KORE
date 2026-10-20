# KORE inference engine — handoff

**Read this first.** It is the entry point for whoever picks up the inference-engine
work on the next machine. The running lab notebook is
[`docs/kore-inference-revamp.md`](kore-inference-revamp.md) — the authoritative plan,
§0–§14.33, ~3000 lines. This file is the map and the to-do list, not a replacement
for it: when the two disagree, the plan is right and this file is stale.

Written 2026-10-18, on the box the work outgrew: **RTX 2070 Max-Q, 8 GB VRAM,
16 GB RAM, i7-10750H**. The next box is an **AMD card with 16 GB VRAM**.

---

## 1. The mission, and the rules that are not negotiable

**Mission.** Run models far larger than the available VRAM — including MoE — at high
tokens/sec, hardware-agnostically, in pure Rust.

**The bar is not "faster than our last build".** It is *running big models on small
hardware at high tok/s*. A change that speeds up a small model on a big card is not
progress. Sections §14.30 and §14.32 in the plan are the record of two levers that
looked like tok/s wins and measured as nothing — read them before optimizing
anything.

**Four rules that were each learned the hard way:**

1. **Byte-identity is the completion gate, per batch shape.** A capability is done
   only when its output is byte-identical to the pre-change implementation *across
   the batch shapes actually used*. Speculative decode is lossless only because the
   verification pass rewinds any column the batched argmax did not reproduce. The
   harnesses exit 1 on divergence; do not relax that.

2. **A reassociated summation is not gated on a token-id diff.** Exposed FMA
   reassociation (P9/L, P9/K) changes low bits. The stated gate is *fewer exposed µs
   in `bench_chain`* **and** a written tolerance argument. "The tokens still match"
   is not evidence, because at those magnitudes they usually still do.

3. **The measurement protocol is part of the result** (§9 below). Two of the biggest
   numbers in this effort's history were retracted because of measurement, not code:
   a 351 ms/token decode that was really 47–61, and a 1.41×-inflated spec-decode
   ratio. Both are corrected in §14.33 of the plan.

4. **Do not delete or "clean up" things you did not personally create.** Models,
   `/tmp/opencode/*` checkpoints, and unrelated user data stay.

---

## 2. What the engine can do today

Everything below is on `main`, byte-identical to its tests, and runs on
**wgpu/Vulkan** — which is the point: no CUDA, no ROCm-specific code, no vendor
kernels. That base is deliberate (`§"CUDA: later"`) and the AMD box does not need to
change it.

**Working, end to end:**

| Capability | Where | State |
|---|---|---|
| GGUF load (K-quants, Q2_0…Q8_0, F32/F16 first-class) | `kore-nn/src/qwen35.rs`, `kore-kernels/` | done |
| Dense decode (CPU + GPU lanes) | `qwen35.rs` | done |
| Batched decode, up to 4 columns | `forward_batch`, `MAX_BATCH_COLS = 4` (`qwen35.rs:131`) | done, capped |
| Chunked prefill | `prefill` (`qwen35.rs:3006`) | done |
| Split-KV long-context decode attention | `kore-vulkan/src/backend.rs` | done |
| Q2_0 / Q4_K / Q5_K / Q6_K / Q8_0 GPU GEMV | `backend.rs` | done |
| MoE engine (host routing, promote-on-miss expert cache, LRU) | `qwen35.rs` `moe_ffn_gpu` (`:2577`), `DeviceBank` (`:1954`) | done, v1 |
| 100B-class MoE coverage gate vs llama.cpp | `kore-nn/tests/qwen36_real.rs` | done |
| Speculative decode: n-gram/suffix draft, chain-only | `spec.rs` | done, byte-identical |
| Rewind engine + per-row Delta-state snapshots | `new_rewind`/`rewind` (`:3040`,`:3097`) | done |
| P6.6a: rotated KV quant | KV dump + probe | done, **opt-in** (capacity-only) |

**Not built, on purpose:** tree/beam drafting, CUDA backend, expert requant (P12),
prefill batching beyond the MoE boundary. See §8.

**The two most useful files to read first:** `crates/kore-nn/src/qwen35.rs` (the
whole model, both lanes) and `crates/kore-vulkan/src/backend.rs` (the whole Vulkan
lane). `crates/kore-nn/src/k_quant.rs` is the *written spec* for quant layouts.

---

## 3. How the engine is verified

Three independent oracles, and the reason to keep all three:

| Oracle | Gates | Where |
|---|---|---|
| **llama.cpp** (CPU, token stream) | GGUF *semantics* — arch key, tensor roles, tied/untied, tokenizer | `kore-tools/llama-oracle/` (newly committed here) |
| **`gguf-py`'s `gguf.quants`** | quant *layout* — byte-for-byte dequant | `crates/kore-kernels/tests/k_quant_fixtures.rs` |
| **`verify_m`** | the engine against *itself* across batch widths | `kore-cli/examples/verify_m.rs` |

The llama.cpp oracle is machine-local by nature (it needs a built llama.cpp), so it
is now committed as a reproducible tool — `kore-tools/llama-oracle/README.md` has
the two commands. **Rebuild it on the new box first**: nothing else can tell you
whether a token disagreement is your bug or the file's semantics.

In-tree test suites: `crates/kore-vulkan/tests/gpu_ops_tests.rs`,
`crates/kore-nn/tests/{gpu_k_quants,llama_tests,qwen36_real}.rs`,
`crates/kore-kernels/tests/k_quant_fixtures.rs`. The 100B-class gate
(`qwen36_real.rs`) skips itself when the 13 GiB checkpoint is absent — keep that
pattern for the next arch.

> **This box can no longer run `cargo test --workspace`.** The root filesystem is
> 99 % full and the debug-profile build hits `No space left on device`. Everything
> here was built and tested with `--release`. On the new box, run the full suite
> before trusting anything below.

---

## 4. Honest performance numbers

Cooled idle die, Vulkan, all arms **interleaved in one process**, harness warmed
before timing (§9). Every row asserted its tokens byte-identical to greedy.

| subject | prompt | n | greedy | spec `k=3 n>=4` | spec `k=3 n>=16` |
|---|---|---|---|---|---|
| Bonsai-2B Q2_0 | 5 tok | 64 | 9.142 ms / **109.4 tok/s** | 6.325 ms / 158.1 / **1.445×** / fire 71.4 % | 6.972 ms / 143.4 / 1.311× / fire 40.0 % |
| Ornith-9B Q4_K_M | 5 tok | 64 | 46.983 ms / **21.3 tok/s** | 29.496 ms / 33.9 / **1.593×** / fire 71.4 % | 33.242 ms / 30.1 / 1.413× / fire 40.0 % |
| Ornith-9B Q4_K_M | `def fibonacci(n):` | 64 | 48.838 ms / **20.5 tok/s** | 50.074 ms / 20.0 / **0.975×** / fire 1.6 % | 49.378 ms / 20.3 / 0.989× / fire 0.0 % |

**Read these four ways:**

- **Spec decode pays ~1.4×** on text with exploitable repetition, and its **no-fire
  floor is 0.975–0.989×** — i.e. it costs 1–2.5 % when there is nothing to draft.
  That floor is the number a payoff has to beat, and it is what P8/I was missing.
- **The dense 9B decodes at ~21 tok/s greedy, ~34 tok/s with spec** on an 8 GB
  2070 Max-Q. An earlier 351 ms/token (2.8 tok/s) reading for the same model and
  binary is **retracted** — it was measured on a busy box, which the protocol
  forbids. If you see ~93 GB/s effective instead of ~4 % of peak, that is the
  correct regime.
- **`3:16` is not a no-fire arm.** On echo-rich prompts it fires 40 %. `min_match`
  is clamped to `max_match = 16`, so no policy never fires; measure the floor on a
  prompt with nothing to match.
- **Both fast prompts are degenerate.** A representative prose continuation
  **cannot be measured on this box at all** — it OOMs (§6). That is the clearest
  single statement of what 8 GB was holding back, and it is the first thing to
  re-measure on 16 GB.

Earlier baseline for context: Bonsai-2 (Qwen35, Q2_0) wide-GEMV decode at ctx 13
went 68.4 → 112.3 tok/s, and the ctx 13→8200 decode curve flattened from −72 % to
−9 % (split-KV sharding). Both are in `docs/kore-improvement-plan.md`.

MoE figures, from §14.28/§14.31/§14.32 — **treat as contended**, they were measured
on the busy box and should be re-taken:

- forced 1 expert/layer (all cached): 26.4 ms/forward, 16.26 tok/s;
- routed warm: 89.9–100.9 ms, 7.03/5.72 tok/s, 170 MB/token;
- routed cold: 288.2–335.0 ms, 1.90/1.77 tok/s, 4550 MB disk reads.

---

## 5. The machine changed: 8 GB NVIDIA → 16 GB AMD

**The good news: nothing in the engine is NVIDIA-specific.** It is wgpu + WGSL;
NAGA compiles it for whatever Vulkan driver is present. Expect it to *run* first try.

Expect to **re-tune**, though, and for these reasons specifically:

- **Wavefront size.** AMD gfx10+ runs wave64; the NVIDIA box ran wave32. Kernels
  written `@workgroup_size(128)` still execute, but occupancy and the ideal workgroup
  size differ. The GEMV row-blocking (`KORE_GEMV_BLOCKS`) and the split-KV sharding
  factors were tuned on wave32 — re-derive them, don't inherit them.
- **Bandwidth ceilings.** All the GB/s figures in the plan are Turing numbers. The
  *shape* of the conclusions (attention is parallelism-bound, not byte-bound;
  expert bytes are not the binding term) should survive; the constants will not.
- **Subgroups are off the table.** NAGA rejects `subgroupAdd`; the attempt is
  documented and closed. Do not reintroduce them.
- **FP16 throughput** differs enough that the Q8_0/Q6_K lane balance may shift.

**AMD-specific operational notes:**

- `kore-tools/cool.sh` (newly committed) already handles `rocm-smi` *and* the
  `amdgpu` hwmon node, so it works without ROCm tooling installed. Use it before
  every timed run.
- Use `rocm-smi` (or the hwmon sysfs nodes) where the plan says `nvidia-smi` — the
  VRAM samplers in §6 and the route dumps are patterns, not scripts.
- The plan's `KORE_COOL_C` / `KORE_COOL_MAX_S` env knobs are consumed by
  `kore-cli/examples/bench_*_cool.rs`.

**What 16 GB actually unblocks** (ranked):

1. **The VRAM cliff disappears** (§6) — representative prompts become measurable.
2. **The MoE slot pool can go to 64.** §14.32 measured slots 32→64 as a 1.46× cut in
   uploaded expert bytes for **+8 % tok/s**; the default stayed 32 only because
   VRAM. `KORE_MOE_SLOTS` overrides.
3. **The whole 13–14 GB MoE class runs.** Note the old constraint was system **RAM**
   (16 GB) and page cache, not just VRAM — check the new box's RAM before assuming
   the 35B-class checkpoints are usable.

---

## 6. The inherited bug: ~2.4 GiB of VRAM is unaccounted for

This is the top open item, and it is the thing that made 8 GB unusable.

**Measured accounting** (`Ornith-1.0-9B-Q4_K_M.gguf`, 5,690 MB of tensors):

- **4,997 MiB of weights are uploaded.** `token_embd` (572 MB, Q4_K) deliberately
  stays on the host — embedding is a host-side row dequant. `output.weight`
  (834 MB, Q6_K) is the head and *is* uploaded.
- The `nvidia-smi` upload ramp tops out near **4,691 MiB**.
- **After** the upload finishes, the footprint rises a further **~2.4 GiB** during
  generation setup, in ~260 MiB and ~772 MiB steps, to a **7,231 MiB plateau /
  7,366 MiB peak**.

That leaves <1 GiB of headroom on an 8 GB card, so the resulting OOM is prompt-shaped:

| prompt | n_new | max_ctx | result |
|---|---|---|---|
| 5 tok | 256 | 325 | OK |
| 21 tok | 1 | 86 | OK |
| 21 tok | 32 | 117 | **OOM** |
| 42 tok | 1 | 107 | **OOM** |
| 42 tok | 64 | 170 | **OOM** |

Note the shape: a 5-token prompt with **256 generated tokens** (max_ctx 325) runs,
while a 42-token prompt with **1 generated token** (max_ctx 107) does not. So it is
not context length and not generation length.

**Two hypotheses already ruled out — do not redo them:**

- **Not allocation granularity.** `alloc_storage_buffer`
  (`crates/kore-vulkan/src/buffer.rs:46`) allocates exact sizes, 4-byte aligned, from
  a size-keyed pool, and creates buffers `STORAGE|COPY_SRC|COPY_DST` with no mapped
  staging mirror.
- **Not a per-generation leak.** On Bonsai the peak is bit-identical (1039 MiB) at
  4 and at 18 generations of the same workload.

**Leading hypothesis** (unproven): pooled buffers (`PooledBuffer`) retained across
the prefill chunks and/or wgpu staging that is never returned — i.e. the P8/H work
item. Reproduce with a 60 ms sampler and bisect with `KORE_OP_COUNT` /
`KORE_SKIP_OPS` (`kore-cli/examples/op_counts.rs`):

```bash
# on the NVIDIA box this was:
( nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 60 ) > /tmp/vram.csv &
./target/release/examples/bench_draft models/ornith-9b/Ornith-1.0-9B-Q4_K_M.gguf \
    models/bonsai2-2b/tokenizer.json "The capital of France is" 64 1 "3:4"
kill %1
```

On 16 GB this bug is a curiosity rather than a wall — which means it is now
*debuggable*. Fixing it is worth ~2.4 GiB of the most valuable resource you have.

---

## 7. Immediate backlog, ordered

Nothing here is speculative; each item is either measured-open or explicitly
deferred in the plan. Order is by (value ÷ risk).

1. **Re-baseline the numbers in §4 on the AMD box.** Before touching code. Same
   harness, same prompts, cooled, interleaved. Everything downstream is a ratio
   against this.
2. **Resolve the ~2.4 GiB VRAM accounting** (§6). It is the difference between
   running the model and not.
3. **Re-tune `k` and `min_match` on the warmed harness.** The plan's optimum
   (`k=3, min_match=4`) was chosen against a ratio the warm-up fix later showed was
   inflated by 1.41×. `kore-cli/examples/draft_headroom.rs` is the sweep tool.
4. **Raise the MoE slot pool to 64** and re-measure. `KORE_MOE_SLOTS=64`; +8 %
   measured, VRAM was the only blocker.
5. **Device-side top-k with one batched readback.** The routed step spends 39–48 %
   of its time in `download` (§14.32), because the host drives routing. Measured,
   and it does not move with the pool — it is the streaming dependency.
6. **Attention inner-loop efficiency at long context.** The split-KV sharding fixed
   the *parallelism*, not the inner loop; the 5.3×–23.5× figures at 2048–8192 show
   headroom remains.
7. **P7 remaining:** E (batched `m>1` attention), F (parallel GDN scan), G (prefix
   caching).
8. **P8:** H (`alloc_storage_buffer` → `PooledBuffer` at ~30 call sites — kills the
   forgotten-recycle bug class that caused the `m>1` decay, and is the same class as
   §6), I (**done** — §4 measured the no-fire floor).
9. **P9 LAST, against a frozen baseline.** L (`dot_wide` FMA reassociation) first,
   then K (`gdn_step` threads). Both need a written tolerance argument, not a token
   diff (§1 rule 2).

---

## 8. Closed — do not rebuild these

Each of these cost real time and was closed by measurement. Rebuilding one is a
regression in effort, not progress. The plan section is the evidence.

| Lever | Verdict | Where |
|---|---|---|
| **Prefill batching across the MoE boundary** | CLOSED. Worse than nothing below m=32 at cap 32. | §14.31, §14.32 |
| **Expert requantization / fewer expert bits (P12)** | DEMOTED. A 2.1× cut in expert bytes buys ~6 % tok/s; bytes are not the binding term. | §14.32 |
| **Clifford / HQMQ weights (Arm C)** | FALSIFIED. 128-wide Hadamard wins; HQMQ falsified. | `410d5da`, `7b7ed95` |
| **P6.6 KV rotation as a speed lever** | REFUTED. It is a *capacity/context* item only; shipped opt-in behind Q8 default. | §14.30 |
| **Dispatch fusion** | CLOSED. | plan |
| **Subgroups** (`subgroupAdd`) | Off the table — NAGA rejects it. | plan |
| **Paged KV as a performance mechanism** | DEMOTED to memory management only. | plan |
| **Qwen2 port** | DROPPED. P6.5 infra kept. | plan |
| **Tree/beam drafting** | Not built, deliberately: chain-only n-gram lookup. | §"Decisions" |
| **CUDA backend** | Deferred by decision; wgpu is the agnostic base. | plan |

---

## 9. The measurement protocol (non-negotiable)

Every retracted number in this effort came from breaking one of these.

1. **Cool the die before every timed run.** `kore-tools/cool.sh` (53 °C default).
   A saturated die read 155.7 ms where the truth was ~64.
2. **Interleave the arms, in one process.** A ratio across two processes on this
   class of box is not a measurement. `bench_draft` rotates arm order per round.
3. **Warm every arm before timing any of them.** wgpu compiles pipelines lazily, so
   whichever arm runs first in round 0 pays for the whole run — and with
   `rounds < n_arms` the rotation never gives it a turn behind a warm GPU.
   `bench_draft` now does a full-`n_new` warm-up pass; without it a 0 %-fire arm
   measured **1.20× faster than greedy**, which is arithmetically impossible.
4. **Same page-cache state for MoE.** Expert bytes are streamed from disk; a warm
   cache changes the answer by an order of magnitude (§14.28).
5. **Never use a token diff as the gate for a reassociated sum.** Tolerance argument
   + `bench_chain`, per §1 rule 2.
6. **State the shape.** "tok/s" without `n`, prompt length, and `max_ctx` is not a
   number; the cliff in §6 is exactly why.
7. **Do not add churn to files with pre-existing `cargo fmt` drift.** Use
   `rustfmt --edition 2021 --check <file>` — `cargo fmt -p <crate> -- --check <files>`
   silently ignores the file arguments.

---

## 10. Repository state and commands

**Remote:** `https://github.com/KidIkaros/KORE`, branch `main`. This handoff lands
with the previously-local work (26 commits + the docs/tooling commits) pushed for
the first time — so `origin/main` is now the truth, and there is no long-lived local
divergence to reconcile.

```bash
# Build the lanes you need (everything is --release; debug does not fit on the old box)
cargo build --release -p kore-cli --example bonsai
cargo build --release -p kore-cli --example bench_draft
cargo build --release -p kore-vulkan --example bench_gemv
cargo build --release -p kore-vulkan --example bench_attn
cargo build --release -p kore-nn --example kv_dump

# Lint the shipped targets (lib/bin are clean under -D warnings; test code is not)
cargo clippy --release -p kore-nn --lib -- -D warnings

# Full test suite (needs disk the old box no longer had)
cargo test --workspace --exclude kore-serve   # kore-serve's layered_gguf_test hangs, pre-existing

# Reference oracle
LLAMA_SRC=/path/to/llama.cpp kore-tools/llama-oracle/build_ref.sh
```

**Harnesses** (all `crates/kore-cli/examples/`, the argument shapes are the point):

| example | what it answers |
|---|---|
| `bonsai.rs <gguf> <tok> <prompt> <n> [cpu\|gpu]` | decode/prefill tok/s + the token stream |
| `bench_draft.rs <gguf> <tok> <prompt> <n> <rounds> [policies]` | spec-decode A/B, exits 1 on divergence |
| `bench_context.rs`, `bench_context_cool.rs` | the ctx→tok/s curve |
| `bench_prefill.rs`, `bench_multi.rs`, `bench_moe.rs` | prefill / multi-column / MoE |
| `draft_headroom.rs` | sweep `k` × `min_match` |
| `verify_m.rs` | engine against itself across widths |
| `op_counts.rs` | per-op dispatch counts |
| `kore-vulkan/examples/bench_gemv.rs` | dense GB/s at a shape (`n,k,m`) |
| `kore-vulkan/examples/bench_attn.rs` | attention bandwidth |
| `kore-nn/examples/kv_dump.rs` | dump KV (`KVDU` v1) for the quant probe |

**Env knobs:** `KORE_MOE_SLOTS`, `KORE_MOE_PROFILE` (an *empty* value still enables
profiling — unset it to disable), `KORE_MOE_ROUTE_DUMP`, `KORE_MOE_BATCH`,
`KORE_MOE_FORCE_EXPERT`, `KORE_GEMV_BLOCKS`, `KORE_ATTN_SPLIT_MIN`, `KORE_HAD_OFF`,
`KORE_NO_FUSE`, `KORE_OP_COUNT`, `KORE_SKIP_OPS`, `KORE_SPEC_DEBUG`, `KORE_MTP`,
`KORE_COOL_C`, `KORE_COOL_MAX_S`, `KORE_KV_*`, `KORE_QWEN36_GGUF`.

**Instruments:** `kore-tools/moe-route-locality.py`,
`kore-tools/clifford-quant-probe.py` (both committed), `kore-tools/cool.sh`,
`kore-tools/llama-oracle/` (new here).

---

## 11. Inherited open questions

Not blockers, but unresolved and worth an hour each before they surprise someone:

- **The ~1.41× spec-decode ratio inflation is fixed in the harness but the plan's
  tuning was done against the inflated number** — hence backlog item 3.
- **`gdn_step` is 10–17 % off** at large widths; it is P9/K and needs a tolerance
  argument. The shader also hard-codes 128-wide staging, so the tiny fixture
  (`d_k = d_v = 32`) cannot exercise it.
- **`m > 4` needs the `gemv_q2_0` weight-axis redesign**, not a config change.
  Measured model: `GEMV(m) ≈ ceil(m/4)·A + c·m` with `A ≈ 3.04 ms`, `c ≈ 2.2 ms` —
  so `m ≤ 4` is an honest cap.
- **`generate_spec` only works on a dense arch** (`moe_ffn_gpu` errors when
  `rows != 1`). If spec decode is wanted on MoE, that boundary is the work.
- **Qwen2.5-0.5B first-token mismatch** — parked by decision, cause unknown.
- **`kore-serve --test layered_gguf_test` hangs** — pre-existing, excluded from test
  runs. Worth fixing since it hides suite failures.
- **`--all-targets` clippy** reports pre-existing lints in test code and in
  `kore-btes`, `kore-data`, `kore-clifford`, `kore-kernels`. Lib/bin targets are
  clean. Do not let these hide a new lint in your own change.
- **`ref_ids` exits 134 (SIGABRT)** after printing `prompt_ids:` when the prompt
  exceeds `n_batch = 2048` — harmless, but it looks like a crash.
- **The `KORE_MOE_PROFILE` `download` term (39–48 %)** is host-driven routing
  serialization, not bandwidth. It is the single best-understood remaining MoE
  lever and is backlog item 5.

---

## 12. Why this is worth continuing

The engine runs a 100B-class MoE byte-identically to llama.cpp on an 8 GB laptop
GPU, with no CUDA and no vendor kernels. The bottleneck was never the design — it
was 8 GB of VRAM and a box that could not hold its own page cache. A 16 GB AMD card
removes both, and the plan has a measured, ordered backlog waiting for it.

The one thing to carry forward from this box is the discipline: cool the die,
interleave the arms, warm the harness, state the shape, and gate byte-identity per
batch shape. Every number in §4 survived that; every number that did not is marked
retracted in §14.33.

---

## 13. MoE on the 16 GB AMD box: device routing, fused down, and the wall

Measured 2026-08-13 on the RX 9060 XT, `Qwen_Qwen3.6-35B-A3B-Q2_K_L.gguf`,
Vulkan, `-ngl` equivalent (every layer resident), 5-token prompt, `n=64`,
`max_ctx=128`, arms interleaved 3 rounds with a 6 s gap. Correctness gate:
`max|d|` against the host path and token ids against the oracle.

### What moved

| arm | tok/s (3 rounds) | ratio |
|---|---|---|
| host-routed expert stream (`default`) | 11.77 / 11.82 / 11.81 | 1.00× |
| device-routed (`KORE_MOE_DEVICE_ROUTE=1`) | 17.11 / 17.22 / 16.99 | **1.44×** |

The llama.cpp reference on the same file and card is **80.25 tok/s**, so the
remaining gap is 4.7×. A warmed run after the dense benchmark reached 20.80
tok/s: the die was at sclk 1184 MHz of 2620 during the interleaved rounds, so
absolute numbers move with the power state. The 1.44× is interleaved and is the
number to quote.

Correctness: routed `down` output `max|d| = 2.2351742e-8` vs host, token ids
exact (`11751 13 198 760 6511 314 9338 369`), repeated 8 times for `n=64`.

### What was built

- `moe_topk.wgsl` / `moe_gather.wgsl`: routing, softmax, top-k, and expert
  gather on device. The host never sees the token's expert list.
- One gather per projection instead of one per expert; the selected expert
  matrices land in `gate_stage` / `up_stage` / `down_stage` in selection order.
- `gemv_q2_k_at` / `gemv_q3_k_at` and their `_into` forms: a matvec against a
  matrix at a word offset inside a stacked bank.
- `gemv_q{2,3}_k_id`: one expert-strided dispatch for all eight routed `down`
  projections, then one `sum_rows`. This is the `MUL_MAT_ID` shape, minus the
  in-place bank read.
- `flush_uploads`: bulk expert uploads now submit the staging belt. GTT peak
  fell from ~16 GB to **1.5 GB**; without it the "resident" tier spilled.
- Q2_0 gained a weight offset and the dense FFN now stores gate and up as one
  packed matrix (rows `0..gate_n` gate, the rest up), read by one GEMV.

### Census (ctx 6, 4 steps, no sync)

| arm | dispatches/step | wall | tok/s |
|---|---|---|---|
| host-routed | 1383 | 75.5 ms | 13.2 |
| device-routed | **1143** | 54.5 ms | 18.4 |

The 280-dispatch cut (fused `down`) bought ~2 ms, i.e. **~6.5 µs per removed
dispatch**. The step is latency-bound on small dependent ops, not launch-count
bound: the remaining 54 ms is 40 layers of ~28 dependent dispatches, each doing a
small matvec. That is why the gap to llama.cpp is a kernel-fusion gap, not a
plumbing gap.

### Closed this session (do not rebuild)

| Lever | Verdict |
|---|---|
| **Full residency vs streaming** | Worthless on 16 GB: warm 16.56 (resident 128) vs 16.51 (streaming) tok/s. Keep `KORE_MOE_RESIDENT=all`; the win was the GTT fix, not the tier. |
| **Accumulate `down` in place** | Slower: 15.75–17.33 vs 20.89 tok/s. The read-after-write on the 8 KB output serializes the GEMVs; `sum_rows` after independent writes is better. |
| **Dense gate+up fusion** | Neutral: 321 vs 345 dispatches, 118 vs 118 tok/s in a 4-round A/B. Keep it (fewer launches, one upload) but it is not a speed lever. `KORE_DENSE_SPLIT_GATE_UP=1` restores the split path. |

Dense at the current clock state: 118 tok/s greedy (`n=200`), 189 tok/s with
spec `k=3, min_match=4` (100 % acceptance, 1.60×). The earlier 142/232 readings
were taken with the die at a higher sclk; the ratio is the stable number.

### Next, in order

1. **One MoE layer, one dispatch (or two).** Route → gate/up → SwiGLU → down →
   weighted sum, with the expert matvecs reading the **resident bank in place**
   by `idx` (no gather, no stage buffers). The `gemv_q{2,3}_k_id` stride logic is
   the prototype; it needs an `idx` binding and an `x_stride` of zero for
   gate/up. This is the only path from 17 to the llama.cpp 80.
2. **Per-expert routing counters** for hot-set pinning on 8 GB cards
   (`KORE_MOE_RESIDENT=N` currently pins the first N, not the hottest N). See
   `docs/deepinfer-scaled-down.md` §3.
3. **Dense:** stop at the fused gate/up. The next dense lever is not another
   dispatch merge; it is occupancy in the Q2_0 GEMV at `m=1`.

DeepInfer mapping and verdicts: `docs/deepinfer-scaled-down.md`.
