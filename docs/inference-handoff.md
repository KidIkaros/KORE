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
| Batched decode, up to 8 columns | `forward_batch`, `MAX_BATCH_COLS = 8` (`qwen35.rs:132`) | done, two row groups; `verify_m` bit-identical |
| Chunked prefill | `prefill` (`qwen35.rs:3006`) | done, `PREFILL_CHUNK = 8` including the routed FFN |
| Split-KV long-context decode attention | `kore-vulkan/src/backend.rs` | done |
| Q2_0 / Q4_K / Q5_K / Q6_K / Q8_0 GPU GEMV | `backend.rs` | done |
| MoE engine (host routing, promote-on-miss expert cache, LRU) | `qwen35.rs` `moe_ffn_gpu` (`:2577`), `DeviceBank` (`:1954`) | done, v1 |
| 100B-class MoE coverage gate vs llama.cpp | `kore-nn/tests/qwen36_real.rs` | done |
| Speculative decode: n-gram/suffix draft, chain-only | `spec.rs` | done, byte-identical |
| Rewind engine + per-row Delta-state snapshots | `new_rewind`/`rewind` (`:3040`,`:3097`) | done |
| P6.6a: rotated KV quant | KV dump + probe | done, **opt-in** (capacity-only) |

**Not built, on purpose:** tree/beam drafting, CUDA backend, expert requant (P12).
See §8.

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

The table above is the cooled-die campaign and is kept for comparability. The
current dense operating point is `k=6, min_match=6` at 1.58× (repetitive) /
1.38× (prose); see §13 for the interleaved numbers and the wider verify pass.

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
3. ~~**Re-tune `k` and `min_match` on the warmed harness.**~~ Done: `k=6,
   min_match=6` at `MAX_BATCH_COLS=8`; `draft_headroom` now sweeps the full grid.
   See §13.
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
`KORE_NO_FUSE`, `KORE_OP_COUNT`, `KORE_SKIP_OPS`, `KORE_STEP_PROFILE` (split one
step into embed / host encode / queue drain), `KORE_SPEC_DEBUG`, `KORE_MTP`,
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
- ~~**`m > 4` needs the `gemv_q2_0` weight-axis redesign.**~~ Resolved: the
  kernel already launched `grid.y = ceil(m/4)`; it simply read column 0..3 again
  in the second group. Indexing `x` by `m_base` and raising the check to 8 gives
  a bit-identical pass (measured 28.3 ms at `m=8` vs 15.5 ms at `m=4`, so the
  per-token cost still falls).
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

## 13. MoE on the 16 GB AMD box: device routing, in-place experts, and the two kernels that dominated

Measured 2026-08-13 → 2026-08-20 on the RX 9060 XT,
`Qwen_Qwen3.6-35B-A3B-Q2_K_L.gguf`, Vulkan, every layer resident, 5-token
prompt (`KORE_IDS="11751,13,198,760,6511"`), `max_ctx` 96, arms interleaved in
one time window. Correctness gate: token ids against the host lane and `max|d|`
on the routed FFN.

### Final numbers (2026-08-20, `n=64`, 3 rounds)

| arm | tok/s | ratio |
|---|---|---|
| host-routed expert stream (`default`) | 13.77 / 13.59 / 13.49 | 1.00× |
| device-routed (`KORE_MOE_DEVICE_ROUTE=1`) | 52.95 / 51.36 / 52.30 | **3.8×** |

The die sat at **sclk 2012 MHz of 2620**, edge 51 °C, during these rounds. The
earlier 1.44× was measured at 1184 MHz with the old top-k and router kernels, so
the ratio moved for two reasons: the code got faster *and* the device lane is
GPU-bound while the host lane is not, so a higher clock lifts the device arm
more. The ablation below is the clock-independent evidence for the code cut.

Current state, same protocol, `n=128`, `max_ctx` 160, three interleaved rounds
at **sclk 1.64–1.83 GHz**, edge 52–57 °C:

| arm | tok/s | ratio |
|---|---|---|
| host-routed | 14.02 / 14.19 / 14.00 | 1.00× |
| device-routed | 55.90 / 55.77 / 56.82 | **3.99×** |

Sustained (`n=400`, `max_ctx=512`): **58.19 / 58.61 tok/s** over 6.9 s at
1.34 / 2.21 GHz. Peak VRAM is unchanged at ~15.6 GiB, because none of these
changes touch residency.

llama.cpp on the same file and card measured **80.25 tok/s**. The gap is now
~1.5× in the current state; it was 4.7× when this section was first written.

A sustained run (`n=400`, `max_ctx=512`, one arm) held **59.8 tok/s** over
6.69 s; peak VRAM was **15.63 GiB** of the ~16.7 GiB usable, so the margin for
the next allocation is ~1 GiB. That is the number to quote for long generations:
attention grows with context but the clock also ramps, and the two roughly
cancel over the first 400 tokens.

### What moved, in order

| change | effect |
|---|---|
| **Parallel router top-k** (`moe_topk.wgsl`) | 49.4 → 29.8 ms/step; real loop 17.6 → 28.2 tok/s (`n=48`) |
| **Row-major router matvec** (`router_matvec.wgsl`) | 29.8 → 17.0 ms/step; real loop 28.2 → 47.5–50.2 tok/s (`n=48`) |
| **In-place expert read** (`idx` + `x_stride` in `gemv_q{2,3}_k`) | 1143 → 1023 dispatches, no gathers, no `down_stage`; ~+3 % |
| **SwiGLU + routing scale, `sum_rows` + shared add** (`swiglu_rows`, `sum_rows_add`) | 1023 → 903 dispatches; 53.7 → 54.6 tok/s (+1.7 %, `n=64`, 3 interleaved rounds) |
| **One router matvec for top-k + shared gate** (`moe_topk_gate`) | 903 → 823 dispatches; no-sync 15.53 → 14.93 ms; real loop 57.6 → 58.0 tok/s |
| **`gdn_step` state i-major** | ~3.8 → ~2.9 ms of the step (ablation delta); 53.9 → 57.1 tok/s (+5.8 %) |
| device routing, fused `down`, `flush_uploads` (previous session) | 11.8 → 17.1 tok/s (1.44×) |

The two kernels were not "small dependent ops". They were the two halves of the
routing decision:

- `moe_topk` was a **single workgroup of one lane**: a max-subtracted softmax
  over 256 experts, then an O(k·n) selection that re-ran the exclusion scan
  inside it, with `exp` on the hot path. ~0.55 ms per layer, **21.8 ms per
  step** of a 49.4 ms step.
- The router matvec went through the generic GEMV, which wants
  `[hidden, experts]`; the engine transposed the GGUF router at load to match.
  Every weight read then touched one 4-byte word per cache line. ~0.18 ms per
  layer for the routed router, plus the shared-expert router, **~15 ms/step**.

`moe_topk` now uses 256 lanes (one expert each): the max and each argmax round
are tree reductions, ties go to the lower expert id exactly as the host scan
does, and only the softmax normaliser stays on one lane so its summation order
stays the host's. `router_matvec` reads the GGUF's native `[experts, hidden]`
rows contiguously, one workgroup of 128 lanes per expert.

The in-place read removed the three per-layer gathers (gate, up, down) and the
down staging buffer: `gemv_q{2,3}_k_id` now takes the routing ids and an
activation row stride, reads expert `idx[position]` straight out of the resident
bank, and gate/up share one activation row (`x_stride = 0`). 120 fewer
dispatches and ~165 MB less traffic per token.

### The ablation that found them

`KORE_SKIP_OPS=<label>` drops a dispatch before it is recorded, leaving shapes
and dependencies intact, so the wall-time delta is that label's cost. At the
final 18.6 ms baseline:

| label | ms/step | note |
|---|---|---|
| `gdn_step_pass` | 4.93 | largest single kernel then; ~2.9 after the i-major state layout |
| `gemv_q4_k_compute_pass` | 3.49 | attention projections, 100 dispatches |
| `gemv_q2_k_compute_pass` | 3.39 | 152 dispatches (MoE gate/up, shared expert) |
| `had_pre_pass` | 2.87 | 171 dispatches (norms, SwiGLU) |
| `gemv_q8_0_compute_pass` | 2.65 | 95 dispatches |
| `gdn_gates_pass` | 1.75 | |
| `sum_rows_pass` | 1.59 | 40 dispatches |
| `gemv_pass` | 1.52 | residual F32 matvecs |
| `elem_compute_pass` | 0.54 | |
| `gdn_conv_pass` / `flash_decode_gqa_q8` / `qknorm_rope` | ~0.2 / ~0.2 / ~0 | |

Second pass on the current tree, `KORE_SKIP_OPS` deltas at ~56 tok/s: `gemv_q8_0`
3.0 ms, `gemv_q2_k` 2.9, `gdn_step` 2.2, `gemv_q4_k` 2.1, `moe_topk` 2.0,
`had_pre` 1.9, `router_matvec` 1.3, `flash_decode_gqa_q8` 0.85, `gemv_q6_k`
(the 248k-wide vocabulary head) 0.35, `quantize_q8_rows` 0.34, `gdn_conv`
0.02. Nothing anonymous is left: the step is those seven kernels plus the
~4.9 ms of dispatch floor that 823 dispatches imply.

Caveat: skipping a **normalisation** (`elem_mul_rows`, `quantize_q8_rows`) lets
garbage and denormals into the next GEMV and made those runs *slower*
(173 ms and 71 ms). Those rows are unusable; the ones above are the ones that
removed work without poisoning the inputs.

### Census (ctx 6, 4 steps, no sync)

| arm | dispatches/step | wall | tok/s |
|---|---|---|---|
| host-routed | 1383 | 75.5 ms | 13.2 |
| device-routed, after fused down | 1143 | 54.5 ms | 18.4 |
| device-routed, after in-place read | 1023 | 17.0 ms | 59.0 |
| device-routed, with the two fusions | 903 | 15.5 ms | 64.4 |
| device-routed, plus router fusion + `gdn_step` layout | **823** | **14.9 ms** | **67.0** |

The real loop is ~20 ms/token (48–53 tok/s): the difference is the per-token
argmax readback and the submission wait, not more dispatch overhead. The step is
no longer launch-bound; it is spread across GDN, attention and the MoE matvecs.

### Correctness

- `swiglu_rows` and `sum_rows_add` are gated bit-for-bit against the two
  dispatches they replace (`test_vulkan_swiglu_rows_matches_two_dispatches`,
  `test_vulkan_sum_rows_add_matches_add`), for both folded and unfolded layouts.
  The fused SwiGLU keeps the scale in the same place `mul_rows` put it:
  `select(h, h*scale, row_scale)`, so the unscaled path is untouched.
- Token ids exact against the host lane, repeated after both kernel rewrites and
  after the fusions:
  `[314, 9338, 11, 11751]` for the gate run, `[11751, 13, 198, 760, 6511, 314,
  9338, 369]` for the oracle prompt.
- Host and device picks are identical (`112 181 238 200 254 208 207 81`).
- Routing weights differ in the last ~1e-7 (WGSL `exp` vs host libm), and the
  routed FFN output `max|d| = 5.59e-8` (it was 2.24e-8 before the router change,
  on a different layer). That is a numerical lane, not a bug: the host lane
  remains the byte-identity oracle for anything that needs it.
- Dense `m=8` verify: `verify_m` reports 37 positions bit-identical to
  sequential decode, 15 rewind keep-points over widths `[8,7]`, and prefill
  agreement.

### Closed this session (do not rebuild)

| Lever | Verdict |
|---|---|
| **Full residency vs streaming** | Worthless on 16 GB: warm 16.56 (resident 128) vs 16.51 (streaming) tok/s. Keep `KORE_MOE_RESIDENT=all`; the win was the GTT fix, not the tier. |
| **Accumulate `down` in place** | Slower: the read-after-write on the 8 KB output serialises the GEMVs. `sum_rows` after independent writes wins. |
| **Dense gate+up fusion** | Neutral in tok/s; kept for the dispatch count. `KORE_DENSE_SPLIT_GATE_UP=1` restores the split. |
| **Gathers → in-place expert read** | Built, matches token ids, worth ~3 %. The gathers were cheap; the ids were not the problem. |
| **`m > 4` "needs a weight-axis redesign"** | Wrong. `gemv_q2_0` already had `grid.y = ceil(m/4)`; it only ignored `m_base` when indexing `x`. Fixing that and lifting the check to 8 gives a bit-identical 8-column pass. |
| **Single-lane k-selection in `moe_topk`** | Much worse: 58.9 → 45.9 tok/s. 2048 dependent shared-memory loads (one lane, 8 rounds × 256) cost more than the 64 workgroup barriers the parallel rounds pay. Reverted. |
| **Hoist the softmax `exp` into a shared `probs` array** | Neutral (53.6 vs 53.4 tok/s): the 8× redundant `exp` per round is not what the kernel spends its time on. Reverted rather than kept unmeasured. |
| **`vec4` router matvec** | Neutral (53.5 vs 53.6 tok/s, ablation delta unchanged): the 2 MB row read is latency-bound, not issue-bound. Reverted so the logits keep their current rounding. |
| **One warp for `moe_topk`** (`workgroup_size(32)`) | Slower (56.3 → 54.8 tok/s): 8 × 5 barrier steps on one warp lose to 1 expert per lane with an 8-step tree. The 256-lane form is at its floor. |
| **Device-side embedding handoff** | Would buy nothing. `KORE_STEP_PROFILE=1` shows the host is idle **11.3 ms of a 16.1 ms step**: the per-token round trip is fully hidden behind the GPU. Not built. |
| **Sparse attention (DSA / CSA / CSA2)** | Not applicable, and not worth it here even if it were free. All three are pretraining changes a frozen GGUF cannot adopt; attention costs at most ~0.9 ms in a ~16 ms step, and the step is only 0.6 ms slower at 8K tokens than at 96, because only 10 of 40 layers attend and they keep 2 KV heads (~11 KB/token). Mechanisms and the spec-decode interaction: `docs/research-sparse-attention.md`. |
| **Pre-batch snapshot + replay instead of per-column snapshots** | Not worth it. `copy_buffer` now records a real buffer copy, so the per-column snapshots cost 1.15 ms per fired step, and the 6 cut steps in 192 tokens rewind in 0.19 ms each. |
| **Two heads per `gdn_step` workgroup** (256 lanes, 8 warps) | Slower: loop greedy 60.8/61.3 → 59.9/60.1, spec 101.7/101.9 → 99.7/99.5 (−2.1 %), prose −0.9 %. Bit-exact (device ids unchanged). This was the wrong shape: 16 workgroups of 256 left half the CUs idle, so it never tested the chain. Splitting each sweep's *rows* across two groups keeps 32 workgroups and halves the chain -- that is the win recorded above (+7.6 % greedy, +12.4 % loop spec). |
| **Routed gate+up fusion** (one dispatch over one `[gate \| up]` bank) | Built, bit-correct, **rejected**: greedy 59.2 → 25.4 tok/s (14.2 → 35.9 ms/step). It did save exactly the 40 dispatches it promised (823 → 783) and 64 device tokens stayed identical to the host lane, so the split uniforms and the `idx` mapping are right. The cause is the **168 MiB contiguous bank**, not the kernel: two 4096-row dispatches over the pair buffer are as slow (36.1 ms) as one 8192-row dispatch (36.3 ms), while the same two over separate 84 MiB buffers are fast (14.6 ms); zeroing `w2_off` and swapping the chunked upload for a single `upload_u8` change nothing. The 21.5 ms delta against 220 MB of weight reads per step is ~10 GB/s — the large allocation is read out of host-visible memory instead of VRAM. The dispatch-shape A/B also shows a merged 8192-row dispatch costs the *same* GPU time as two pipelined 4096-row ones, so the fusion could only ever have saved the launches (~0.24 ms host, ~0.12 ms GPU, ~1.6 %). Patch kept at `/tmp/opencode/gate_up_fusion_rejected.patch`; a two-binding variant (separate `gate`/`up` bindings chosen by `half`, no large allocation) is the only way this comes back. |
| **Payoff-guard re-probe backoff** (16 → 32 → 64 plain steps) | Slower on the arm it was meant to fix: technical prose 66.5 → 64.8 tok/s. The fires it suppressed were winners -- `6:6:0` fires 24 times for 66.62 tok/s, the shipped guard 18 for 66.46, the backoff 12 for 64.77 -- because ~4.3 tokens per 52.5 ms pass beats ~14.7 ms per plain token. The 0.95× reading was the clock, not the guard (see the note under the regime table). Reverted. |

### Dense at the current state

`n=200`, same prompt as the §13 MoE work (repetitive), 2 rounds interleaved:

| arm | tok/s | ratio |
|---|---|---|
| greedy | 109.63 / 109.43 | 1.00× |
| spec default (`k=6, min_match=6`) | 172.98 / 173.51 | **1.58×** |
| spec `k=3, min_match=4` | 166.29 / 167.49 | 1.53× |

On a prose prompt the same two arms read 147.5 vs 145.8 tok/s (greedy 107):
1.38× vs 1.36×. Lowering `min_match` to 1 buys 1.62× on repetitive text but only
1.18× on prose — it is not the default.

`draft_headroom` now sweeps `m=1..8`/`k=1..7` and is the tool to re-pick the
policy: on the repetitive stream the fully-accepted cost per token falls to
3.51 ms at `m=7` (from 3.88 ms at `m=4`), and the projected optimum moved from
`k=3` to `k=6`; `k=7` loses because the eighth draft almost never lands.

### MoE speculative headroom, measured offline

`spec_replay` replays an existing greedy stream through the real `DraftPolicy`
and projects throughput, so the batch path's value was settled before any
kernel work. Four 400-token MoE streams, `k=6 min_match=6`, `c1 = 16.1 ms`,
1.15 ms snapshot per fire, `C(m) = c1 * (1 + alpha*(m-1))`:

| stream | fires | drafts accepted | ratio at alpha 0.2 / 0.3 / 0.4 / 0.5 |
|---|---|---|---|
| repeated sentence (loop) | 100 % | 100 % | 3.09× / 2.44× / 2.02× / 1.72× |
| narrative prose (400-token story) | 9.4 % | 89.3 % | 1.34× / 1.28× / 1.22× / 1.17× |
| technical explanation (mutex vs semaphore) | 2.3 % | 14.8 % | 0.99× / 0.98× / 0.97× / 0.96× → **1.00×** under the guard |
| code from the prompt | 0.8 % | 38.9 % | 1.01× / 1.01× / 1.00× / 1.00× |

The win is **workload-dependent**, and the shape is the same as dense (1.58×
repetitive / 1.38× prose) with a lower prose floor: ~1.2–1.3× on echo-rich
narrative, 2–3× on repetition, nothing on technical exposition — there the few
fires lose, and the payoff guard pauses them, so it is a no-op rather than a
loss. `min_match=8` is within noise of 6 on prose (1.29× vs 1.28×) and removes
the technical fires entirely; the default stays 6 to match dense.

The projection's one unknown is `alpha`, the marginal cost of a verify column.
Dense measures ~0.33 (24.6 ms at `m=7` against a 7.9 ms plain step); if the MoE
pass amortises its expert reads the same way it lands in that band, where
repetitive is ~2.3× and prose ~1.26×. The first batch measurement replaces the
sweep — and it is the only number that can still cancel the milestone.

### MoE speculative decoding: built and measured

The batch path is in: `router_matvec_cols`, `moe_topk_gate_cols`,
`sum_rows_add_groups`, `pos_div` in the expert-strided GEMVs, and
`moe_ffn_gpu_resident` over `cols` rows. The host lane loops the single-column
path per column as the correctness oracle. Gates:

- `verify_m` on the MoE: `m=4` and `m=8` (device lane) and `m=2` (host lane) —
  21 positions bit-identical to sequential decode, state probe agrees, prefill
  agrees, rewind legs bit-identical.
- 64-token greedy ids identical between the device and host lanes.
- GPU suite 32/33 (only the pre-existing `quantize_q8_rows` failure).

Measured hot, `max_ctx=512`, device routing, `n=400` per arm so prefill is
amortised the same way a user's run would be. The prompt is the same in both
arms (each arm pays it once), so these are end-to-end tok/s:

| regime | greedy | spec `6:6` | ratio | `7:6` |
|---|---|---|---|---|
| repeated sentence (loop) | 66.8 | 117.9 | **1.76×** | 122.2 |
| repetitive code | 68.6 | 101.5 | **1.48×** | — |
| narrative prose | 69.2 | 76.9 | **1.11×** | — |
| technical prose | 68.8 | 65.6 | 0.95× | — |

Interleave the arms and reverse the order within one batch. A batch that runs
`gpu` then `spec` for each prompt gives the spec arm the settled (slower) clock
every time and reads 10-30 % low -- the code arm measured 71.4 tok/s that way
and 101.5 in a proper `gpu, spec, spec, gpu` A/B. The table above is the latter.

**A low-acceptance arm's ratio moves with the core clock.** Technical prose
reads 0.95× with the plain step at 68.7 tok/s and 1.00× with it at 66.3 -- same
binary, same prompt, four interleaved runs each way. The plain step is
latency-bound and tracks the core clock; a fired pass has more parallel work and
sits closer to bandwidth-bound, so the two arms do not scale together. Always
report the greedy arm next to a sub-1.05× ratio: one absolute pair is a
measurement of the clock as much as of the policy.

Repetitive code is the second-biggest win, not a wash: the replay's code stream
happened not to repeat, but real code reuses the same call shape and the 6-gram
trigger catches it. `min_match=8` removes the technical loss but costs 22 % on
code (those wins are 6- and 7-gram repeats), so `6:6` stays.

**MoE prefill batches again.** `prefill` used to force width 1 whenever
`n_experts > 0`, which was only true because the routed FFN rejected
`rows != 1`. With `moe_ffn_gpu_resident` taking `cols`, the chunk is back to
`PREFILL_CHUNK = 8`: the dense projections, both Hadamards and the shared expert
stream once per chunk and the dispatch count drops 8x. The routed bank read
does *not* amortise -- each column picks its own eight experts -- so that sets
the floor, and it is why the win is 1.8x and not 5x.

| measurement | before | after |
|---|---|---|
| 501-token prompt, prefill only | 6.29 s (12.6 ms/token) | **3.50 s (7.0 ms/token)** |
| 501-token prompt + 64 greedy | 8.92 tok/s | **14.54 tok/s (1.63x)** |
| 23-token prompt, 128 spec | 79.9 tok/s | **86.8 tok/s** |
| 23-token prompt, 400 spec | 101.8 tok/s | **105.3 tok/s**, ids identical |

Gates: `verify_m` prefill leg at `m=4` and `m=8` (57 positions bit-identical to
sequential decode), and the 64-token device-vs-host id check still exact. The
table above predates this change (it adds ~2-3 % at `n=400` and ~9 % at
`n=128`).

Absolute tok/s tracks the card's clock -- greedy reads 57-64 across batches --
but the loop ratio does not: 1.64 at 57 tok/s greedy and 1.65 at 64. That
stability is a property of the loop arm (100 % acceptance, so both arms are the
same fired pass), not of the ratio in general; see the technical-prose note
above. One
outlier batch read 69.6 tok/s on the loop arm and was chased as a cold-start
payoff-guard effect; it was **cleared**: three runs after 60 s idle read 94.4
(guard on), 94.6 (guard off), 94.1 (guard on), so the guard is not the variable
and no seeding change is warranted.

On a 128-token prose run with no 6-gram match the ratio is 0.96×: the whole
difference is trigger and loop overhead, so drafting is free when it does not
fire, but not literally free.

**Cost of a verify pass**, read from `KORE_SPEC_PROFILE` on the loop prompt
(100 % acceptance, so every run is fired passes):

| m | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| ms/pass | 25.8 | 31.7 | 40.9 | 46.8 | 52.9 | 58.9 |

Linear with no knee, `C(m) ≈ 13.8 + 6.0·(m−1)`. Against a ~15 ms plain step a
marginal column is **~6.0 ms, about 40 % of one token** (`alpha ≈ 0.40`). It is
not expert traffic -- gate/up and down read the bank once -- it is per-column
work that cannot amortise: `gdn_step` (the state sweep runs per column), the
attention loop (10 attended layers, one dispatch per column) and GEMV arithmetic
that stops being bandwidth-bound once `m` columns share a weight read.

**What a step is made of.** `KORE_OP_COUNT=1` census, greedy, 8 steps, 823
dispatches per step:

| label | /step | | label | /step |
|---|---|---|---|---|
| `had_pre_pass` | 171 | | `gdn_gates/conv/step_pass` | 30 each |
| `gemv_q2_k` | 152 | | `gemv_q3_k` | 20 |
| `gemv_q4_k` | 100 | | `quantize_q8_rows` | 20 |
| `gemv_q8_0` | 95 | | `qknorm_rope` | 10 |
| `router_matvec` / `moe_topk` / `sum_rows` | 40 each | | `kv_append` x3 / `flash_decode_gqa` | 10 each |

`KORE_STEP_PROFILE` splits the 14.24 ms step into **5.07 ms host encode + 9.05 ms
queue drain** (1.6 ms of the encode hides under the GPU because a flush happens
every 48 ops). That is **6.2 µs of host time per dispatch** -- the dispatch floor
-- and ~13 µs of GPU time per dispatch, of which ~6 µs is launch. So the step is
limited by *dispatch count on both sides*, not by arithmetic. The sweep in the
`KORE_BATCH_OPS` comment shows the host/GPU overlap is already at its optimum:
finer flushes move time from the drain to the encode and the sum rises.

`KORE_SKIP_OPS=had_pre_pass` is the one ablation that is safe to read (a rotation
has no denormals, so the timing stays representative): it takes the step
**14.27 → 12.45 ms (58.7 → 66.4 tok/s)**, i.e. 171 dispatches of 2048-element
rotation cost **1.82 ms, 12.7 %**, split 1.23 host / 0.55 GPU. They cannot be
batched across layers -- each one feeds the next GEMV in a strict sequence -- so
the only lever there is folding the norm+rotate into a neighbour. The other skip
rows (router, topk, sum_rows, gdn_gates) produce garbage that changes downstream
timing, `gdn_gates` even makes the step slower; they are not ablations.

**`gdn_step` fused update + readout.** The third state sweep is gone: the lane
computes `S = decay*S + k*delta`, stores it, and accumulates `S·q` on the way.
The values and the summation order are unchanged, so the kernel stayed
bit-exact -- the device lane still emits the same 64-token ids as the host lane
and `verify_m` is still green -- and one 64 KB re-read per head disappeared.
Same-clock old/new pairs: loop arm 77.6/78.0 → 80.2/80.2 tok/s (+3.0 %), prose
67.7 → 69.4 (+2.4 %), `m=7` pass 63.5 → 59.6 ms.

**`gdn_step` splits each state sweep across two thread groups.** Group `g` owns
rows `g*d_v/2 ..` of every head, so each lane's dependent `i` chain is 64 steps
instead of 128 and a workgroup carries 8 warps instead of 4. The two partial
sums are added at the end of the dot and of the readout, which reassociates two
128-term folds into 64-term pairs. Everything else keeps its order: the state
update is still per element, and both L2 norms plus the RMS reduction are run
redundantly by each group on identical values, so they are unchanged.

The tolerance evidence, since this is a reassociation and not a bit-exact
change:

- `test_vulkan_gdn_step_split_matches_reference` against an f64 reference:
  state `max|d| = 2.07e-8` on a scale of `0.27`, output `max|d| = 4.75e-7` on a
  scale of `1.44` -- a few ulp.
- 400-token greedy stream: identical to both the pre-split device lane *and*
  the f32 CPU lane, i.e. the ulp change did not flip a single argmax.
- `verify_m m=8` still bit-identical to sequential decode (every column runs the
  same kernel, so this is unaffected by the tolerance).

Same-clock old/new: greedy 62.1/62.0 → 66.8/66.8 (+7.6 %), loop spec
104.4/105.0 → 117.5/118.2 (+12.4 %), prose spec 70.8 → 76.3 (+7.8 %), `m=7`
pass 59.6 → 52.9 ms. Note the side effect: the plain step gains more than the
fired pass, so the *ratio* narrows on low-acceptance prose (1.16 → 1.11) and
technical prose (0.98 → 0.95) even though every arm is faster in tok/s.

Because `C(m)` is linear and acceptance is 100 %, `k=7` now beats `k=6` on a
loop (122.2 vs 117.9 tok/s); it has not been re-measured on prose, so `6:6`
stays the default and `7:6` is a loop-only choice.

### Next, in order


1. ~~**Speculative decoding for the MoE.**~~ Done: 1.76× on a loop, 1.48× on
   repetitive code, 1.11× on narrative prose, 0.95-1.00× on technical prose
   (clock-dependent, see the note above), and 0.96× when nothing fires at all.
   The remaining leverage is the **6.0 ms marginal column**, and it is
   structural. The delta-rule state
   cannot batch across columns (column `r+1`'s state depends on `r`), and the
   per-column attention call is deliberate, not an oversight: the split-KV
   kernel is `seq_q == 1` only, and the comment at the call site records that one
   batched call would force the latency-bound single-group kernel and read the
   full `seq_k` for every row. So the lever is the *per-step* cost of
   `gdn_step`, which helps greedy as much as spec.

   The per-step cost of `gdn_step` is now split across two thread groups (see
   above), which took the `m=7` pass from 59.6 to 52.9 ms.

   The payoff guard's re-probe was the obvious suspect for the technical-prose
   loss, and it was **measured and rejected**. Backing the re-probe off
   (16 -> 32 -> 64 plain steps after each losing probe, with the wait reset by a
   winning fire) made the arm *worse*: 64.8 vs 66.5 tok/s on the same profile.
   The profile says why. `6:6:0` (guard off) fires 24 times for 66.62 tok/s; the
   shipped guard fires 18 for 66.46; the backoff fires 12 for 64.77. The fires it
   suppressed were winners -- 79 drafts accepted over 24 fires is ~4.3 tokens per
   52.5 ms pass against ~14.7 ms per plain token -- and the guard pauses on a
   6-fire window that two low-acceptance draws can dominate. Suppressing fires
   is not free on this arm, so there is no backoff to win.
2. ~~**`gdn_step` chain split**~~ Done: two thread groups, tolerance pinned by
   `test_vulkan_gdn_step_split_matches_reference` (`max|d|` ~2e-8 state, ~5e-7
   output) and a 400-token stream identical to both the pre-split device lane and
   the CPU lane.
3. ~~Fuse SwiGLU with the routing scale, and `sum_rows` with the shared-expert
   add~~ Done: `swiglu_rows` (mode-2 `row_scale`) and `sum_rows_add` (bias
   binding), 120 dispatches and +1.7 %. The host lane reads the same kernels but
   is bandwidth-bound, so its tok/s did not move (15.7 vs 15.7, `n=64`).
4. ~~**Fuse routed gate+up into one dispatch.**~~ **Measured and rejected.** The
   census argument was right -- 40 dispatches, ~0.25 ms of host encode plus
   ~0.12 ms of GPU launch -- but the layout it needs (one 168 MiB `[gate | up]`
   bank, so the upper half can read the `up` slab with the same `idx`) is read
   out of host-visible memory on this card and costs 21.5 ms/step, three times
   the whole step. Details and the isolating A/Bs are in the rejected table
   above. Do not rebuild the single-bank form; only a two-binding kernel (no
   large allocation) could pay, and the shape test caps its upside at ~1.6 %.
5. **Dispatch count is the binding cost on both sides** (823/step, 6.2 µs host
   + ~13 µs GPU each). The remaining candidates, in decreasing size, are
   `had_pre` (171, but sequential -- fold it into the neighbouring
   normalize/GEMV rather than trying to batch it) and the three `kv_append`
   dispatches per attended layer (30). **Do not** fuse `router_matvec` into
   `moe_topk` by moving the router into one workgroup: the router reads 256 x
   2048 f16 = 1 MB per layer, which the current 256-workgroup matvec spreads
   across every CU in ~5 us, while a single workgroup is stuck on one CU at
   memory latency -- ~100 us a layer, 4 ms a step. The pair only looks cheap
   because both labels cost the same 40 dispatches.
6. **Per-expert routing counters** for hot-set pinning on 8 GB cards
   (`KORE_MOE_RESIDENT=N` currently pins the first N, not the hottest N). See
   `docs/deepinfer-scaled-down.md` §3.
7. **Device routing stays opt-in.** Peak VRAM is ~16.1 of 17.1 GB, so enabling
   it without a free-memory check would fail on a busy desktop.

DeepInfer mapping and verdicts: `docs/deepinfer-scaled-down.md`.

### Known failures at this commit

- `kore-vulkan` integration test
  `test_vulkan_quantize_q8_rows_matches_reference` fails on the committed tree
  too (`q[0,13]: got 191, want 192`). It is a pre-existing tie-rounding
  disagreement between the test's expectation and the shader, not a regression
  from this work.
