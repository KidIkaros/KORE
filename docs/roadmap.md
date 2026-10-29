# Roadmap

State as of **0.2.1** (2026-10-29). The campaign log with every measurement and
every rejected experiment is `docs/inference-handoff.md`; this file is only what
is left, in the order it should be done, with a way to tell whether it is done.

## Where this stands

Green and shipped: qwen3.6-35B-A3B and dense 2B on the device lane with
speculative decoding, a host-path port of Flint with route profiling, the sink +
window decode-attention kernel and the flat QK-norm/RoPE kernel (both gated
against CPU oracles), the H2D ceiling measurement, and a fully green workspace:
34 test suites in both the debug and release profiles, and
`cargo clippy --workspace -- -D warnings` clean.

Not shipped: any end-to-end *device* model for Flint, and any external oracle
for Flint. That is the whole gap between "kernels exist" and "a second
architecture runs on this card".

## P0 — Flint on the device

Four slices, in this order. Each has a correctness gate that must pass before
the timing below it means anything.

### 1. One-layer `attn_step` device path

norm -> qkv GEMV -> `flint_qknorm_rope` -> `kv_cache_append_pair` ->
`flash_decode_gqa_sink` -> o GEMV, compared against the host
`flint.rs::attn_step` on the same weights.

**Exit:** per batch shape, `max|Δ|` and the first divergence recorded for the
output; a stream of 48 greedy ids identical to the host lane. Reassociated sums
get the `max|Δ|` treatment, not a token-id diff.

**Known snag:** `kv_cache_append_pair` is a copy-based append over f32
`[rows, cols]`, not a ring. Windowed layers (35 of 49, window 513) need either a
ring write index or a pre-rotated buffer layout before the window path is
meaningful. Decide which one before writing the slice; the choice changes
`flash_decode_gqa_sink`'s `kv_start` handling.

### 2. Routed expert lane

Reuse qwen35's host-routed lane: experts live in `DeviceBank`, routing happens
on the host, one 8 KB readback per layer, plain GEMV per selected expert.

**Exit:** routed layer matches the host lane at `max|Δ|`; a 48-token stream
identical. Then report tok/s with the sanitised cache hit rate, because the
number is meaningless without it.

**Sizing:** slots come from the within-document curve in the handoff
(section 14.6): ~4211 slots (8 GiB) is 71 % hit / 145 MB per step / ~18 ms at
8.1 GiB/s; 6316 slots (12 GiB) is 82.6 % / 87 MB / 11 ms. Pick the size from
free VRAM at load time, not a constant.

### 3. Model wiring

Embedding row lookup and the Q6_K head (2048 -> 150144) on the device, with
device argmax so the step readback is 4 bytes.

**Exit:** end-to-end 48-token greedy stream identical to the host lane. Then the
one number that matters: tok/s versus the 7.4-7.8 tok/s host baseline, measured
in the same process, arms interleaved, every arm warmed.

### 4. Decide on speculative decoding

Flint's GGUF has no MTP/draft head (779 tensors, zero matches for
`mtp|nextn|draft|eh_proj`), so there is nothing to draft from. Either accept
plain decode, or add a tiny separate draft model — the latter is only worth it
if slice 3 lands above ~15 tok/s, otherwise the draft is the bottleneck.

**Not** an exit criterion: beating qwen's ratios. Flint is 40B-A2.5B with a 24.5
GB expert bank on a 16 GB card; the comparison is against its own host lane.

## P1 — The correctness gates that are still missing

| gap | why it matters | exit |
|---|---|---|
| No external oracle for Flint | It matches itself across prompts, which cannot catch a systematic error shared by host and device. | Greedy output vs llama.cpp on the same GGUF and tokenizer, or a documented reason the comparison is not possible. |
| q8 KV codes are not byte-identical across lanes | Vulkan divides to 2.5 ULP, the CPU is correctly rounded, so `.5` ties can differ by one code. The cache stays self-consistent today. | Only if a bit-identity gate needs it: pass a host-computed reciprocal into `quantize_q8_rows` instead of dividing on device. |
| Release build never run on the new workspace | `release.yml` builds and tests `--workspace --release` on tag. | Done in 0.2.1: `cargo build --workspace --release` (1m43s) and `cargo test --workspace --release` (34 suites, 3m44s) both clean with `CARGO_TARGET_DIR` off the vfat volume. Keep that env var in the tag recipe. |

## P2 — Performance after the device port lands

- **Dispatch count is the binding cost**: 823 dispatches per step, 6.2 us host
  + ~13 us GPU each. The candidates are `had_pre` (171, sequential — fold it
  into a neighbour, do not try to batch it) and the three `kv_append`
  dispatches per attended layer (30). Do not fuse `router_matvec` into
  `moe_topk` via a single workgroup; the arithmetic is in the handoff.
- **The 6.0 ms marginal spec column is structural**, not a batching miss:
  column `r+1`'s state depends on `r`. The lever is `gdn_step`'s per-step cost,
  which helps greedy too.
- **Report per regime, always**: loop 1.76x, code 1.48x, prose 1.11x, tech
  0.95-1.00x. Any headline that drops the tech row is wrong on purpose.
- Per-expert routing counters on 8 GB cards: `KORE_MOE_RESIDENT=N` pins the
  first N experts, not the hottest N. The counter exists in the route profile;
  the policy does not.

## P3 — Debt

- ~115 clippy warnings remain in *test* targets (unused names in cfg-gated
  helpers, mostly `kore-nn`, `kore-kernels`, `kore-vulkan`). CI does not gate on
  them because it runs `--workspace` without `--all-targets`. Worth clearing
  before the next contributor touches those tests.
- The host path is page-cache dependent: a cold pass is ~40 tokens in 60 s, and
  competing I/O drops it to ~1.4 tok/s. Documented, not fixed — it is a
  property of reading 24 GB from a SATA SSD.
- `0.2.0` was never tagged. Either tag it retroactively at the commit the
  changelog names, or leave it untagged and treat 0.2.1 as the first release of
  the current line. Decide before `release.yml` runs, because it triggers on
  `v*`.
- The PCIe switch upstream port negotiates Gen4 x16 (31.5 GB/s) while the card
  and the downstream port are Gen5. The ~9 GB/s software copy path is not the
  link. Recorded as a diagnostic; no plan attaches to it.

## Non-goals

Everything in the rejected table (handoff section 13) and section 14.7 stays
rejected unless the measurement that killed it is invalidated. That includes
layer splitting for Flint, full residency, sparse attention, and device-side
embedding handoff.
