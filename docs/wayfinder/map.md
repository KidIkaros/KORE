# KORE wayfinding map

## Destination

A decision set (a plan) that tells KORE **how to deliver massive performance gains
and raised tokens/sec output** — measured on the user's actual hardware, with the
success metric and measurement protocol settled *before* any kernel is optimized,
and with the quality-vs-speed trade pinned to a number instead of a feeling.

This is a **speed** effort, not a "make it fit a small card" effort. The old
device-framing (8-16 GB gaming-card fiction) is closed out.

## Notes

Domain: ML inference on (unknown-from-the-map) hardware; pure-Rust ML engine (KORE).
Skills this session consults: wayfinder (this), grilling, domain-modeling, plus prior
research (Strata + sudoingX repos and the supporting papers).
Standing: pin invariants before deleting; no parallel systems (no second keying or
second benchmarking scheme); verified claims only; decision-by-decision, not
build-first.
Tracker: local markdown default (no GitHub tracker provided for KORE). Update via
body-file, never inline `--body`.

## Decisions so far

- [LLM Inference on "Improbable" Hardware — KORE Research Findings](docs/research/llm-inference-performance-small-gpu.md) (verified 2026-10-16). The speed/technique map for massive performance on 8–16 GB GPUs. Key result: **native 1.58-bit ternary (BitNet b1.58)** is the single greatest lever and KORE already owns it; the largest *new-work* levers are **dense 4-bit PTQ (GPTQ/AWQ)** and **FP8 FA-3 attention**. See the file's [ranked table](#ranked-technique-table) and [measurement protocol](#measurement-protocol).
- **Real-run verification (2026-10-16, RTX 2070 Max-Q 8 G b)**: Qwen35 Q2_0 on the Vulkan back-end. 7/7 steady-state runs >= 72.20 tok/s — max 81.82, median 80.52, min 79.81. The dry-run's 72.20 was a first-call figure; warm steady-state is ~80.5 tok/s, above target on every run. Baseline captured. Repeatable ladder: `kore-tools/run-kore-dryrun.sh` (README has the Kaggle recipe).

## Frontier

- **Define the performance metric + measurement protocol** (wayfinder:grilling) -
  conversation-first: what number is "massive," and how is it measured (raw
  tokens/sec on the user's hardware vs FP16 baseline, vs current build)?

- **Benchmark harness + baseline** (wayfinder:task) - calibrated harness + baseline
  tokens/sec on the actual hardware before any optimization. Depends on the metric.

- **Kernel performance (matmul GEMM + attention)** (wayfinder:task) - CPU SIMD +
  CUDA/ROCm/Vulkan kernels, fusion, thread-block clusters; pin invariants.

- **KV-cache + paged KV + KV quantization** (wayfinder:task) - larger batches and
  longer context without OOM; cut memory-bound stalls.

- **Speculative decoding (MTP)** (wayfinder:task) - raise effective tokens/sec with a
  cheap draft head; lossless.

## Not yet specified

- User's actual hardware (CPU/GPU/VMM: CUDA or ROCm or neither) - to be settled by
  the baseline benchmark, not assumed.
- What "massive" reads as numerically (candidate: >=2x the current build on the
  same hardware) - decision, not a guess.
- Acceptable quality loss at the target tokens/sec - decision, not a guess.

## Out of scope

- **The 12 G b / 8-16 G b device fiction** + project. The original map assumed an
  8-16 G b gaming card, and the user has no 12 G b card on the laptop. The tickets
  `engine-12gb-card-support`, `turnkey-packaging-and-trust`, and `serve-endpoint-and-ui`
  were built on that fiction - closed, never graduated, never in `Decisions so far`.
- Executing and shipping the plan (code, binaries, serving to users). This map
  resolves decisions only.
