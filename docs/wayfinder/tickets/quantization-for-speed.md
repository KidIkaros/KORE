## Question

Which quantization scheme and bit-width **maximizes tokens/sec at an acceptable
quality loss**? Evidence-only ticket; subagent gathers, I verify before closing.

Candidates (each with the supporting literature):
- 2-bit ternary / quaternary (KORE's native `kore-btes`; Strata's Q2_0/IQ2_XS/IQ3_S;
  sudoingX's PTQ1_0 ternary) - likely the biggest size-to-speed win but quality risk.
- 4-bit (GPTQ / AWQ / K8V4-style) - the safer mid-point.
- 8-bit - quality-safe, slower speed gain.
- Weight-only vs weight+K V quantization, and whether to combine (KVQuant's 3-bit KV).

What to decide: which scheme the KORE kernel strategy should commit to first,
against what quality bar, and how that bar is pinned (e.g. ">=98% same top token
as FP16" or "<=x perplexity delta").

## Claimed by

(unassigned)

## Type / mode

`wayfinder:research` (AFK). Subagent gathers; I verify against sources and record.

## RESOLVED (2026-10-16) — verified on disk

Concluded from the research file `docs/research/llm-inference-performance-small-gpu.md`
(subagent result, artifact confirmed 95 lines / ~15 KB on disk).

- **Greatest lever — native 1.58-bit ternary (BitNet b1.58):** arXiv:2402.17764 +
  bitnet.cpp arXiv:2502.11880 (ACL 2025). Memory −2.40× to −3.60×; decode latency
  +1.23×–2.71× (0.7–3.9B), up to ~6.25× over FP16 on edge CPUs. KORE already owns the
  path: `kore-kernels/cpu_ternary_matmul.rs`, `cpu_ternary_dequant.rs`,
  `cuda/dequant_matmul.cu`, `cpu_binary_matmul.rs`, Vulkan `gemv_q2_0.wgsl`.
- **Largest new-work levers:** dense 4-bit PTQ (GPTQ arXiv:2210.17323, AWQ
  arXiv:2306.00978) — KORE has only ternary, so PTQ pipeline + 4-bit dequant GEMM are
  new; and FA-3 FP8 attention (arXiv:2407.08608) — but KORE's `flash.rs` is f32-only,
  so FP8 attention is a separate missing piece.
- Ranked: (1) ternary weights, (2) 4-bit dense PTQ, (3) 3-bit KV (KVQuant), (4) KV
  eviction (H2O), (5) paged KV + continuous batching (PagedAttention/SGLang),
  (6) speculative decoding (Leviathan/SpecInfer), (7) FA-3 FP8 attention.
- Seed-brief citation for bitnet.cpp was corrected: 2410.16144 is a different paper;
  bitnet.cpp is arXiv:2502.11880.
