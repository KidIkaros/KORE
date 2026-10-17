## Question

What KORE work directly raises **attention + KV-cache** throughput on the
user's backends (CPU SIMD / CUDA / ROCm / Vulkan)? The probe below pins the
bottleneck: it is NOT matmul GEMM.

Pinned invariant (verified 2026-10-16, RTX 2070 Max-Q 8 GB, Qwen35 Q2_0, Vulkan):

- Short-context decode is fine: **~80 tok/s** (median of 7 real runs).
- Long-context decode collapses to **~1.8 tok/s at 4096 context** (and 7.3 at
  1024, 3.8 at 2048). Each decode step's `fused_attention_decode_gqa` reads the
  ENTIRE KC/VC KV cache from VRAM — O(seq_len) VRAM traffic per token.
- Verdict: the attention + KV-cache memory path is the bottleneck. Matmul GEMM
  is already fast on the 8 GB card.

### Redirection (2026-10-16)

The straight "kernel performance" reading (more matmul GEMM) is closed out.
The KORE equivalent of Strata/sudoingX's wins is here: paged KV blocks, fused
decode attention, and KV-cache quantization so longer context fits in VRAM and
the per-token KV read no longer dominates decode.

### Correction (2026-10-17): the pinned numbers above do not reproduce

Re-measured on the same card under identical (AC-power) conditions:

- The "~80 tok/s" short-context figure is **not reproducible**: `HEAD` itself
  (F32 KV, pre-Q8) measures **33.7 tok/s** (median of 7 runs, ctx 5). The
  machine currently thermal-throttles (GPU ~1500 MHz of 2100, ~86 C, 50 W of an
  80 W limit) and `llama-server` holds ~3.2 GB RAM with swap exhausted.
- The long-context collapse is also far milder than recorded: `HEAD` F32 gives
  **28.7 tok/s @1024 and 21.0 @2048**, not 7.3 / 3.8.
- Net: treat the 80 / 22 / 7.3 / 3.8 / 1.8 curve as superseded. The *shape*
  (monotone decay with context) still holds; the absolute floor no longer does.

With the new INT8 KV path + repacked Q2_0 GEMV (this branch), A/B vs `HEAD`
under identical conditions: ctx5 **33.7 -> 41.2**, ctx1024 **28.7 -> 37.3**,
ctx2048 **21.0 -> 25.9** tok/s (+22-30%), with GPU output byte-identical to the
F32 reference.

## Claimed by

(unassigned)

## Type / mode

`wayfinder:task` (AFK). Ends when a profiling run + a KV/attention patch + a
before/after number exists for the bottleneck kernel, measured on the user's
actual hardware.

## Notes

The KORE attention module already owns:
- `kore-attention/src/kv_cache.rs` — dual CPU/GPU KV cache, F32 / Q8_0 / K8V4
  (asymmetric INT8-K + INT4-V) modes.
- `kore-attention/src/flash.rs` — FlashAttention-2 tiled attention (CPU).

The Vulkan backend already owns GPU decode shaders: `flash_decode.wgsl` /
`flash_decode_gqa.wgsl`. The gap is structural (whole-KV re-read per step,
no paged blocks, KV still in full F32 for most layers), not missing kernels.
