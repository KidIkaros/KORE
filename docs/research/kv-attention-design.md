# KORE — KV-cache / attention struct: what it needs to deliver on 8 GB

> Grounded in the pinned measurement (2026-10-16, RTX 2070 Max-Q 8 GB, Qwen35
> Q2_0, Vulkan) and the Strata/sudoingX technique map.
> Related: `docs/research/llm-inference-performance-small-gpu.md`.

---

## The pinned failure

| context | decode tok/s |
|---|---|
| 5 | ~80 (median, 7 real runs) |
| 128 | 22 |
| 1024 | 7.3 |
| 2048 | 3.8 |
| 4096 | 1.8 |

At 8 GB the short-context matmul path is fine (~80 tok/s). The bottleneck is the
**attention + KV-cache memory path**: every decode step's
`fused_attention_decode_gqa` re-reads the ENTIRE KV cache (KC/VC) from VRAM.
KC and VC are allocated as `Tensor::zeros(&[max_ctx, kv_dim], F32)` per layer, and
the GQA shader reads K/V as `array<f32>` storage buffers. At seq_len=4096 that is
~1.6 GB VRAM read per token per decode step (before any heads/weights).

This is an **O(seq_len) VRAM output per token** problem, not a compute problem.
On a Max-Q GPU the per-token KV read + the decode loop's implicit sync is what
caps decode at ~2 tok/s.

## What KORE needs (the fix)

The KORE attention module already owns the ingredients:

- `kore-attention/src/kv_cache.rs` — CPU+GPU dual-buffer KV cache, F32 / Q8_0 /
  K8V4 (asymmetric INT8-K + INT4-V) modes.
- `kore-vulkan/src/backend.rs` — GPU decode shaders, including
  `flash_decode.wgsl` / `flash_decode_gqa.wgsl` (fused, workgroup-tiled, GQA).

But the GPU path never uses them: KC/VC stay F32, and the GQA shader reads
unquantized K/V. The fix is structural, in this order:

### 1. K8V4 by default on the GPU path

Store KC/VC as **K8V4** on the device: INT8 keys + Hadamard-rotated INT4 values
(1.25 bytes/token vs 8 bytes for F32 → ~6.4× less VRAM read traffic).
Dequantize K and V **per-thread inside `flash_decode_gqa.wgsl`** (read a quantized
row, rotate, scale) instead of reading unquantized float arrays. This is the
single largest win for long-context decode on 8 GB.

- Keep the existing K8V4 CPU/quant + `dequantize_kv()` code (it already exists);
  make the GPU `kc`/`vc` buffers hold the K8V4 representation and pass it to the
  shader together with per-row scales (`k_scales`, `v_scales`).

### 2. Paged/block KV allocation

Replace the single `[max_ctx, kv_dim]` F32 buffer with **fixed-size pages (blocks)**
plus a block table (vLLM SOSP 2023 style). Benefits:

- No forced `max_ctx` pre-allocation; memory scales with actually-kept context.
- Drop/shift of whole pages slides a sliding window without reallocating.
- Enables multi-block attention so a long-context step reads block A (history)
  + block B (new token) instead of one giant contiguous tensor.

### 3. Keep and extend fused decode attention

`flash_decode_gqa.wgsl` is already the right shape (GQA, 8-tiled scoring, fused
Q×K + softmax + V-accumulate). Extend it to:
- dequantize K/V inline (single-pass, no second KV round-trip),
- stream pages from VRAM with coalesced reads,
- hide memory latency behind the next layer's small matmul.

### 4. Verify the gain on the pinned measurement

Re-run the exact probe (5 / 128 / 1024 / 2048 / 4096 context, 64 generated
tokens each) after the patch. Target: long-context decode should move from
~1.8 tok/s (4096) toward double-digit tok/s, with short-context decode **not
below 80 tok/s** (the existing baseline must hold).

## Sources

- Strata (Niko1221): native 1.58-bit ternary + fused prompt kernels, small-card
  inference (see `docs/research/llm-inference-performance-small-gpu.md`).
- sudoingX (`local-ai-on-your-gaming-gpu`): community PR sweeps for kernels and
  small-card optimization.
- Pinned measurement above (RTX 2070 Max-Q 8 GB, Qwen35 Q2_0, Vulkan).
