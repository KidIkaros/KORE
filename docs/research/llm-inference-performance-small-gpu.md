# LLM Inference on "Improbable" Hardware — KORE Research Findings

**What this file is:** a technique-category map (failure → fix) for maximizing tokens/sec on old, entry-level, or 8–16 GB VRAM consumer/professional GPUs (RTX 20/30/40/50, RX 6000/7000-class, and CPU). Every claim is pinned to an arXiv primary ID. Model names are not the subject — techniques are.

**Prerequisite link:** KORE is a pure-Rust ML engine (no C/C++ in the runtime path). It already owns **native 1.58-bit (ternary) kernels**, **paged KV-cache**, and **FlashAttention-2-style attention** (see §3). This file says which of the techniques below are *de-risked* and which are new model+kernel work for KORE.

---

## 1. Ranked technique table

Ranked by mass on 8–16 GB old/entry GPUs: rows 1–4 decide whether the model **fits**; rows 5–7 decide **tokens/sec** once it fits.

| # | Technique category | Masses gain on 8–16 GB | Failure it kills | Upgrade paper (arXiv) | Concrete implementation for KORE | Expected magnitude (primary source) |
|---|---|---|---|---|---|---|
| 1 | **1.58-bit ternary weight quantization** (BitLinear + sub-2-bit mpGEMM) | Weights drop 2.4–3.6× (700M: 2.08→0.80 GB; 3B: 7.89→2.22 GB). Decode latency 1.23–2.71× vs FP16 on 0.7–3.9B. Edge: up to 6.25× vs FP16 / 2.32× vs low-bit. | VRAM overflow of the teacher; weight-loading dominates per-token time on old GPUs (bandwidth-bound decode). | BitNet b1.58: arXiv:2402.17764 (ICLR 2024) · bitnet.cpp: arXiv:2502.11880 (ACL 2025) | kORE already has native ternary: `kore-kernels/cpu_ternary_matmul.rs`, `cpu_ternary_dequant.rs`, `cuda/dequant_matmul.cu`, `cpu_binary_matmul.rs`; Vulkan shaders `gemv_q2_0.wgsl`, `gemv_q4_k*.wgsl`, `quantized_matmul.wgsl`. Remaining: packed-format dequant inside the paged KV decoder for GQA. | Memory: −2.40× to −3.60×. Latency: +1.23×–2.71× (paper, 0.7–3.9B). bitnet.cpp reports up to 6.25× over FP16 / 2.32× over low-bit on edge CPUs (sub-2-bit, lossless). |
| 2 | **Dense 4-bit weight quantization** (post-training, accuracy-preserving) | 3-bit/4-bit weights → 2×+ capacity; a 13–175B model can run on 1–2 GPUs that otherwise need 80GB+ (A100/A6000). | Model-to-VRAM fit for non-ternary teachers (FP16/BF16) — the most common case for KORE's LLM-teacher path. | GPTQ: arXiv:2210.17323 (NeurIPS 2022) · AWQ: arXiv:2306.00978 (MLSys 2024) | No KORE PTQ path yet (only ternary). New work: quantize-on-load from safetensors → `.kwe`; CPU/Vulkan kernels for 4-bit dequant GEMM + per-channel scale; calibration-data loader. | GPTQ: ≈3.25× decode latency on A100, ≈4.5× on A6000 vs FP16; ≤0.25 PPL on OPT-175B at 4-bit, 0.3–0.6 at 3-bit. AWQ: >3× speedup over HF FP16 on desktop+mobile GPUs (4-bit, kernel fusion). |
| 3 | **3-bit KV cache quantization** | KV cache 4×–8× smaller → 4.8× longer context at <0.1 PPL degradation. | KV cache is the dominant memory cost on long context; OOMs on 8–16 GB during 10K–100K token prompts. | KVQuant: arXiv:2401.18079 (NeurIPS 2024) | `kore-attention/kv_cache.rs` already stores F32/Q8_0/K8V4. New work: KVQuant's 3-bit kernel (remove 1% of outliers; custom CUDA) into `kore-kernels`; integrate with the paged KV allocator. | Under 0.1 PPL degradation at 3-bit on Wikitext-2 and C4; 10M-context serving on 8× A100; 4.8× longer context for LLaMA-7B on one A100-80GB. |
| 4 | **KV cache eviction (long-context)** | Keeps near-SOTA quality with sub-linear cache on 8–16 GB for very long contexts. | Capped VRAM + growing KV cache; uniform retention wastes memory on the 20% of tokens that get 80% of attention. | H2O: arXiv:2306.14048 (NeurIPS 2023) | Policy module in `kore-serve`/`kore-attention`; maintain heavy-hitter index over token attention; evict non-hitters. KORE's B2 memory-balance loop already drops KV dtype (Q8→Q4) under pressure. | 20% of tokens receive ~80% of cumulative attention in a 512-token sequence; eviction with submodular-greedy guarantee; quality preserved on OPT/LLaMA/GPT-NeoX. |
| 5 | **Paged KV cache + continuous batching** (chunked prefill, prefix reuse) | 2–4× serving throughput; constant VRAM under fragmented lengths; decode layers fed while prefill runs. | Contiguous pre-allocation fragmentation; KV-cache memory grows with sequence length; prefill→decode pipeline bubbles on old GPUs. | PagedAttention: arXiv:2309.06180 (SOSP 2023) · SGLang chunked prefill: arXiv:2312.07104 (OSDI 2024) | Paged KV already present at Rust layer (`gpu_keys`/`gpu_values` pre-allocated at `max_seq_len`; block-granular allocation). Remaining: scheduler/block-manager, continuous-batching policy, chunked-prefill kernel, RadixAttention-style prefix reuse. | 2–4× throughput over contiguous allocation (vLLM's own measurement). SGLang reports up to 5× on complex LM programs via RadixAttention + interpreter. |
| 6 | **Speculative decoding** (tree-based) | 1.5–3.5× tokens/sec with identical output distribution; fewer parameter-memory passes per accepted token. | One forward pass per token is the whole per-token wall on old GPUs; decode rounds trip the full model weight + KV path each time. | Speculative decoding: arXiv:2211.17192 (ICML 2023) · SpecInfer: arXiv:2305.09781 (ASPLOS 2024) | Draft model (small SSM/student, 1–5% of teacher size) + verifier inside KORE's pure-Rust runtime (single process, no cross-process sync). Tree-based verify kernel on Vulkan; reuse paged KV cache across tree paths (SpecInfer DFS). | SpecInfer: 1.5–2.8× on distributed, 2.6–3.5× on offloading-based serving, identical output. Tree verify raises per-pass acceptance from ~52–57% (single-sequence) to 96–97% (tree). |
| 7 | **FA-3-style async + FP8 attention** | 1.5–2.0× decode on H100-class (75% util, 740 TFLOPS FP16; ~1.2 PFLOPS FP8); FP8 attention 2.6× lower error than per-tensor FP8. | Softmax/memory-bound attention overhead in decode — on old GPUs, FP8 kernels typically don't exist, so gain is partial. | FlashAttention-3: arXiv:2407.08608 (NeurIPS 2024) | `kore-attention/flash.rs` implements FA v2 (tiled, O(n)) but is **f32-only**; Vulkan has `flash_decode.wgsl` / `flash_decode_gqa.wgsl`. New work: FA-3 warp-specialized async + FP8 kernels on CUDA/Vulkan; keep FP8 KV (distinct from INT8 Q8_0) as a separate storage mode. | 1.5–2.0× on H100; 2.6× lower numerical error than per-tensor FP8 (block quantization + asynchrony). Old/low-VRAM GPUs: partial — depends on FP8 hardware + kernel availability. |

### Rank-order rationale
- **Rows 1–2** are the "fit" levers: on 8–16 GB, a 7B+ FP16 teacher literally cannot run; ternary (§1, row 1) and 4-bit dense (row 2) convert a capacity mismatch into a fit.
- **Rows 3–4** are the "context" levers: they decide how many tokens you can keep in flight on a fixed 8–16 GB budget (KV cache, not weights, is the growing cost with context).
- **Rows 5–6** are the "throughput" levers: they cut the per-token round-trip (paged+batched decode, speculative acceptance).
- **Row 7** is the most specialized: it exists because newer GPUs got FP8 + async hardware; on old cards it is only partially applicable.

---

## 2. Levers already de-risked for KORE

KORE already owns the primitive, so the map for each is **"apply the paper's algorithm, finish the kernel"**, not "reimplement from scratch":

| KORE primitive (file evidence) | Paper whose technique it implements | Remaining work |
|---|---|---|
| `kore-kernels/cpu_ternary_matmul.rs`, `cpu_ternary_dequant.rs`, `cuda/dequant_matmul.cu`, `cpu_binary_matmul.rs` | BitNet b1.58 (arXiv:2402.17764) — ternary {-1,0,1}, no multiplies | Finish the Vulkan quantized GEMV path for paged GQA decode (`gemv_q2_0.wgsl`, `gemv_q4_k*.wgsl` exist in `kore-vulkan/shaders/`); packed-format dequant |
| `kore-vulkan/shaders/flash_decode.wgsl`, `flash_decode_gqa.wgsl`, `kv_append_pair.wgsl`, `repack_q6_k.wgsl` | PagedAttention-style paged KV + flash-style decode (arXiv:2309.06180) | Scheduler/block-manager + continuous batching (chunked prefill, arXiv:2312.07104) are **not** in the design-phase list |
| `kore-attention/src/flash.rs` (FA v2: tiled, O(n), block 64) + `kv_cache.rs` (F32/Q8_0/K8V4; pre-allocated `gpu_keys`/`gpu_values` at `max_seq_len`) | FlashAttention-2 algorithm (arXiv:2307.08681) + PagedAttention memory management (arXiv:2309.06180) | `flash.rs` accepts only `f32` — no FP16/quantized attention path yet; FA-3 FP8 kernels are a separate missing piece |
| `kore-serve/state.rs` (layered sharding + prefetch), safetensors load (`kore-convert`, `kore-nn/serialization.rs`), CI guard (no C/FFI in runtime) | llama.cpp operator-sovereignty structural insights (borrowed, not copied — see design doc §4) | Serving policy (batch scheduler, stream prefetch) |
| Design doc `docs/inference-engine-design.md` (phases P0–P4) | KORE's own phased plan | Quantized KV = **DEFERRED (P4)**; speculative decoding = **P3 "nice to have"**; 4-bit dense PTQ = not planned |

**Verification note:** the design doc's "operator" scenario is literally a 9060 XT / 16 GB desktop, which is the "improbable hardware" target of this file.

---

## 3. New levers needing new model + kernel work (not yet in KORE)

| Lever | Why it needs new work | What to build |
|---|---|---|
| 4-bit dense weight quant (GPTQ/AWQ) | KORE has only **ternary**; the FT-1 / calibration / outlier-removal steps are model-specific | PTQ pipeline on safetensors → `.kwe`; 4-bit dequant GEMM kernels (CPU + Vulkan); accuracy gate vs HF reference |
| 3-bit KV cache (KVQuant) | KORE's KV storage is Q8_0 (INT8) / K8V4 (INT8+INT4); 3-bit kernels are not implemented | KVQuant-style 3-bit KV kernel in `kore-kernels`; add "remove 1% of outliers" pass; validate <0.1 PPL |
| H2O eviction policy (arXiv:2306.14048) | Not in KORE; the KV cache grows by token append, no eviction | Heavy-hitter index + sub-linear eviction policy + B2 memory-balance integration |
| Speculative decoding (arXiv:2211.17192, arXiv:2305.09781) | No draft model path in KORE; verified only as an abstract P3 item | Small draft SSM (or quantized teacher-student pair) inside KORE runtime; tree-based verifier kernel; paged KV reuse across tree paths |
| FA-3 FP8 attention (arXiv:2407.08608) | FA v2 in KORE is f32-only; no FP8 path anywhere | FP8 attention kernel (Vulkan/CUDA); FP8 KV storage as separate mode; verify 2.6× lower error vs per-tensor FP8 |
| Chunked prefill / continuous batching | Not in the design doc at all | Scheduler (block-grained), prefill/decode pipelining kernel, prefix-reuse (RadixAttention) |

---

## 4. Measurement protocol — how to make "massive" a number on old/low-VRAM GPUs

The KORE design doc's own metric ticket (`docs/wayfinder/tickets/define-performance-metric.md`) asks exactly this. What follows is the protocol, with the physical reason why each item matters for 8–16 GB old GPUs.

1. **Correctness first, throughput second.** No throughput claim ships until the load path is green: token-for-token match vs the HF reference on a fixed prompt (design doc P0 assertion). Report the model, dtype, and KV dtype so the claim is reproducible.
2. **Report two numbers, never conflate them.** Absolute tok/s on the user's GPU *and* speedup vs the current KORE build on the same hardware. They answer different questions (raw capability vs improvement).
3. **Pin the hardware and software.** GPU name + VRAM; driver version; CUDA or Vulkan API version; build profile (release); and the Rust toolchain if relevant. Old GPUs mean the driver/FP64-FP32 ratio and the PCIe bandwidth matter — report them.
4. **Pin the workload.** Batch size (1 for decode), prompt length, max new tokens, context length, temperature (0 for deterministic quality), per-layer dtype (f16/bf16/q4/q8), KV dtype (F32/Q8_0/K8V4), and whether the model is ternary (BitNet family) or dense-quantized (GPTQ/AWQ). Old GPUs are bandwidth-bound at decode, so the dtype mix *is* the throughput number.
5. **Report phases separately** — prefill tok/s (prompt processing), decode tok/s (steady state), end-to-end — and VRAM at steady state (peak, not just allocation start). A technique that fits a bigger model but stalls decode at 1 tok/s is not "massive".
6. **Report bandwidth-bound arithmetic.** On 8–16 GB old GPUs, per-token time ≈ weight-loading time (model weights are read from DRAM per token). This is precisely why rows 1–2 (weight memory traffic) dominate here: GPTQ's own numbers are latency, not FLOPs — 3-bit OPT-175B went 589 ms → 130 ms per token on 2× A6000 (arXiv:2210.17323). Beautify the "token dead-end" framing: KORE measures **time from idea → apples-to-apples eval vs teacher** (design doc R0), not a leaderboard.
7. **Acceptance bar from the docs.** Design-doc assertions are decidable: 27B-class ternary load + fit in 16 GB; 4K context sustained within budget; correctness gate green before any perf PR merges (Meadows #5).

---

## 5. Primary-source register (the ground truth for every row)

| Claim being pinned | Primary ID | Venue / note |
|---|---|---|
| Speculative decoding — sample faster without output changes, multiple tokens per forward pass | **arXiv:2211.17192** | ICML 2023 Oral |
| SpecInfer — tree-based verification; 1.5–2.8× distributed / 2.6–3.5× offloading, same output | **arXiv:2305.09781** | ASPLOS 2024 |
| PagedAttention — 2–4× serving throughput; block-based non-contiguous KV cache | **arXiv:2309.06180** | SOSP 2023 (vLLM) |
| KVQuant — <0.1 PPL at 3-bit KV; 4.8× longer context; 10M ctx on 8× A100 | **arXiv:2401.18079** | NeurIPS 2024 |
| GPTQ — 3.25× decode latency (A100), 4.5× (A6000); ≤0.25 PPL at 4-bit | **arXiv:2210.17323** | NeurIPS 2022 |
| AWQ — >3× speedup over HF FP16, desktop+mobile GPUs, 4-bit with per-channel scale | **arXiv:2306.00978** | MLSys 2024 |
| SqueezeLLM — lossless to 3-bit; 0.05%/0.45% sparse | **arXiv:2306.07629** | ICML 2024 |
| BitNet b1.58 — ternary {-1,0,1}; matches FP16 perplexity; 2.4–3.6× memory cut, 1.23–2.71× latency on 0.7–3.9B | **arXiv:2402.17764** | ICLR 2024 |
| bitnet.cpp — up to 6.25× over FP16, up to 2.32× over low-bit; sub-2-bit lossless mpGEMM | **arXiv:2502.11880** | ACL 2025 (long paper 457) |
| FlashAttention-3 — 1.5–2.0× on H100, 740 TFLOPS FP16 (75% util), ~1.2 PFLOPS FP8; 2.6× lower error than per-tensor FP8 | **arXiv:2407.08608** | NeurIPS 2024 |
| FlashAttention-2 — 2–4× wall-clock vs FlashAttention, linear memory | **arXiv:2307.08681** | ICLR 2024 |
| SGLang — up to 5× throughput/latency on complex LM programs via RadixAttention + chunked prefill | **arXiv:2312.07104** | OSDI 2024 |
| H2O — KV cache eviction; ~80% of attention in 20% of tokens; submodular-greedy guarantee | **arXiv:2306.14048** | NeurIPS 2023 |

### Corrections flagged against the brief's seed list
1. **bitnet.cpp is arXiv:2502.11880** (ACL 2025, long paper 457) — the brief's 2410.16144 does not resolve to the bitnet.cpp paper. The primary source is the ACL Anthology (DOI 10.18653/v1/2025.acl-long.457) and arXiv:2502.11880.
2. KORE's FlashAttention-2 (`kore-attention/src/flash.rs`) is present as an **algorithm** but the exposed `flash_attention()` guard is **f32-only**; FP16/quantized attention is not yet on the path.
3. Per the design doc, **quantized KV (P4), dense 4-bit PTQ, and speculative decoding (P3)** are not yet implemented — they are new model+kernel work, not de-risked primitives.
