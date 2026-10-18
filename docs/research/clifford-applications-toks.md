# Clifford / hypercomplex algebra and tokens/sec on existing Hugging Face models

> Survey date: 2026-10-18. Scope: **every** application of Clifford, quaternion,
> octonion, rotor, or hypercomplex algebra that could raise decode throughput on
> models that already exist on Hugging Face — not only inside KORE's engine.
>
> Companion measurements: arm A/B in `crates/kore-kernels/examples/clifford_probe.rs`
> and `kore-tools/clifford-quant-probe.py` (arm C added here); verdict in
> `~/.opencode/plan/kore-inference-revamp.md` §14.21 and §14.22.
> Related: `docs/research/kv-attention-design.md`.

---

## 0. The answer in one paragraph

Clifford algebra has **two** roles that survive contact with an existing
checkpoint, and they are both the *same* underlying object: an orthogonal
transform used to make low-bit quantization survivable.

1. **Rotation before low-bit weight quantization.** Applicable post-hoc, folds
   into the weights offline, costs ~0.3% at runtime. This is the largest
   tok/s lever available to KORE on dense models.
2. **Rotation before low-bit KV-cache quantization.** Applicable post-hoc,
   and — because an orthogonal transform commutes with the attention weighted
   sum — free if applied at append time. This is the largest tok/s lever at
   long context.

Everything else that is genuinely Clifford — quaternion attention, conformal-GA
sequence models, Clifford SSMs, rotor-regenerated MoE experts, Hurwitz
multiplicative codebooks — either **requires training** (so it cannot touch an
existing HF checkpoint) or **loses to a cheaper rotation at matched bits**
(measured here, §5). The algebra that pays is the orthogonal/Hadamard factor.
The Clifford-specific vocabulary adds no bits, no flops, and no quality.

The one place a Clifford-native *object* genuinely wins is narrow: on a
4-element chunk with heavy tails, the Hurwitz 24-cell codebook beats a
4-wide rotation + independent scalars by **18%** (§5.2). It loses to a
128-wide rotation by **11%**. So the codebook is a good answer to a question
we do not have to ask.

---

## 1. Method and evidence grades

Web search was unavailable; the survey ran against the arXiv API, the Hugging
Face models API, and the GitHub API, plus direct reads of paper abstracts and
repository READMEs. ~40 distinct queries across `quaternion`, `Clifford`,
`geometric algebra`, `rotor`, `hypercomplex`, `octonion`, `Hurwitz`, `Kronecker`,
`Monarch`, `butterfly`, `rotation + quantization`, `Hadamard + KV cache`.

Grades used below:

| grade | meaning |
|---|---|
| **A** | peer-reviewed **and** independent code/reproduction exists |
| **B** | preprint with public code, or reviewed without independent reproduction |
| **C** | preprint or community claim, no independent reproduction |
| **D** | vendor/README claim, unreviewed, unmaintained, or contradicted elsewhere |

Also useful: **F** = falsified by our own measurement in this repo.

---

## 2. Ranked map of every application found

"Applies to existing HF weights?" is the decisive column. Anything marked **no**
is an architecture, not a lever.

| # | Family | Representative work | Applies to existing weights? | Effect on bytes/token | Effect on compute | Grade | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | Rotation + low-bit **weights** | QuaRot, SpinQuant, FlatQuant, GyRot, ThinQuant, SpectrumQuant | **yes** | **−25…−50%** (4→3 or 2 bits) | ~0 (folded; +0.3% activation rotate) | A | **adopt** |
| 2 | Rotation + low-bit **KV** | SAW-INT4, KVarN, KVLinC, VecInfer, UltraQuant, PolyKV, TurboQuant | **yes** | **−50%…−80%** vs f16 | 0 if rotated at append; +7× if rotated per key in-kernel | A/B | **adopt** |
| 3 | **Quaternion / SO(4) block** rotation for KV | IsoQuant, PlanarQuant, RotorQuant | **yes** | same as #2 | rotation is O(d), 512 vs 16 384 FMAs at d=128 | B/C | adopt *only* if rotating inside the attention kernel |
| 4 | **Hurwitz multiplicative codebook** for KV | HQMQ (arXiv:2605.27646) | yes | 5.05× at 3.17 b/e, ~3.8–5.2 b/e | **2.5× slower than f16 SDPA** | C | **F** on error at matched rate; **reject** on cost |
| 5 | Vector-quantized KV (spherical / Lloyd-Max) | VecInfer, PolyKV, TurboQuant centroids | yes | as #2 | codebook lookup, latency-bound | B | dominated by #2 |
| 6 | Rotation + lattice + entropy coding | HyperQuant (RHT → E8 → Rice) | yes | 3.79× KV, 3.9× weights at 4 bps | variable-length decode, bad for SIMD | C | reject (format) |
| 7 | **Quaternion self-attention** (shared scores) | arXiv:2605.24920, ICML 2026 | **no** — needs quaternion-trained Q/K | 0 | −44.3% GPU inference | B | not applicable |
| 8 | **Conformal-GA sequence model** | Versor (arXiv:2602.10195) | **no** | 200× fewer params | 1.05 ms/step, "100×" kernels | C | not applicable |
| 9 | Clifford SSM language models | `MonumentalSystems/clifford-ssm-checkpoints` | **no** | — | — | D | not applicable |
| 10 | **Rotor-generated MoE experts** | FORGE (arXiv:2610.16890), ButterflyMoE (arXiv:2601.13563) | **no** — trained | 80–150× | regenerated on-chip | C | not applicable, but the *idea* is the only real answer to 8 GB MoE |
| 11 | Quaternion block-circulant layers | EdgeLDR (arXiv:2601.05379) | **no** — trained | large | FFT | C | not applicable |
| 12 | Kronecker / Monarch / butterfly structured weights | Monarch, "Fast Inference with Kronecker-Sparse", KronQ | **no** post-hoc (needs training or recovery FT) | large | O(d log d) | B | not applicable post-hoc |
| 13 | Kronecker-factored **Hessian** for quantization | KronQ | yes (as a GPTQ accelerator) | 0 | faster calibration only | C | not a tok/s lever |
| 14 | GA-native kernels / accelerators | Versor bit-masked contraction, "Fast Clifford Neural Layers" | **no** (dense algebra is already dense) | 0 | **113–119× slower** in Cartan–Dieudonné form | **F** | reject |
| 15 | Rotor / quaternion embeddings, GeoPE | CARE (arXiv:2511.11665), GeoPE | **no** — trained | — | — | C | not applicable |
| 16 | Rotation-aware **QAT / QLoRA** | GyRot, "When Quantization Is Free" ablations | yes (needs a short fine-tune) | enables −25…−50% | 0 | B | fallback if #1 loses quality |

**Count:** 16 families, 4 that apply post-hoc to an existing checkpoint (#1, #2,
#3, #4), of which #1 and #2 are the only ones with a direct tok/s mechanism, and
#3/#4 are variants of #2 that lose on cost or error.

---

## 3. The two leaked numbers nobody quotes

Three independent 2026 results say the same thing, and it is the single most
useful fact in this survey:

1. **KV quantization can be faster than fp16, not just smaller.**
   *When Quantization Is Free* (arXiv:2605.05699) runs a fused int4 KV kernel
   and gets **−3 to −8% ms/token vs fp16** on Gemma-3 1B across 256–4096-token
   prefixes, with 3× persistent compression, at ~25 ns/vector kernel overhead.
   The overhead is *below* the bandwidth saved.

2. **It can also be much slower — and the difference is implementation, not
   math.** vLLM's TurboQuant reports **40–52% lower throughput** for 2.3–3.7×
   capacity. KVarN (arXiv:2606.03458) does Hadamard + dual-axis variance
   normalization and instead reports **up to ~1.3× fp16 throughput** at 3–5×
   capacity, and ~2.4× TurboQuant's throughput at the same capacity.

3. **HQMQ's own kernel is 2.5× slower than fp16 SDPA.** Its 5× memory win is
   real; its *speed* win is negative. On a capacity-bound box that can be a good
   trade. On a bandwidth-bound decode step it is a loss.

**Consequence for KORE:** the metric to optimize is not bits/element, it is
`bytes read per token ÷ achieved bandwidth`, and the dequant must be fused into
the kernel that consumes it. A format that halves bytes but doubles kernel
latency is a net loss.

---

## 4. What the shipped ecosystem actually converged on

This is the strongest evidence in the survey, because it is post-selection.

| system | rotation | quant | status |
|---|---|---|---|
| **llama.cpp mainline** | **Hadamard on KV activations, on by default since Apr 2026** (PR #21038, opt-out `LLAMA_ATTN_ROT_DISABLE=1`) | `q8_0` / `q4_0` KV | shipped |
| **KVarN** (Huawei CSL, ~510★) | Hadamard + **dual-axis variance normalization** | `k4v2`, tile 128 | shipped in vLLM; **supports MLA, hybrid linear-attention, and MTP spec decode** |
| **SAW-INT4** (arXiv:2604.19157) | **block-diagonal Hadamard** | token-wise INT4 | fused into paged KV, *zero measurable end-to-end overhead* |
| **KVLinC** (arXiv:2510.05373) | Hadamard + linear correction adapters | 2-bit | custom kernel, up to 2.55× faster than FlashAttention |
| **RotorQuant / PlanarQuant / IsoQuant** | 3D rotor / 2D Givens / 4D quaternion | 3–4 bit | llama.cpp **fork, unmaintained**; the TurboQuant PR was **closed unmerged** (Jun 2026) |
| **HQMQ** | Hurwitz 24-cell × random codebook | ~5 bit | 2★, research code |

The trajectory is unambiguous: **the Clifford rotor was an intermediate step
that got simplified to 2D/4D block rotations, and then mainline shipped the
plain Hadamard.** The RotorQuant README says so itself:

> "Each step traded algebraic richness for speed. The PPL results show the
> simpler rotations work *better*."

Its own matched-bit table on Llama-3-8B Q4_K_M at 10.3× compression:

| KV config | decode tok/s | prefill tok/s | PPL | grade |
|---|---:|---:|---:|---|
| f16 / f16 | 140 | 6 156 | 6.63 | — |
| **iso3 / iso3** (4D quaternion) | **118** | 3 397 | **6.91** | C |
| **planar3 / planar3** (2D Givens) | **119** | **3 822** | 7.05 | C |
| turbo3 / turbo3 (128-wide WHT) | 93 | 722 | 7.07 | C |
| **planar3 / f16** (K-only) | **134** | — | **~6.63** | C |

Read it carefully, because it is usually misread:

- At equal 10.3× compression the **4D quaternion is 2.3% better PPL than the
  full WHT** (6.91 vs 7.07) and **28% faster decode / 4.7× faster prefill**.
  That is a genuine, if small, quality win for the quaternion.
- But **2D Givens matches it on speed and nearly on PPL**, for half the FMAs.
- And **rotating K only, leaving V at f16, is lossless and already faster than
  f16 decode** (134 vs 140 tok/s) at 5.1× K compression. That is the best
  risk-adjusted configuration in the table.
- On Qwen2.5-3B the same K-only config is **367 tok/s vs 356 fp16** on a 5090
  and **48.3 vs 47.4** on an M4 — i.e. strictly faster.

---

## 5. New measurement: rotation width, and the HQMQ codebook

Arm B tested a *uniform-angle spherical grid*, which is the weakest possible
spherical quantizer — so it did not falsify HQMQ, whose scheme is a
**multiplicative codebook**: `{q_p · q_s}` with `q_p` ranging over the 24
Hurwitz units (vertices of the 24-cell, the binary tetrahedral group `2T`) and
`q_s` over `S` random unit quaternions, giving `24S` codewords on `S³` from `S`
stored parameters.

Arm C implemented it exactly, with honest rate accounting, and added the two
rotation widths the KV literature argues about.

### 5.1 Setup

Every arm is scored at **one shared rate convention**: `b` adjustable bits per
value + one f16 scale per 128-wide block = `b + 0.125` bits/weight. Every
transform is orthonormal and fixed, so its parameters cost zero stored bits —
the same assumption the Hadamard arm has always made.

| arm | transform | cost at d=128 |
|---|---|---:|
| `AFF128s` | none | 0 |
| `AFF2p` | 2D Givens, per pair | 256 FMAs |
| `AFF4q` | SO(4) isoclinic `T(v) = q_L v conj(q_R)`, per 4-chunk | 512 FMAs |
| `AFF128h` | 128-wide normalized Hadamard | 16 384 FMAs (896 butterflies) |
| `HQMQ` | none explicit; 4-chunk direction → `q_p · q_s` | 24S-entry lookup |

`HQMQ` rate = `(⌈log₂ 24S⌉ + b_r)/4 + 16/128`. Two real tensors, 1 Mi values
each: POCKET-35B f32 router weights (block excess kurtosis **+4.77**,
heavy-tailed) and Ornith-9B `blk.0.ffn_down` Q6_K (**+1.76**, near-Gaussian).

### 5.2 Result — relative RMSE, lower is better

**POCKET router weights** (heavy-tailed):

| scheme | rate | rel RMSE |
|---|---:|---:|
| `AFF128s` (no rotation) | 4.125 | 19.26% |
| `AFF2p` (2-wide) | 4.125 | 17.86% |
| `AFF4q` (4-wide quaternion) | 4.125 | 15.65% |
| `HQMQ2304` | 4.125 | 12.82% |
| **`AFF128h` (128-wide Hadamard)** | **4.125** | **11.43%** |

**Ornith `ffn_down`** (near-Gaussian):

| scheme | rate | rel RMSE |
|---|---:|---:|
| `AFF128s` | 4.125 | 12.16% |
| `AFF2p` | 4.125 | 12.06% |
| `AFF4q` | 4.125 | 11.93% |
| **`AFF128h`** | **4.125** | **11.70%** |
| `HQMQ2304` | 4.125 | 12.55% |

Three findings:

1. **Rotation width matters monotonically, and wide wins on weights.**
   On heavy tails, 128-wide cuts error **41%** vs no rotation (19.26 → 11.43),
   4-wide cuts 19%, 2-wide cuts 7%. On Gaussian data the whole effect is 4%,
   confirming that rotation only pays where kurtosis is high — the same
   conclusion as arm B, now with the width axis resolved.

2. **HQMQ's codebook is a real win over a *narrow* rotation.** At equal
   locality and equal rate, `HQMQ2304` beats `AFF4q` by **18%** on
   heavy-tailed data (12.82 vs 15.65). That is a genuine, previously-unmeasured
   point in favor of the Hurwitz construction: on a 4-chunk whose information is
   in its direction, a spherical code is more efficient than four independent
   scalars after a 4-wide mix.

3. **…and a clear loss to a *wide* one.** Comparing at exactly matched rate
   (interpolating `AFF128h` log-linearly between its measured points):

| rate | HQMQ (router) | `AFF128h` (router) | HQMQ (ffn_down) | `AFF128h` (ffn_down) |
|---:|---:|---:|---:|---:|
| 3.375 | 22.20% | **21.60%** | **20.43%** | 22.11% |
| 4.125 | 12.82% | **11.43%** | 12.55% | **11.70%** |
| 4.875 | 7.86% | **6.45%** | 7.78% | **6.60%** |

HQMQ wins only at the lowest rate on near-Gaussian data, and loses
progressively as rate rises (−11%, then −18% at 4.875 on routers). That is the
signature of a **covering-radius** limit, not a packing limit: the product set
is `S` rigid isometric copies of one 24-point code, so its covering constant is
fixed by the 24-cell and `S` only buys packing. Scalar quantization after a wide
rotation improves its covering without bound as bits grow; a fixed-shape
codebook does not.

### 5.3 Verdict on HQMQ

**Falsified as a general inference quantizer**, on two independent axes:

- **Error:** at matched rate it loses to a 128-wide Hadamard + scalar on both
  tensor families above 3.4 bits.
- **Cost:** it needs `⌈log₂ 24S⌉ ≈ 12–14` bits of *table lookup per 4 values*,
  i.e. 3–3.5 dependent lookups per element, versus one shift + one FMA per
  element for scalar. Table lookups are latency-bound and defeat vectorization.
  This is exactly why HQMQ's own kernel is 2.5× slower than dense fp16 SDPA.

It remains correct that `24S` codewords cost `S` parameters, and correct that
random initialisation works (left-multiplication is an `S³` isometry). Those are
elegant. They are also irrelevant to a kernel that has to run at memory speed.

---

## 6. The two design rules this survey produces

### Rule 1 — rotation width is set by the cost of the work it feeds, not by quality

The probe says wider is better; the KV literature says 4-wide is better. Both
are right, because they feed different consumers.

| consumer | work per element | cost of a full-width transform | affordable width |
|---|---|---|---|
| weight GEMV, `O(d²)` | `d` MACs/weight | `log₂d / d ≈ 0.3%` | **full width (128 / row)** |
| KV attention dot, `O(d)` per (q,k) | 1 MAC/pair | `log₂d ≈ 7×` | **≤4 (quaternion / Givens)** |
| KV **appended once, transform commutes with the sum** | amortized | `log₂d / ctx` | **full width — again free** |

The third row is the one that matters for KORE and the one the literature
under-uses. Because rotation is orthogonal and attention weights are scalars,

```
Σ_j α_j (R v_j)  =  R (Σ_j α_j v_j)
```

so you can rotate **V at append time and un-rotate the single output vector**,
and rotate **q once per query** so that `(Rq)·(Rk) = q·k`. Neither costs anything
per (query, key) pair. **Full-width rotation of both K and V is therefore free
in an append-time design**, which is strictly better than the 2D/4D compromise
that llama.cpp's *in-kernel* dequant forces.

### Rule 2 — rotation and per-token scaling are not substitutes

Rotation fixes *channel-aligned static* outliers; a per-token (per-row) absmax
scale fixes *per-token dynamic* outliers. Modern outlier-heavy models need both
— that is the whole finding of SAW-INT4 and of KVarN's dual-axis variance
normalization (`k4v2` = 4-bit K, 2-bit V, tile 128). Our own arm-B/C data agrees:
the Hadamard's benefit is exactly the kurtosis, and kurtosis is a per-block
property.

---

## 7. What this means for KORE, in order

The engine currently:
- runs Q2_K/Q3_K/Q4_K/Q5_K/Q6_K resident weights with a **512-wide Hadamard fold**
  (`had_pre`) that is **19% of the m=1 decode budget**;
- stores KV as **Q8_0** in the main `qwen35` path (`kv_cache_append_q8`);
- has a **rotated `K8V4`** mode (INT8 K + Hadamard-rotated INT4 V) in
  `kore-attention/src/kv_cache.rs`, **wired into the generic `llama` lane but not
  the main engine, and it rotates V only**;
- streams MoE experts over PCIe at ~198 MB/token, already at Q2_K (2.6 bits).

Ranked by expected tok/s, with the regime stated (this matters — the binding
constraint is different in each):

### 7.1 KV: rotate **K and V**, full width, at append; INT4 both
**Applies now. Biggest long-context lever. ~2× KV bytes vs Q8, free at attention.**

- Rotate `K` on append, rotate `q` once per query, rotate `V` on append,
  un-rotate the attention output once. Costs `O(d log d)` per token.
- 4-bit both, per-token scale (and per-32 sub-block for finer granularity).
- Keep the **deferred** trick: hold K in f16 during prefill, quantize on decode
  insert. It avoids error compounding *and* removes dequant cost from prefill
  (this is why RotorQuant's decode can beat f16).
- **Correctness trap, recorded:** RotorQuant's V path read PPL 15 369 → 7.05
  after adding the *inverse* rotation on V dequant. Any V rotation must be
  inverted, or the attention sum silently mixes bases.

Expected: for a dense model at 8k ctx, KV is a large fraction of per-step bytes;
halving it is up to ~1.5× tok/s, and doubles max context at fixed VRAM. For
POCKET-35B (expert-bound) it is capacity-only — accept that.

### 7.2 Weights: rotation-assisted re-quantization Q4_K → Q3_K
**Offline first, then a free runtime win. ~25% fewer weight bytes on dense models.**

- Dequantize → rotate (128-wide or full-row Hadamard, folded into `W`) →
  requantize with error feedback → emit a standard K-quant block layout.
- Runtime: rotate the activation once before the GEMV. At `O(d log d)` against an
  `O(d²)` GEMV this is ~0.3%, so the win is nearly the full byte reduction.
- Gate on **perplexity**, not token ids (llama.cpp's CPU repack legitimately
  reassociates; the byte-identity contract is KORE↔KORE only).
- Only worth it where weight bytes dominate: dense models resident in VRAM.
  **POCKET-35B is already at 2.6 bits — there is nothing left to take.**

### 7.3 Fix `had_pre`'s 19%, but do not narrow it
**Engineering, not algebra.** 19% of a 9 ms step is ~1.7 ms, which is ~100× the
arithmetic cost of a 512-wide butterfly on a 2048-vector. At 26 µs/dispatch
this is **dispatch + shared-memory staging**, not math. The fix is fusion
(`had` flag inside the consuming GEMV, as already designed for the no-Hadamard
case) or fewer, larger blocks — **not** a narrower transform, which would change
the model's function.

### 7.4 Do not do
- Do not adopt HQMQ (§5.3).
- Do not put a wide transform inside the attention kernel (7× the score cost).
- Do not expect a Clifford win on MoE bytes: POCKET is at the floor, and
  FORGE/ButterflyMoE require training.
- Do not re-open GA kernels; arm A's 113–119× penalty stands.

---

## 8. What is dead, and why (recorded so it is not re-litigated)

| claim | verdict | evidence |
|---|---|---|
| A Clifford rotor is a faster way to compute a Hadamard | **dead** | arm A: Cartan–Dieudonné product is 113–119× slower than the butterfly at equal output (6.0e-5 vs butterfly, 2.1e-7 vs exact) |
| `rotor8` is a drop-in for the model's fold | **dead** | arm A: runs at butterfly speed but is a *different* operator — 1.43 relative L2 from the model's fold |
| A spherical/rotor quantizer beats affine at matched bits | **dead** | arm B: 25.48% vs 20.20% (4.1 b/w, heavy tails); 12.12% vs 7.91% (5.0 b/w, Gaussian) |
| Hurwitz multiplicative codebook beats rotation + scalar | **dead above 3.4 b/w** | arm C §5.2: loses 11–18% at 4.1–4.9 b/w on both families |
| Quaternion attention speeds up an existing model | **dead** | needs quaternion-trained Q/K; the 75%-fewer-multiplies result is for models trained that way |
| GA sequence models / Clifford SSMs help existing checkpoints | **dead by definition** | they are architectures |
| Structured (Kronecker/Monarch) post-hoc weight compression | **dead post-hoc** | needs training or recovery fine-tuning to recover quality |
| Clifford helps MoE routing | **no such result exists** | routing is a 256-way top-k; not a bottleneck |

---

## 9. Open questions worth one experiment each

1. **Does append-time full-width K rotation beat the in-kernel 4-wide scheme on
   *our* models, at 4-bit?** This is the direct KORE↔RotorQuant comparison and it
   decides §7.1's width. Needs real KV: dump `Kcur`/`Vcur` for a 2k prompt and run
   arm C on it. Cheap; do it before writing the kernel.
2. **Rotation-assisted Q4_K → Q3_K: what does it cost in perplexity on Ornith-9B?**
   If ≤2%, a 25% byte cut is free tok/s.
3. **Does per-token absmax alone (no rotation) capture most of the KV win on
   Qwen-family models?** If yes, ship the cheap version first.
4. **Deferred quantization:** measure the prefill cost of quantizing on insert
   vs on append for KORE's chunked prefill (`forward_batch`). RotorQuant claims it
   *reduces* prefill work; our chunked path may already get this for free.
5. **Non-power-of-two blocks.** Qwen hidden sizes are not powers of two; the
   Paley construction (arXiv:2504.13989) gives Hadamard-like matrices for those
   widths. Relevant to `had_pre` if the fold ever needs a non-2^k block.

---

## 10. Sources

Measured in this repo: `crates/kore-kernels/examples/clifford_probe.rs`;
`kore-tools/clifford-quant-probe.py` (arms B and C).

Papers, by family:
- KV rotation: 2604.19157 (SAW-INT4), 2606.03458 (KVarN), 2510.05373 (KVLinC),
  2510.06175 (VecInfer), 2606.20474 (UltraQuant), 2604.24971 (PolyKV),
  2606.23406 (HyperQuant), 2605.05699 (When Quantization Is Free),
  2504.13989 (Gradual Binary Search / Paley).
- Quaternion/Clifford KV: 2605.27646 (HQMQ), 2603.28430 (IsoQuant),
  2605.24920 (Quaternion Self-Attention), 2602.10195 (Versor),
  2511.11665 (CARE).
- Weight rotation: 2405.16406 (SpinQuant), FlatQuant, ThinQuant (2609.36120),
  SpectrumQuant (2610.20303), GyRot (2607.27694).
- Structured/trained: 2601.13563 (ButterflyMoE), 2610.16890 (FORGE),
  2601.05379 (EdgeLDR), Monarch, KronQ.
- Code: `scrya-com/rotorquant`, `ParaMind2025/isoquant`, `johndpope/llama-cpp-turboquant`
  (`feature/planarquant-kv-cache`), `huawei-csl/KVarN`, `kswain98/hqmq`.
- Model card evidence: `majentik/gemma-4-26B-A4B-it-RotorQuant-GGUF-Q4_K_M`
  (documents llama.cpp's default Hadamard KV rotation and upstream `q8_0`/`q4_0`
  perplexity deltas).
