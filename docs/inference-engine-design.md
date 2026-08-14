# KORE Inference Engine — Systems Design

> Applied systems thinking (Meadows, *Thinking in Systems*) to the design of a
> pure-Rust LLM/MoE/hybrid inference engine. Reimplements llama.cpp's *proven
> structural decomposition* as native Rust — **without importing llama.cpp or
> its container formats.** The whole model zoo is in scope.

---

## §1 System Map (Step 1)

### Elements
- **Weights** (the stock): model parameters in f32/f16/Q8/Q4/**ternary/binary**.
- **Backend** (CPU `kore-kernels`, GPU `kore-vulkan`): the compute substrate.
- **Architecture registry** (`kore-models`): one module per architecture.
- **KV cache**: state carried across tokens; the memory bottleneck.
- **`kore-convert`**: importer from the universal HF **safetensors** format.
- **`.kwe`** (KORE Weights Exchange): KORE-native packed weight container.
- **`kore-serve` / `kore-cli`**: the human- and program-facing surface.
- **Operator** (the user, on a 9060 XT / 16 GB desktop): the actor.

### Interconnections
- *Information flows*: safetensors → `kore-convert` → `.kwe` → loader → `Model`.
  Forward pass reads weights+KV, writes logits → sampler → tokens.
- *Control flows*: `kore-serve` batches requests → `Backend` allocates buffers.
- *Feedback signals*: output quality vs. HF reference → drives quant tolerance;
  VRAM pressure → drives KV dtype selection (Q4 vs Q8).

### Operational purpose (deduced from structure, not the tagline)
The system optimizes for **"any model, any quant, one Rust engine, on the
operator's own GPU — with no vendor lock-in."** Not "run Bonsai." Not "beat
llama.cpp's speed." Those are downstream effects of the purpose, not the purpose.

### Boundary
- **Controllable**: weight format, quant kernels, arch registry, backend.
- **Observable-only**: upstream model releases, HF format changes, the
  operator's GPU driver. We read these via `kore-convert`, never by forking
  upstream.

### Hierarchy (bottom-up)
`kore-core` (tensor/device/autograd) → `kore-btes` (quant primitives) →
`kore-vulkan`/`kore-kernels` (backend) → `kore-nn` (layers) →
`kore-models` (architectures) → `kore-serve` (surface).
**Suboptimization risk at each level** and its prevention:
- core optimizes for generality → prevented by backend trait (no monolithic op).
- btes optimizes for compression ratio → prevented by accuracy test vs f32.
- backend optimizes for raw throughput → prevented by `Backend` trait parity tests.
- models optimize for one arch → prevented by **registry as a balancing loop** (§2 B1).

---

## §2 Stocks, Flows, Feedback Loops (Step 2)

### Stocks
| Stock | Inflows | Outflows | Buffer role |
|---|---|---|---|
| Weights (VRAM) | load from `.kwe` | evict on OOM | fixed by model size |
| KV cache | append per token | cleared per sequence | **tunable** — the key buffer |
| Quantized accuracy | quantize | drift via calibration error | bounded by tolerance |
| Supported architectures | add module | (rarely remove) | grows with zoo |
| Throughput (tok/s) | backend compute | request load | smoothing buffer |

### Balancing loops
- **B1 (Registry homeostasis):** more architectures added → more test surface →
  more registry discipline required → slows reckless arch additions. Prevents
  monoculture-of-one-model. *Monitor:* number of arch modules vs. test coverage.
  *Delay:* until CI runs. *Oscillation risk:* under-testing a new arch.
- **B2 (Memory balancing):** VRAM fills → KV dtype drops Q8→Q4 → frees memory →
  steady state. *Delay:* one alloc cycle. *Risk if too short:* thrash between
  dtypes; *if too long:* OOM kill.
- **B3 (Accuracy balancing):** quant error rises → output diverges from HF ref →
  CI fails → quant re-tuned. *Delay:* test latency.

### Reinforcing loops
- **R1 (Virtuous — zoo flywheel):** more arch support → more models runnable →
  more users → more contributions → more arch support.
- **R2 (Vicious — vendor lock-in drift):** adopt one vendor's container → depend
  on their format quirks → can't load others → deeper lock-in. **This is the trap
  we just escaped** (PrismML GGUF). Structural defense: **`kore-convert` reads
  only the vendor-neutral safetensors format**; `.kwe` is KORE-owned. We never
  parse a vendor's proprietary container.

### Shifting dominance
- *Early:* R2 (lock-in) is the dominant threat → defense is the safetensors-only
  rule (Level 5 rule, §4).
- *Middle:* B2 (memory) dominates as models grow.
- *Late:* R1 (flywheel) dominates as the zoo matures.

### Critical delays
| Delay | Duration | Risk if too short | Risk if too long |
|---|---|---|---|
| convert→load | seconds | none | stale weights in RAM |
| quant tolerance check | CI run | false-pass | slow iteration |
| VRAM rebalance | 1 alloc | thrash | OOM |

---

## §3 Goal Hierarchy (Step 3)

- **P0 (Non-negotiable):**
  - **Pure Rust, zero llama.cpp/GGUF dependency.**
  - Runs on operator's 9060 XT (Vulkan) and CPU.
  - Correct: output within tolerance of HF reference for a fixed prompt.
- **P1 (Primary value):**
  - Whole model zoo (dense, GQA, MoE, SSM/linear-attn hybrid, drafters).
  - Native **ternary/binary** inference (the moat; `kore-btes` already exists).
  - Quantize-any-HF-model-on-load (no vendor file needed).
- **P2 (Secondary):** throughput, long context (quantized KV), streaming API.
- **P3 (Nice-to-have):** speculative decoding, multi-GPU, serving at scale.

**Conflict matrix:**
- Purity (P0) > speed (P2): if a fast path requires a C dep, refuse it.
- Correctness (P0) > compression (P1): if ternary diverges past tolerance, fall
  back to Q8 automatically.
- Zoo breadth (P1) > one-model perfection (P2): registry over Bonsai-only.

---

## §4 Leverage Points for Evolution (Step 4)

Mapped to Meadows' 12-point ladder. **We spent 4 hours at Level 12; the fix is
Levels 3–8.**

| Axis | Meadows level | Leverage | Why |
|---|---|---|---|
| Own the weight format (`.kwe`/safetensors) | **3 (Goals)** + **6 (Info flows)** | High | Decides what is loadable at all |
| Architecture registry as balancing loop | **8 (Balancing loops)** | High | Prevents monoculture/lock-in |
| Backend trait (CPU/Vulkan parity) | **4 (Self-org)** | Med-High | New backends = new module |
| Quant tolerance as a rule | **5 (Rules)** | Med | Bounds accuracy drift |
| KV dtype auto-selection | **9 (Delays)** | Med | Memory homeostasis |
| Kernel micro-opts, block sizes | **12 (Parameters)** | LOW | What we were grinding |

### Per-point wrong-direction analysis (high-leverage points)
- **Weight format (Level 3/6):** *Temptation:* "just parse PrismML's GGUF to get
  Bonsai working fast." *Wrong direction:* re-integrates the exact problem we fled
  (proprietary, unparseable container) and locks us to one vendor. *Right
  direction:* safetensors (vendor-neutral) + KORE-owned `.kwe`. Lose the one model
  today, gain the whole zoo forever.
- **Registry (Level 8):** *Temptation:* special-case Bonsai with a giant switch.
  *Wrong:* monoculture, fork-wide edits on every model. *Right:* one module per
  arch, registry-driven. Adding a model is additive, not invasive.
- **Backend trait (Level 4):** *Temptation:* hardcode Vulkan-only. *Wrong:* loses
  CPU fallback and portability. *Right:* trait with CPU+Vulkan; parity tests keep
  them honest.
- **Quant rule (Level 5):** *Temptation:* push ternary everywhere for max
  compression. *Wrong:* silent accuracy collapse. *Right:* ternary by default,
  auto-fallback to Q8 when divergence exceeds tolerance (B3 loop enforces).

**Direction-check:** does this deepen "any model, any quant, one Rust engine,
operator's GPU, no lock-in" while preserving purity? Yes → push. No → stop.

---

## §5 Resilience, Limits, Self-Organization (Step 5)

- **Resilience ≠ stability:** KORE should *recover* from a bad quant (fall back to
  Q8) and from OOM (drop KV dtype), not resist change until it breaks.
- **Limits (self-imposed architectural constraints):**
  - No C/C++ dependency in the runtime path (purity boundary).
  - Weights enter only via safetensors or `.kwe` (container boundary).
  - VRAM ceiling = operator's device; KV cache yields first.
- **Self-organization boundary:** new *architectures* and *backends* can emerge as
  modules (the registry and trait invite this). The *core tensor abstraction*
  cannot be restructured by a module — it is the fixed jazz stage; modules improvise
  on it.

---

## §6 Design Principles from Systems Wisdom (Step 6)

Applied as structural requirements, not aspirations:

1. **Get the beat (§1):** first release *observes* (loads a tiny model, prints
   token-for-token match vs HF) before optimizing speed.
2. **Honor information (§2):** weight metadata (shape, quant type) travels with the
   tensor; no out-of-band assumptions.
3. **Value the unquantifiable:** correctness and purity are absolute standards;
   tok/s is a vanity metric until correctness is proven.
4. **Feedback policies (B1–B3):** KV dtype and quant tolerance are *adaptive*
   (driven by VRAM/accuracy loops), not static config flags.
5. **Good of the whole:** the registry optimizes for zoo breadth, not one model.
6. **Expand time horizons:** design for the 1B-parameter Xura student + the 27B
   teacher in the *same* engine; don't optimize for today's file.
7. **Don't erode goodness:** purity (P0) is an absolute floor — no "temporary" C
   dependency "just to make Bonsai run."

---

## §7 llama.cpp's Base Principles, Reintegrated (the brief)

We keep llama.cpp's *structural insights* (the parts that are true regardless of
implementation), reimplemented in Rust:

| llama.cpp principle | KORE reimplementation |
|---|---|
| Decomposed tensor core | `kore-core` Tensor/Device/autograd |
| Quant layer abstraction | `kore-btes` (ternary/binary/Q4/Q8) |
| KV cache, quantizable | `kore-models` KvCache, dtype-configurable |
| `LLM_ARCH_*` registry | `kore-models` `Model` trait + module per arch |
| Backend abstraction (CPU/Metal/CUDA) | `Backend` trait (CPU/Vulkan) |
| GGUF container | **Replaced** by safetensors + `.kwe` (vendor-neutral) |
| C/C++ kernels | **Replaced** by Rust + `kore-vulkan` (pure Rust) |

**Innovation (the moat):** native sub-2-bit as a first-class dtype;
quantize-any-HF-model-on-load; and one engine for *both* Xura training and
inference (same tensor/autograd/backend).

---

## §8 Phased Build (leverage-ordered)

- **P0 (Level 3/6):** `.kwe` spec + `kore-convert` (safetensors → `.kwe`, ternary option).
- **P1 (Level 8):** `kore-models` registry + Qwen3/Llama reference arch.
- **P2 (Level 4):** CPU matmul f16/f32 + ternary/binary dequant (`kore-btes`).
- **P3 (Level 4):** Vulkan matmul (`kore-vulkan`) on 9060 XT.
- **P4 (Level 9):** Quantized KV cache (Q4/Q8) for long context.
- **P5 (Level 4):** Zoo expansion (MoE, SSM/linear-attn, drafter).
- **P6 (Level 6):** `kore-serve` API + streaming.

## §9 Verification (no "green tests = done")
1. Tiny Qwen3-0.6B safetensors → `.kwe` → KORE **CPU** → token-for-token match vs
   HF `transformers` on a fixed prompt.
2. Ternary-quantize → KORE CPU → within tolerance.
3. Same on **Vulkan / 9060 XT** → identical output, faster.
4. 27B-class model → KORE quantizes to ternary → fits 16 GB → runs.
