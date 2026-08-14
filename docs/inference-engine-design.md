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
**Who is this for? The operator** — someone with their own hardware (laptop,
16 GB GPU, phone, Pi) and a model file they did *not* write, who wants to run
it **locally, privately, on their own metal, without asking NVIDIA, a cloud, or
the model's vendor for permission.** That is exactly who llama.cpp is for, and
it is KORE's reason to exist. llama.cpp's goal was never "run Llama well" — it
is **"run whatever they ship next, on whatever you have, with no code change."**
KORE inherits that purpose and restates it without the C++ and without the
proprietary-container dependency.

Therefore KORE optimizes for: **an operator points KORE at any HuggingFace
model and runs it on their own GPU/CPU, locally, in pure Rust, with
quantizations that fit their hardware.** Not "run Bonsai." Not "run Qwen3."
Not "beat llama.cpp's speed." Those are downstream effects, not the purpose.
The model is a *stranger's* output; KORE is the universal local runtime between
the operator and the zoo.

### Boundary
- **Controllable**: weight format, quant kernels, arch registry, backend.
- **Observable-only**: upstream model releases, HF format changes, the
  operator's GPU driver. We read these via `kore-convert` (vendor-neutral
  safetensors), never by forking upstream or parsing a vendor's proprietary
  container. The next architecture the operator tries is *unknown by design* —
  the registry exists so it is a module, not a rewrite.

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

- **P0 (Non-negotiable) — operator sovereignty:**
  - **The operator runs any HF model on their own hardware, locally, in pure
    Rust, with no vendor container, no cloud, no C++.**
  - Pure Rust, zero llama.cpp/GGUF dependency (this is *how* sovereignty is
    enforced, not a separate goal).
  - Runs on the operator's Vulkan GPU and CPU.
  - Correct: output within tolerance of the source framework for a fixed prompt.
- **P1 (Primary value — means to P0):**
  - **Universality:** the next architecture the operator tries is a module, not a
    rewrite (registry). Whole zoo: dense, GQA, MoE, SSM/linear-attn hybrid,
    drafters.
  - **Fit-the-hardware quantization:** native ternary/binary (sub-2-bit) so
    bigger models run on operator hardware — a *buffer* (Level 11) serving P0,
    not the goal itself.
  - Quantize-any-HF-model-on-load (safetensors, vendor-neutral).
- **P2 (Secondary):** throughput, long context (quantized KV), streaming API.
- **P3 (Nice-to-have):** speculative decoding, multi-GPU, serving at scale.

**Explicitly NOT a goal (anti-tunnel-vision):**
- "Support Bonsai" / "support Qwen3" / "support model X" is **not** a P0/P1
  goal. It is a Level-12 parameter — which test rig validates the harness. The
  engine must not encode any specific model as special. Picking a reference
  architecture is a *validation vehicle*, chosen for layer-coverage, discarded
  after it proves the pipeline.

**Conflict matrix:**
- sovereignty (P0) > speed (P2): if a fast path requires a C dep or a vendor
  container, refuse it.
- correctness (P0) > compression (P1): if ternary diverges past tolerance, fall
  back to Q8 automatically (B3 loop enforces).
- universality (P1) > one-model perfection (P2): registry over any single model.

---

## §4 Leverage Points for Evolution (Step 4)

Mapped to Meadows' 12-point ladder. **We spent 4 hours at Level 12; the fix is
Levels 2–8.** The deepest correction was **Level 2 (paradigm):** we kept
asking "which model?" when the real question is **"who is this for?"** — the
*operator*, running a stranger's model on their own hardware. That single
reframe dissolved the tunnel vision (Bonsai-only, Qwen3-default, GGUF-forensics).

| Axis | Meadows level | Leverage | Why |
|---|---|---|---|
| **Paradigm: "who is it for?"** (operator sovereignty over unknown zoo) | **2 (Paradigm)** | Highest | Dissolves model-specific tunnel vision |
| Goal: operator runs any HF model locally, pure Rust | **3 (Goals)** | High | Decides what the system is *for* |
| Own the weight format (`.kwe`/safetensors) | **6 (Info flows)** | High | Decides what is loadable at all |
| Architecture registry as balancing loop | **8 (Balancing loops)** | High | Next model = module, not rewrite |
| Backend trait (CPU/Vulkan parity) | **4 (Self-org)** | Med-High | New backends = new module |
| Quant tolerance as a rule | **5 (Rules)** | Med | Bounds accuracy drift |
| KV dtype auto-selection | **9 (Delays)** | Med | Memory homeostasis |
| **Which reference architecture to validate with** | **12 (Parameters)** | LOW | Test rig only; discard after pipeline works |
| Kernel micro-opts, block sizes | **12 (Parameters)** | LOW | What we were grinding |

### Per-point wrong-direction analysis (high-leverage points)
- **Paradigm / who is it for (Level 2/3):** *Temptation:* "let's get Bonsai or
  Qwen3 running, that's the point." *Wrong direction:* re-centers a specific
  model as the purpose; the next model breaks the design. *Right direction:*
  build for the *operator + the unknown next model*. Bonsai/Xura are just two
  possible inputs, not the reason KORE exists.
- **Weight format (Level 6):** *Temptation:* "just parse PrismML's GGUF to get
  Bonsai working fast." *Wrong direction:* re-integrates the exact problem we fled
  (proprietary, unparseable container) and locks us to one vendor. *Right
  direction:* safetensors (vendor-neutral) + KORE-owned `.kwe`. Lose the one
  model today, gain the whole zoo forever.
- **Registry (Level 8):** *Temptation:* special-case one model with a giant
  switch. *Wrong:* monoculture, fork-wide edits on every new model. *Right:* one
  module per arch, registry-driven. Adding a model is additive, not invasive.
- **Backend trait (Level 4):** *Temptation:* hardcode Vulkan-only. *Wrong:* loses
  CPU fallback and portability. *Right:* trait with CPU+Vulkan; parity tests keep
  them honest.
- **Quant rule (Level 5):** *Temptation:* push ternary everywhere for max
  compression. *Wrong:* silent accuracy collapse. *Right:* ternary by default,
  auto-fallback to Q8 when divergence exceeds tolerance (B3 loop enforces).
- **Reference architecture (Level 12):** *Temptation:* agonize over which model
  to support first. *Wrong:* treats a validation rig as a design commitment.
  *Right:* pick the *simplest* model that exercises every layer (embedding →
  mixed-quant linear → norm → attention → sampler); use it to prove the
  pipeline; never optimize it.

**Direction-check:** does this deepen "an operator runs any model on their own
hardware, locally, in pure Rust, no vendor lock-in" while preserving sovereignty?
Yes → push. No → stop.

---

## §4b System Traps

- **Shifting the Burden (B→R):** *How it manifests:* lean on a vendor's container
  (PrismML GGUF) as a crutch to "get a model running fast," atrophying KORE's own
  weight pipeline. *Structural cause:* missing B-loop that would force
  vendor-neutrality. *Defense:* the safetensors-only rule (Level 5/6) — KORE
  ingests only the universal format; proprietary containers are refused by
  construction.
- **Success to the Successful / monoculture (R):** *How:* over-invest in one
  model (Bonsai) → it becomes the only tested path → others rot. *Defense:* the
  registry balancing loop (B1) + the "no model is special" rule (§3).
- **Drift to Low Performance (R+B):** *How:* accept silent ternary accuracy loss
  to hit a size target. *Defense:* B3 accuracy loop + auto-fallback to Q8.
- **NOT vulnerable:** tragedy of commons, escalation, rule-beating — this is a
  single-operator local engine, not a multi-actor contested system.

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

llama.cpp is **for the operator**: run a stranger's model on your own hardware,
locally, no permission asked. We keep its *structural insights* (true regardless
of implementation), reimplemented in Rust:

| llama.cpp principle | KORE reimplementation |
|---|---|
| Built for the operator, not the model | P0 goal: operator sovereignty over unknown zoo (§1/§3) |
| Decomposed tensor core | `kore-core` Tensor/Device/autograd |
| Quant layer abstraction (fit any hardware) | `kore-btes` (ternary/binary/Q4/Q8) — a *buffer* serving P0 |
| KV cache, quantizable | `kore-models` KvCache, dtype-configurable |
| `LLM_ARCH_*` registry | `kore-models` `Model` trait + module per arch |
| Backend abstraction (CPU/Metal/CUDA) | `Backend` trait (CPU/Vulkan) |
| GGUF container | **Replaced** by safetensors + `.kwe` (vendor-neutral) |
| C/C++ kernels | **Replaced** by Rust + `kore-vulkan` (pure Rust) |

**Sub-2-bit / one-engine-for-train-and-infer are means, not the moat.** They
serve P0 (run bigger models on operator hardware; one sovereign runtime for the
operator's teacher *and* student). The moat is the *purpose*: an operator can
point KORE at any model and own the run. Native ternary is just the sharpest
expression of "fit the operator's hardware."

---

## §8 Phased Build (leverage-ordered)

- **P0 (Level 6):** `.kwe` spec + `kore-convert` (safetensors → `.kwe`, ternary option).
- **P1 (Level 8):** `kore-models` registry (`Model` trait) + **one reference
  architecture chosen only for layer-coverage** (embedding → mixed-quant linear →
  norm → attention → sampler). Treated as a *validation rig*, not a committed
  target. Bonsai/Xura slot in later as ordinary modules.
- **P2 (Level 4):** CPU matmul f16/f32 + ternary/binary dequant (`kore-btes`).
- **P3 (Level 4):** Vulkan matmul (`kore-vulkan`) on operator GPU.
- **P4 (Level 9):** Quantized KV cache (Q4/Q8) for long context on constrained VRAM.
- **P5 (Level 4):** Zoo expansion — new arch = new module (MoE, SSM/linear-attn,
  drafter). Each is additive; registry stays the balancing loop.
- **P6 (Level 6):** `kore-serve` API + streaming.

**Validation rule (anti-tunnel-vision):** every phase is verified against the
*operator's* success criterion — "point KORE at a model file, get correct output
on my hardware" — not "Bonsai runs." The reference arch is swapped freely.

## §9 Verification (no "green tests = done")
1. A small **HF model (chosen for layer-coverage, not fame)** → `.kwe` → KORE
   **CPU** → token-for-token match vs the source framework on a fixed prompt.
2. Ternary-quantize → KORE CPU → within tolerance.
3. Same on **Vulkan / operator GPU** → identical output, faster.
4. A 27B-class model → KORE quantizes to ternary → fits 16 GB → runs.
5. **Sovereignty check:** swap the validation model for a *different*
   architecture → it loads as a new module with no core change.
