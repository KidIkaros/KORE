# KORE — Inference Subsystem Design (within the KORE ML Framework)

> Applied systems thinking (Meadows, *Thinking in Systems*) to the design of
> KORE's **inference subsystem** — one part of a larger whole. **KORE is a
> complete pure-Rust ML framework** (tensor core, autograd, optimizers, training,
> serving). The inference engine is being *added now*; it is a subsystem, not the
> product. We are **not** building a PyTorch/TensorFlow competitor — we are
> building the toolchain that fits *this operator's research*: figuring out what
> architectures work best, beyond the token dead-end.

---

## Step 0 — Paradigm & Goal Gate (mandatory, first)

> Per the systems-design skill: state the paradigm (Level 2) and the P0 goal
> (Level 3) *before* any component, format, or kernel is named. A single concrete
> model/format name may appear ONLY as a throwaway validation rig, never as a goal.

**Level 2 — Paradigm (who is this for / what is it actually FOR?):**
KORE is *the operator's sovereign ML research instrument* — a complete pure-Rust
framework for discovering what architectures work (SSM, world models,
tokenless/V-JEPA), using LLMs as a *teacher input*, not as the destination. The
inference engine is a **subsystem** added so candidate architectures and an LLM
teacher run in the *same* runtime, making "does my architecture capture
intelligence?" a measurable question. Tokens are a dead end; we are not building a
PyTorch/TensorFlow competitor and not an LLM-serving product.

**Level 3 — P0 purpose (one line, names no model/format/vendor):**
> *One pure-Rust runtime where the operator trains and evaluates novel
> (non-token) architectures beside LLM teachers, owning the whole stack — so
> architecture research is fast, comparable, and vendor-free.*

Everything else in this doc (speed, a specific architecture, safetensors, `.kwe`,
Vulkan, ternary) is a **means** to that P0 goal or a **Level-12 parameter**. The
gate holds: P0 is stated without naming any model, format, or vendor.

---

## §0 Frame: what KORE is, and why the inference engine exists

- **KORE = a complete ML framework**, not an inference engine. Training,
  autograd, optimizers, and inference all live in one pure-Rust runtime.
- **The operator's thesis:** tokens are a dead end. The budding future is
  **alternative architectures — SSM, world models, tokenless / V-JEPA-style.**
  LLMs are a *resource to learn from* (teacher/distillation source), not the
  destination.
- **KORE exists to build new architectures** (e.g. **Xura**: V-JEPA / Mamba3
  lineage). The inference engine's real job is to be the **evaluation harness**:
  run a candidate architecture and an LLM teacher through the *same* KORE runtime
  so you can measure whether the new architecture actually captures intelligence.
  "One engine for train and infer" matters because the comparison must be
  apples-to-apples in one framework — not as a feature, but as the research method.
- **Therefore the inference subsystem is scoped as:** load/run models (including
  LLM teachers) on the operator's hardware, in pure Rust, vendor-neutral — *so
  that* the operator can train and evaluate novel architectures alongside them.
  llama.cpp's lessons inform the *inference* piece; they do not define KORE.

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
**KORE's purpose is architectural research**, not model-serving. The operator is
building novel (non-token, SSM / world-model / V-JEPA-style) architectures such as
**Xura**, and using LLMs as a *teacher* to bootstrap them. The inference
subsystem exists so a candidate architecture and an LLM teacher run in the *same*
pure-Rust runtime — making "does my architecture capture intelligence?" a
measurable question.

Within that, the inference piece inherits llama.cpp's operator-sovereignty goal
(run any model on your own hardware, locally, vendor-neutral, pure Rust) — but
that is a *means* to the research end, not KORE's reason to exist. The operator is
not "the user who wants to chat with Llama"; the operator is **the researcher
training the next architecture**, for whom the LLM is one input among many.

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
- **R0 (Virtuous — research acceleration, the actual purpose):** KORE trains a
  candidate architecture *and* runs the LLM teacher in the *same* runtime → the
  operator measures "does my architecture capture intelligence?" cheaply →
  iterates faster → discovers what works → builds the next candidate in KORE.
  This is the loop the whole framework exists to tighten. *Monitor:* time from
  "new arch idea" → "apples-to-apples eval vs teacher." *Risk if too weak:*
  research stalls back into manual glue-code and the token dead-end.
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

*Delay rule (Meadows "estimate the delay, multiply by three"):* the dangerous
delays here are the **quant-tolerance check** and the **convert→load** path. We
err toward *too slow*: a CI accuracy gate that lags a day still beats a false-pass
that ships a broken quant; a weight-ingest that takes seconds is fine if it means
the format is unambiguous. The one delay we must keep *short* is VRAM rebalance
(oscillation risk), bounded by a single-alloc hysteresis.

---

## §3 Goal Hierarchy (Step 3)

- **P0 (Non-negotiable) — enable architectural research:**
  - **One pure-Rust runtime for train *and* infer**, so candidate architectures
    (Xura/SSM/world-model) and LLM teachers are compared apples-to-apples.
  - Tokens are not the ceiling: the framework must express **non-autoregressive /
    non-token / SSM / world-model** architectures natively, not as a bolt-on.
  - The operator owns the whole stack — pure Rust, no hidden C++/vendor runtime.
- **P1 (Primary value — the inference subsystem, a means to P0):**
  - **Operator sovereignty:** run any HF model (the teacher) on the operator's
    own hardware, locally, vendor-neutral, pure Rust.
  - **Universality:** the next architecture is a module, not a rewrite (registry).
  - **Fit-the-hardware quantization:** native ternary/binary so bigger teachers
    (and bigger students) run on operator hardware — a *buffer* (Level 11) serving P0.
  - Quantize-any-HF-model-on-load (safetensors, vendor-neutral).
- **P2 (Secondary):** inference throughput, long context (quantized KV), streaming.
- **P3 (Nice-to-have):** speculative decoding, multi-GPU, serving at scale.

**Explicitly NOT a goal (anti-tunnel-vision):**
- "Beat PyTorch/TensorFlow" is **not** the goal. KORE is a research tool for *this*
  operator, not a general-purpose framework competitor.
- "Support Bonsai" / "support Qwen3" / "support model X" is **not** a P0/P1 goal.
  It is a Level-12 parameter — which teacher validates the harness. The framework
  must not encode any specific model as special.
- "Serve tokens fast" is a *subordinate* concern; tokens are the dead-end the
  research is moving beyond.

**Conflict matrix:**
- research enablement (P0) > inference speed (P2): a fast path that can't train
  novel archs is worthless here.
- sovereignty (P0) > speed (P2): refuse a C dep or vendor container.
- correctness (P1) > compression (P1): ternary diverges past tolerance → Q8 fallback.
- universality (P1) > one-model perfection (P2): registry over any single model.

### Goal Verification Tests (each goal → decidable assertion)
| Goal | Verification assertion (must hold to claim the goal is met) |
|---|---|
| **P0** research enablement | A candidate *novel* architecture trains **and** an LLM teacher runs in the **same** KORE runtime; both consume the same tensor/autograd/backend primitives (no forked path). |
| **P0** non-token native | A non-autoregressive / SSM / world-model arch is expressible as a `Model` module without patching core; a token-only path is not required to compile. |
| **P0** sovereignty (pure Rust) | `cargo build` of the runtime path succeeds with **zero** C/C++ deps; no `cc`/`vcpkg`/FFI to llama.cpp in the dep tree. |
| **P1** operator sovereignty | An HF teacher model loads from safetensors (no vendor container) and produces output within tolerance on the operator's own GPU/CPU. |
| **P1** universality | Adding a new architecture = adding one module + one registry entry; zero edits to existing arch modules or core. |
| **P1** fit-the-hardware quant | A 27B-class teacher quantizes to ternary and fits 16 GB VRAM with KV budget remaining. |
| **P2** throughput / long ctx | Quantized KV (Q4/Q8) sustains a 4K context within the VRAM budget; streaming emits tokens. |
| **P3** serving | `kore-serve` exposes an API + streaming for the teacher path. |

*(Assertions are decidable, not vanity metrics. Correctness/tolerance checks gate
any throughput claim — Meadows #5.)*

---

## §4 Leverage Points for Evolution (Step 4)

Mapped to Meadows' 12-point ladder. **We spent 4 hours at Level 12; the fix is
Levels 2–8.** The deepest correction was **Level 2 (paradigm):** we kept
asking "which model?" when the real question is **"who is this for?"** — the
*operator*, running a stranger's model on their own hardware. That single
reframe dissolved the tunnel vision (Bonsai-only, Qwen3-default, GGUF-forensics).

| Axis | Meadows level | Leverage | Why |
|---|---|---|---|
| **Paradigm: "what is KORE for?"** (architectural research, not model-serving) | **2 (Paradigm)** | Highest | Dissolves both token-tunnel-vision AND model-specific tunnel vision |
| Goal: one pure-Rust train+infer runtime for novel archs + LLM teachers | **3 (Goals)** | High | Decides what the framework *is* |
| Own the weight format (`.kwe`/safetensors) | **6 (Info flows)** | High | Decides what is loadable at all |
| Architecture registry as balancing loop | **8 (Balancing loops)** | High | Next arch = module, not rewrite; student AND teacher coexist |
| Backend trait (CPU/Vulkan parity) | **4 (Self-org)** | Med-High | New backends = new module |
| Quant tolerance as a rule | **5 (Rules)** | Med | Bounds accuracy drift |
| KV dtype auto-selection | **9 (Delays)** | Med | Memory homeostasis |
| **Which teacher model to validate with** | **12 (Parameters)** | LOW | Test rig only; discard after pipeline works |
| Kernel micro-opts, block sizes | **12 (Parameters)** | LOW | What we were grinding |

### Per-point wrong-direction analysis (high-leverage points)
- **Paradigm / what is KORE for (Level 2/3):** *Temptation:* "let's get an LLM
  running, that's the point" — or "let's beat PyTorch." *Wrong direction:* both
  re-center KORE on *serving tokens* (the dead end) or on *competing with
  general frameworks* (not our game). *Right direction:* KORE is the operator's
  research instrument for **novel architectures** (SSM/world-model/V-JEPA/Xura).
  The LLM is a *teacher input*, the inference engine a *subsystem* for comparing
  candidate vs teacher in one runtime. Build for the researcher, not the chatbot.
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

**Direction-check:** does this deepen **R0 (research acceleration)** — training a
candidate architecture *and* running the LLM teacher in one pure-Rust runtime — while
preserving **P0 (sovereignty: operator owns the whole stack)**? Yes → push. No → stop.
If a proposal serves only "run a model faster" without feeding the architecture-research
loop, it is Level-12 tuning — defer it.

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

### Borrow vs Refuse (trap-mitigation research)

The skill directs us to research real systems that solve the same traps
structurally, then apply the borrow-vs-refuse test. KORE's applicable traps are
*Shifting the Burden* (vendor-container crutch) and *Drift to Low Performance*
(silent quant loss). Sources surveyed and the verdict:

- **llama.cpp (vendor-neutrality via GGUF + backend trait):** *Borrow* the
  structural insight — a single loadable container + a backend trait so CPU/Metal/
  CUDA/Vulkan are interchangeable modules. *Refuse* GGUF itself and the C++ core:
  KORE owns a vendor-neutral **safetensors** ingress + a pure-Rust `.kwe`, and
  its backend trait is Rust-only (CPU/Vulkan). The insight is kept; the container
  and the language are refused because they conflict with P0 (sovereignty/purity).
- **llama-cpp-rs (Rust binding to llama.cpp):** *Refuse.* It is an FFI wrapper —
  it still chains KORE to llama.cpp's container and kernels, i.e. it *re-introduces*
  the lock-in trap (R2) we escaped. Useful only as an API-shape reference, not as
  a dependency.
- **mue-x / Hermes-curator / Mercury "soul files" (drift defense):** *Borrow the
  shape, not the mechanism.* These systems fight *behavioral drift* in self-modifying
  agents via pinned invariants, curators, and declared-identity anchors. KORE's
  drift risk is narrower — *accuracy* drift under quantization, not goal drift.
  So we borrow the *principle* (pin an absolute floor + a check that enforces it)
  and apply it as **B3 (accuracy loop) + the "purity is an absolute floor" rule
  (§6.7)**: no "temporary" C dependency, no silent tolerance breach. We refuse the
  agent-governance machinery (curators, soul files) — that's for multi-actor
  autonomous systems, not a single-operator ML framework.
- **RotorQuant / llama.cpp KV-quant (Q4/Q8 KV):** *Borrow* the technique — quantized
  KV cache is a pure memory win on constrained VRAM (the B2 loop). Refuse nothing
  here; it serves P0 directly and is format-agnostic.

**Borrow-vs-refuse summary:** keep *structural insights* (vendor-neutral ingress,
backend trait, accuracy floor, quantized KV); refuse *concrete artifacts* that
re-create the lock-in or purity traps (GGUF, C++ core, FFI wrappers, agent
  governance). The test applied throughout: does this deepen R0 (research
  acceleration) while preserving P0 (sovereignty)? Borrow if yes, refuse if no.

## §5 Resilience, Limits, Self-Organization (Step 5)

- **Resilience ≠ stability:** KORE should *recover* from a bad quant (fall back to
  Q8) and from OOM (drop KV dtype), not resist change until it breaks.
- **Limits (self-imposed architectural constraints — table):**

  | Resource / boundary | Value | Why self-imposed (architectural, not policy) |
  |---|---|---|
  | Runtime dependencies | No C/C++ in the runtime path | Purity/sovereignty floor (P0); refuses the lock-in trap |
  | Weight ingress | safetensors or `.kwe` only | Vendor-neutral; proprietary containers refused by construction |
  | VRAM ceiling | operator's device; KV cache yields first | Hardware reality; B2 loop governs rebalance |
  | Core tensor abstraction | immutable by modules | Jazz stage; modules improvise on it, can't restructure it |

- **Self-organization boundary:** new *architectures* and *backends* can emerge as
  modules (the registry and trait invite this). The *core tensor abstraction*
  cannot be restructured by a module — it is the fixed jazz stage; modules improvise
  on it.

---

## §5b Monitor & Iterate (Step 7)

Track signals *aligned to goals*, not vanity metrics. Per Meadows #5, do not
metric the unquantifiable (research insight, architecture quality) — reflect,
don't optimize.

| Signal | Tied to | Watch how | Cadence |
|---|---|---|---|
| New-arch idea → apples-to-apples eval vs teacher (latency) | **R0** research acceleration | time from idea to running eval in one runtime | per project |
| Cargo dep-tree has C/FFI to llama.cpp? | **P0** sovereignty | CI guard fails build on violation | every build |
| Arch-module count vs test coverage | **B1** registry homeostasis | coverage must rise with modules | every PR |
| VRAM at steady state (KV dtype chosen) | **B2** memory | within budget; no OOM, no thrash | per run |
| Quant divergence vs HF reference | **B3** accuracy | within tolerance; else Q8 fallback | per quant change |
| Zoo breadth (distinct arch modules) | **R1** flywheel | growth, not monoculture | quarterly |

*Direction-check on monitoring itself:* if a metric starts driving decisions that
hurt R0/P0 (e.g. optimizing tok/s over eval fidelity), drop it. Metrics serve the
goal; the goal is not the metric.

---

## §6 Design Principles from Systems Wisdom (Step 6)

Meadows' 14 guidelines applied as *structural requirements* (not aspirations).
Each: guideline → concrete meaning for KORE → implementation note.

1. **#1 Get the beat** — first cycle *observes*, doesn't act. → The first
   inference release loads a tiny model and prints token-for-token match vs the
   source framework *before* any speed work. → CI gate: correctness check runs
   before the throughput benchmark exists.
2. **#3 Honor information** — no distortion/delay/withholding. → Weight metadata
   (shape, quant type, arch) travels *with* the tensor; no out-of-band
   assumptions about layout. → `.kwe` header carries dtype+shape; loader refuses
   tensors whose metadata is missing.
3. **#5 Value the unquantifiable** — reflect on quality, don't optimize metrics. →
   Correctness and purity are absolute standards; tok/s is a vanity metric until
   correctness is proven. → No perf PR merges without a green accuracy test.
4. **#6 Feedback policies** — adaptive parameters, not static config. → KV dtype
   and quant tolerance are *driven by loops* (B2 memory, B3 accuracy), not flags
   the operator must hand-tune. → auto-select KV dtype on VRAM pressure; auto-fallback
   Q8 on divergence.
5. **#7 Good of the whole** — optimize the system, not one actor. → The registry
   optimizes for zoo breadth, not one model's speed. → Every arch is a module; no
   arch gets a fast-path exception in core.
6. **#12 Expand time horizons** — think in years, not sessions. → Design for the
   1B-parameter Xura student *and* the 27B teacher in the *same* engine; don't
   optimize for today's file. → Tensor/autograd abstraction is stable; only
   modules evolve.
7. **Don't erode goodness** (bonus) — absolute standards, no drift. → Purity (P0)
   is an absolute floor: no "temporary" C dependency "just to make a model run."
   → The purity boundary is enforced in the build (no C in runtime path) and in
   review (PR checklist item).

*(Guidelines not enumerated here — #2, #4, #8–#11 — are either N/A to a
single-operator local framework or already covered by the goal/leverage/trap
sections above; we do not pad the doc with inapplicable principles.)*

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
serve P0: native ternary fits bigger teachers *and* bigger students on operator
hardware, and one runtime lets you compare a candidate architecture against an LLM
teacher apples-to-apples. The moat is the *purpose*: KORE is the operator's
sovereign instrument for discovering what architectures work — tokens optional,
research first. The inference subsystem is one part of that instrument.

---

## §8 Phased Build (leverage-ordered)

The inference subsystem is built *to serve research*, not as a product. Each phase
is sequenced by leverage (§4), and the whole thing exists so the operator can
train-and-evaluate novel architectures (Xura/SSM/world-model) next to LLM teachers
in one runtime. KORE already has the training side (`xura-*` crates: autograd,
optimizers, V-JEPA teacher); this plan adds the inference/evaluation half.

- **P0 (Level 6):** `.kwe` spec + `kore-convert` (safetensors → `.kwe`, ternary option).
- **P1 (Level 8):** `kore-models` registry (`Model` trait) + **one reference
  architecture chosen only for layer-coverage** (embedding → mixed-quant linear →
  norm → attention → sampler). Treated as a *validation rig* for the teacher
  path, not a committed target. Xura/SSM/world-model slot in later as ordinary
  modules — and those are the point.
- **P2 (Level 4):** CPU matmul f16/f32 + ternary/binary dequant (`kore-btes`).
- **P3 (Level 4):** Vulkan matmul (`kore-vulkan`) on operator GPU.
- **P4 (Level 9):** Quantized KV cache (Q4/Q8) for long context on constrained VRAM.
- **P5 (Level 4):** Zoo + novel-arch expansion — new arch = new module (MoE,
  SSM/linear-attn, world-model, drafter). Each is additive; registry stays the
  balancing loop. **This is where Xura lives.**
- **P6 (Level 6):** `kore-serve` API + streaming (for the teacher/serving side).

**Validation rule (anti-tunnel-vision):** every phase is verified against the
*operator's* success criterion — "point KORE at a teacher model, get correct
output on my hardware, and run a candidate architecture through the same
runtime" — not "Bonsai runs" and not "tokens per second." The reference arch is
swapped freely; the novel-arch path is the real deliverable.

## §9 Verification (no "green tests = done")
1. A small **HF model (chosen for layer-coverage, not fame)** → `.kwe` → KORE
   **CPU** → token-for-token match vs the source framework on a fixed prompt.
2. Ternary-quantize → KORE CPU → within tolerance.
3. Same on **Vulkan / operator GPU** → identical output, faster.
4. A 27B-class model → KORE quantizes to ternary → fits 16 GB → runs.
5. **Sovereignty check:** swap the validation model for a *different*
   architecture → it loads as a new module with no core change.
6. **Research-enablement check (the real P0):** a candidate *novel* architecture
   (e.g. an Xura/SSM/world-model variant) trains and runs in the **same** KORE
   runtime as the LLM teacher, so the two are comparable apples-to-apples. This is
   the criterion that distinguishes KORE from "an inference engine."
