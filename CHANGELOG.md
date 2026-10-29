# Changelog

All notable changes to KORE will be documented in this file.

## [0.2.1] — 2026-10-29

### Inference on the 16 GB AMD box

Device-path work for Qwen3.6-35B-A3B and Flint (40B-A2.5B) on gfx1200 (RX 9060 XT).
Every number, and the reasoning behind each rejected experiment, is in
`docs/inference-handoff.md`; `docs/roadmap.md` lists what remains.

#### Added
- `kore-vulkan`: sink + sliding-window GQA decode attention
  (`flash_decode_gqa_sink.wgsl`) and flat per-head QK-RMSNorm + split-half RoPE
  with a NoPE path (`flint_qknorm_rope.wgsl`), both gated against CPU oracles
  (max |Δ| 2e-5 and 1e-5).
- `kore-vulkan`: q2_K / q3_K GEMV entry points on the device backend.
- `kore-nn`: host-path port of Flint (`flint.rs`), including route profiling
  behind `KORE_ROUTE_PROFILE` / `KORE_ROUTE_TEST` / `KORE_ROUTE_SPLIT` /
  `KORE_PREDICT`.
- `kore-nn`: `DeviceBank` expert residency and the opt-in device routing lane
  (`KORE_MOE_DEVICE_ROUTE=1`); host routing stays the default.
- `kore-cli`: `upload_bw` H2D probe (group / pipeline / burst) and the `flint`
  example.

#### Changed
- The dispatch batch is tunable at run time (`KORE_BATCH_OPS`), default 48.
- `gdn_step` sweeps each recurrent state in two 64-row halves, one per thread
  group.

#### Fixed
- `quantize_q8_rows` no longer leaves `.5` ties to the driver: the shader pins
  half-away-from-zero and the test asserts a contract the device can actually
  hold (scale within 1 ULP; codes byte-equal except within one code of a tie,
  with a bound on how many codes may take that allowance). Fixes the one
  long-standing red test.
- Workspace is clean under `cargo clippy --workspace -- -D warnings`, including
  the `too_many_arguments` lints on the `DeviceBackend` trait and its Vulkan
  implementation.
- `kore-data`, `kore-clifford` and `kore-serve` tests no longer trip
  deny-by-default clippy lints (`erasing_op`, `approx_constant`), so
  `clippy --all-targets` reports no errors.

## [0.2.0] — 2026-02-12

### Architecture Refactor — Pure ML Engine

KORE is now a **model-agnostic ML engine**. Model implementations (Transformer, Mamba, VL-JEPA)
have been extracted to the [Xura](https://github.com/KidIkaros/Xura) project.

#### Removed Crates
- `kore-transformer` — extracted to `xura-transformer`
- `kore-mamba` — extracted to `xura-mamba`
- `kore-vljepa` — extracted to `xura-vljepa`
- `kore-glial`, `kore-glial-vljepa` — removed (empty)

#### Changed
- **kore-serve**: now model-agnostic via `InferenceModel` trait — any model implementing
  `generate_with_config()` can be served through the OpenAI-compatible API
- **kore-cli**: removed `generate` command (model-specific); `serve` starts in placeholder mode;
  `export` parses config.json directly without model-specific loaders
- **kore-nn**: added `sampler` module (SamplerConfig, Rng, top-k/top-p/repetition penalty sampling)
  — previously in kore-transformer, now a framework-level utility

## [0.1.0] — 2026-02-10

First hardened release of the KORE ML Training Framework.

### Python Bindings (maturin)
- Autograd: `backward()`, `requires_grad_()`, `.grad`, `zero_grad()`
- Tensors: `randn()`, `rand_uniform()`
- Layers: `Conv2d`, `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`
- I/O: `save_state_dict`, `load_state_dict` (safetensors)
- Activations: `tanh`, `silu`

### Rust Framework
- `RMSNorm` module in kore-nn
- `clip_grad_norm_` and `clip_grad_value_` in kore-optim
- `Tensor::randn` and `Tensor::rand_uniform` in kore-core

### Hardening
- 7 rounds of Greptile AI review, **0 comments on final round**
- All bare `unwrap()` replaced with `?` or descriptive `expect()`
- `usize` underflow guards, NaN-safe sorting, softmax div-by-zero guard
- Precondition asserts for kernel_size, group_size, ngroups, prompt length
- `Result`-based error propagation throughout core/nn/optim crates

### Stats
- **439 tests passing**, zero failures
- 13 crates, 59 files changed, +12,136 lines
