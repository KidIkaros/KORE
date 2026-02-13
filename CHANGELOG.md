# Changelog

All notable changes to KORE will be documented in this file.

## [0.2.0] — 2025-07-11

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
- **588 tests passing**, zero failures
- 16 crates, 59 files changed, +12,136 lines
