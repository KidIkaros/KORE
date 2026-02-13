```
 _  _____  ____  _____
| |/ / _ \|  _ \| ____|
| ' / | | | |_) |  _|
| . \ |_| |  _ <| |___
|_|\_\___/|_| \_\_____|
```

# KORE

**A pure-Rust ML engine — from training to edge inference.**

[![CI](https://github.com/KidIkaros/KORE/actions/workflows/ci.yml/badge.svg)](https://github.com/KidIkaros/KORE/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org)

---

## Highlights

- **Autograd engine** — computation graph with automatic backward pass
- **Native quantization** — 1.58-bit (ternary) and 2-bit (quaternary) layers with 8–16× compression
- **Model-agnostic** — build any architecture (transformers, SSMs, vision models) on top of KORE primitives
- **Flash Attention** — paged KV-cache, multi-head & grouped-query attention
- **Edge inference** — BitNet, SqueezeNet, `.koref` export, `no_std` runtime (WASM, iOS, Android)
- **Python bindings** — `import kore` with full NumPy interop, autograd, and safetensors I/O

---

## Quickstart: Python

### Install

```bash
pip install maturin
git clone https://github.com/KidIkaros/KORE.git && cd KORE
maturin develop --release
```

### Tensors & NumPy

```python
import kore
import numpy as np

x = kore.Tensor.randn([4, 16])
print(x.shape, x.dtype)          # [4, 16] f32

arr = np.ones((4, 16), dtype=np.float32)
t = kore.Tensor(arr)             # zero-copy from NumPy
print(t.numpy())                 # back to NumPy
```

### Autograd

```python
w = kore.Tensor.randn([4, 2])
w.requires_grad_(True)

loss = (w * w).sum()
loss.backward()
print(w.grad.shape)              # [4, 2]
```

### Layers & Training (low-level)

```python
model = kore.nn.Linear(16, 1)
optimizer = kore.optim.Adam(lr=0.001)

x = kore.Tensor.randn([32, 16])
y = kore.Tensor.randn([32, 1])

pred = model(x)
loss = kore.functional.mse_loss(pred, y)
print(f"loss = {loss.numpy().item():.4f}")
```

### High-Level Training (PyTorch-style)

```python
import kore

# 1. Define model
model = kore.nn.Sequential([
    kore.nn.Linear(16, 64),
    kore.nn.Linear(64, 1),
])

# 2. Prepare data
x = kore.Tensor.randn([200, 16])
y = kore.Tensor.randn([200, 1])
dataset = kore.data.TensorDataset(x, y)
loader  = kore.data.DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Train
trainer = kore.training.Trainer(
    model, kore.optim.Adam(lr=0.001), loss="mse"
)
losses = trainer.fit(loader, epochs=10)
print(f"final loss: {losses[-1]:.4f}")

# 4. Evaluate / predict
val_loss = trainer.evaluate(loader)
preds    = trainer.predict(loader)
```

### Quantized Layers

```python
# 1.58-bit ternary — up to 10× compression
bit = kore.nn.BitLinear(256, 128)
print(bit)                       # BitLinear(256→128, 10.7x)

# 2-bit quaternary — up to 16× compression
quat = kore.nn.QuatLinear(256, 128)
print(quat)                      # QuatLinear(256→128, 16.0x)
```

### Save & Load (safetensors)

```python
kore.save_state_dict({"weight": w}, "model.safetensors")
loaded = kore.load_state_dict("model.safetensors")
```

---

## Quickstart: Rust

Add KORE crates to your `Cargo.toml`:

```toml
[dependencies]
kore-core    = { path = "crates/kore-core" }
kore-nn      = { path = "crates/kore-nn" }
kore-optim   = { path = "crates/kore-optim" }
kore-data    = { path = "crates/kore-data" }
```

```rust
use kore_core::Tensor;
use kore_nn::prelude::*;
use kore_data::{TensorDataset, DataLoader};

// Build a model
let model = Sequential::new(vec![
    Box::new(Linear::new(16, 64, true)),
    Box::new(Linear::new(64, 1, true)),
]);

// Prepare data
let x = Tensor::randn(&[100, 16]);
let y = Tensor::randn(&[100, 1]);
let ds = TensorDataset::new(&x, &y);
let loader = DataLoader::new(Box::new(ds), 32, true, true, Some(42));

// Train
let optimizer = kore_optim::Adam::default_with_lr(0.001);
let config = TrainerConfig { log_every: 1, grad_clip_norm: 0.0 };
let mut trainer = Trainer::new(model, optimizer, kore_nn::mse_loss, config);
let history = trainer.fit(&loader, 10);
println!("final loss: {:.4}", history.losses().last().unwrap());
```

### Build & Test

```bash
cargo build --workspace
cargo test  --workspace
cargo clippy --workspace -- -D warnings
```

---

## CLI

The `kore` CLI provides built-in tools for inspection, training, serving, and exporting.

| Command | Description |
|---------|-------------|
| `kore info` | System info — SIMD capabilities, supported dtypes, available crates |
| `kore inspect --path model.koref` | Inspect `.koref`, `.safetensors`, or model directory metadata |
| `kore shard --model ./hf_model --output ./shards` | Shard HuggingFace safetensors into per-layer files |
| `kore run --model model.koref --prompt "Hello"` | Local generation from a `.koref` model |
| `kore bench --sizes 128,256,512` | Matrix multiply & Flash Attention benchmarks |
| `kore train --steps 100 --lr 0.001 --scheduler warmup_cosine` | Demo training loop with LR schedulers |
| `kore serve --addr 0.0.0.0:8080` | OpenAI-compatible inference server (model-agnostic) |
| `kore export --model ./my_model --quantize ternary` | Export to `.koref` for edge inference |

---

## Inference Server

KORE ships a model-agnostic inference server with an **OpenAI-compatible REST API**. Any model that implements the `InferenceModel` trait can be served instantly.

### Implement `InferenceModel`

```rust
use kore_serve::{InferenceModel, state::AppState};
use kore_nn::sampler::{SamplerConfig, Rng};

struct MyModel { /* your weights */ }

impl InferenceModel for MyModel {
    fn generate_with_config(
        &mut self,
        prompt_tokens: &[usize],
        max_tokens: usize,
        config: &SamplerConfig,
        rng: &mut Rng,
    ) -> Result<Vec<usize>, String> {
        // Your generation logic here
        Ok(prompt_tokens.to_vec())
    }
}

// Wire it up
let model = MyModel { /* ... */ };
let state = AppState::with_model(model, "my-model".into());
kore_serve::server::serve_with_state("0.0.0.0:8080", state).await?;
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/completions` | Text completion (+ SSE streaming) |
| `POST` | `/v1/chat/completions` | Chat completion (+ SSE streaming) |
| `GET`  | `/v1/models` | List loaded models |
| `GET`  | `/health` | Health check |

### curl Examples

```bash
# Health check
curl http://localhost:8080/health

# Text completion
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"my-model","prompt":"Hello, ","max_tokens":50}'

# Chat completion with streaming
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"my-model","messages":[{"role":"user","content":"Hi"}],"stream":true}'
```

---

## Layered Inference (Large Model Support)

Run **70B+ parameter models on 4GB VRAM** using layer-by-layer sharded inference — inspired by [AirLLM](https://github.com/lyogavin/airllm), rebuilt in Rust.

**How it works:** Instead of loading the entire model into GPU memory, KORE loads one layer at a time, runs the forward pass, frees VRAM, and moves to the next layer. Combined with KORE's ternary/quaternary compression (8–16× smaller files), disk I/O becomes the bottleneck — which async prefetching solves.

```
Time →
Layer N:   [====COMPUTE====]
Layer N+1:      [LOAD][DECOMPRESS][GPU_XFER]
Layer N+2:                  [LOAD][DECOMPRESS][GPU_XFER]
```

Key features:
- **Async double-buffered prefetching** — loads layer N+1 while computing on layer N
- **LRU RAM cache** — keeps frequently-used layers in memory for faster autoregressive generation
- **BTES compression** — ternary (1.58-bit) and quaternary (2-bit) layer shards via `kore-btes`
- **Model sharding** — one-time conversion from HuggingFace safetensors to per-layer shards

See `kore_serve::layered` for the full API.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Python API (PyO3)                        │
│                     import kore                              │
├──────────┬───────────┬───────────┬───────────┬──────────────┤
│ kore-nn  │ kore-optim│ kore-btes │ kore-     │ kore-        │
│ Module   │ Adam/SGD  │ Ternary/  │ clifford  │ attention    │
│ Linear   │ Schedulers│ Quaternary│ Geometric │ Flash/Paged  │
│ Conv/Norm│ Clipping  │ VT-ALU    │ Algebra   │ KV-Cache     │
│ Sampler  │           │           │           │              │
├──────────┴───────────┴───────────┴───────────┴──────────────┤
│                     kore-autograd                            │
│              Computation graph, backward, tape               │
├──────────────────────────────────────────────────────────────┤
│                      kore-core                               │
│           Tensor, DType, Device, Storage, SIMD               │
├──────────────────────────────────────────────────────────────┤
│                    kore-kernels                              │
│      CUDA (cudarc + PTX) │ ROCm (HIP) │ CPU (AVX2/512, NEON)│
└──────────────────────────────────────────────────────────────┘
```

## Crates

| Crate | Description |
|-------|-------------|
| `kore-core` | Tensor, DType, Device, Storage, shape ops, SIMD |
| `kore-autograd` | Computation graph, backward pass, gradient tape |
| `kore-nn` | Module trait, Sequential, ModuleList, Trainer, Linear, Conv, LayerNorm, RMSNorm, BitLinear, QuatLinear, LoRA, SqueezeNet, Sampler |
| `kore-optim` | SGD, Adam, LR schedulers (cosine, warmup, one-cycle, step), gradient clipping |
| `kore-btes` | Binary/Ternary/Quaternary encoding, VT-ALU, matmul |
| `kore-kernels` | CUDA kernels (cudarc + PTX), ROCm/HIP, CPU SIMD (AVX2, NEON) |
| `kore-clifford` | Geometric algebra engine |
| `kore-attention` | Flash Attention, paged KV-cache |
| `kore-edge` | No-std inference runtime: WASM, iOS, Android |
| `kore-data` | StreamingDataset, TensorDataset, DataLoader, MultipackSampler, TokenBatcher |
| `kore-serve` | Model-agnostic inference server (axum, OpenAI-compatible) |
| `kore-python` | PyO3 bindings — `import kore` with nn, optim, data, training submodules |
| `kore-cli` | CLI: info, inspect, shard, run, bench, train, serve, export |

---

## Model Implementations

KORE is the engine. Model architectures live in **[Xura](https://github.com/KidIkaros/Xura)**:

| Xura Crate | Architecture |
|------------|-------------|
| `xura-mamba` | Mamba / Mamba-2 / Mamba-3 SSM |
| `xura-vljepa` | Vision-Language JEPA (Mamba3-JEPA) |
| `xura-transformer` | Decoder transformers (LLaMA, GPT, Mistral-style) |

Implement `kore_serve::InferenceModel` in your model crate to serve it via the built-in OpenAI-compatible API.

---

## Examples

| Notebook | Description |
|----------|-------------|
| [`colab_quickstart.ipynb`](examples/colab_quickstart.ipynb) | Install, tensors, autograd, training, quantized layers, CNN, save/load |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for build instructions, code style, and PR conventions.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

Copyright 2025–2026 Ikaros Digital LLC
