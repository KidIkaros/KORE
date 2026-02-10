```
 _  _____  ____  _____
| |/ / _ \|  _ \| ____|
| ' / | | | |_) |  _|
| . \ |_| |  _ <| |___
|_|\_\___/|_| \_\_____|
```

# KORE

**A pure-Rust ML framework — from training to edge inference.**

[![CI](https://github.com/KidIkaros/KORE/actions/workflows/ci.yml/badge.svg)](https://github.com/KidIkaros/KORE/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org)

---

## Highlights

- **Autograd engine** — computation graph with automatic backward pass
- **Native quantization** — 1.58-bit (ternary) and 2-bit (quaternary) layers with 8–16× compression
- **LLM training** — decoder transformers (LLaMA, GPT, Mistral-style) with LoRA fine-tuning
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

### Layers & Training

```python
model = kore.nn.Linear(16, 1)
optimizer = kore.optim.Adam(lr=0.001)

x = kore.Tensor.randn([32, 16])
y = kore.Tensor.randn([32, 1])

pred = model(x)
loss = kore.functional.mse_loss(pred, y)
print(f"loss = {loss.numpy().item():.4f}")
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
kore-autograd = { path = "crates/kore-autograd" }
kore-optim   = { path = "crates/kore-optim" }
```

```rust
use kore_core::Tensor;
use kore_nn::{Linear, Module};

let x = Tensor::randn(&[32, 16]);
let model = Linear::new(16, 1, true);
let out = model.forward(&x).unwrap();
println!("output shape: {:?}", out.shape().dims());
```

### Build & Test

```bash
cargo build --workspace
cargo test  --workspace          # 588+ tests
cargo clippy --workspace -- -D warnings
```

---

## CLI

The `kore` CLI provides built-in tools for benchmarking, training, serving, and exporting.

| Command | Description |
|---------|-------------|
| `kore info` | System info — SIMD capabilities, supported dtypes, available crates |
| `kore bench --sizes 128,256,512` | Matrix multiply & Flash Attention benchmarks |
| `kore train --steps 100 --lr 0.001 --scheduler warmup_cosine` | Demo training loop with LR schedulers |
| `kore serve --addr 0.0.0.0:8080 --model ./my_model` | OpenAI-compatible inference server |
| `kore generate --prompt "Hello" --max-tokens 32` | Text generation demo |
| `kore export --model ./my_model --quantize ternary` | Export to `.koref` for edge inference |

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
├──────────┴───────────┴───────────┴───────────┴──────────────┤
│                     kore-autograd                            │
│              Computation graph, backward, tape               │
├──────────────────────────────────────────────────────────────┤
│                      kore-core                               │
│           Tensor, DType, Device, Storage, SIMD               │
├──────────────────────────────────────────────────────────────┤
│                    kore-kernels                              │
│          CUDA (cudarc + PTX) │ CPU (AVX2/512, NEON)         │
└──────────────────────────────────────────────────────────────┘
```

## Crates

| Crate | Description |
|-------|-------------|
| `kore-core` | Tensor, DType, Device, Storage, shape ops, SIMD |
| `kore-autograd` | Computation graph, backward pass, gradient tape |
| `kore-nn` | Module trait, Linear, Conv, LayerNorm, RMSNorm, BitLinear, QuatLinear, LoRA, SqueezeNet |
| `kore-optim` | SGD, Adam, LR schedulers (cosine, warmup, one-cycle, step), gradient clipping |
| `kore-btes` | Binary/Ternary/Quaternary encoding, VT-ALU, matmul |
| `kore-kernels` | CUDA kernels (cudarc + PTX), CPU SIMD (AVX2, NEON) |
| `kore-clifford` | Geometric algebra engine |
| `kore-attention` | Flash Attention, paged KV-cache |
| `kore-transformer` | Decoder transformer, BitNetTransformer, QuatNetTransformer |
| `kore-edge` | No-std inference runtime: WASM, iOS, Android |
| `kore-data` | StreamingDataset, MultipackSampler, TokenBatcher |
| `kore-serve` | Inference server (axum, OpenAI-compatible) |
| `kore-python` | PyO3 bindings — `import kore` (maturin) |
| `kore-cli` | CLI: info, bench, train, serve, generate, export |

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

Copyright 2025 Ikaros Digital LLC
