# Kore

**A pure Rust ML framework** — tensor engine, autograd, native ternary/quaternary quantization, geometric algebra, and CUDA kernel dispatch.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Python API (PyO3)                         │
│                     import kore                               │
├──────────┬───────────┬───────────┬───────────┬───────────────┤
│ kore-nn  │ kore-optim│ kore-btes │ kore-     │ kore-         │
│ Module   │ Adam/SGD  │ Ternary/  │ clifford  │ attention     │
│ Linear   │ Schedulers│ Quaternary│ Geometric │ Flash/Paged   │
│ Conv/Norm│           │ VT-ALU    │ Algebra   │ KV-Cache      │
├──────────┴───────────┴───────────┴───────────┴───────────────┤
│                     kore-autograd                             │
│              Computation graph, backward, tape                │
├──────────────────────────────────────────────────────────────┤
│                      kore-core                                │
│           Tensor, DType, Device, Storage, SIMD                │
├──────────────────────────────────────────────────────────────┤
│                    kore-kernels                               │
│          CUDA (cudarc + PTX) │ CPU (AVX2/512, NEON)          │
└──────────────────────────────────────────────────────────────┘
```

## Crates

| Crate | Description |
|-------|-------------|
| `kore-core` | Tensor, DType, Device, Storage, shape ops, SIMD |
| `kore-autograd` | Computation graph, backward pass, gradient tape |
| `kore-nn` | Module trait, Linear, Conv, LayerNorm, RMSNorm, BitLinear, QuatLinear, SqueezeNet |
| `kore-optim` | SGD, Adam, LR schedulers, gradient clipping |
| `kore-btes` | Binary/Ternary/Quaternary encoding, VT-ALU, matmul |
| `kore-kernels` | CUDA kernels (cudarc + PTX), CPU SIMD (AVX2, NEON) |
| `kore-clifford` | Geometric algebra engine |
| `kore-attention` | Flash Attention, paged KV-cache |
| `kore-transformer` | Decoder transformer, BitNetTransformer, QuatNetTransformer |
| `kore-edge` | No-std inference runtime: WASM, iOS, Android |
| `kore-data` | StreamingDataset, MultipackSampler, TokenBatcher |
| `kore-mamba` | Mamba (selective state-space) model |
| `kore-vljepa` | Vision-Language JEPA model |
| `kore-serve` | Inference server (axum, OpenAI-compatible) |
| `kore-python` | PyO3 bindings → `pip install kore` (maturin) |
| `kore-cli` | CLI: train, serve, bench, info |

## Build

```bash
cargo build --workspace
cargo test --workspace
```

## Python

Install from source with [maturin](https://github.com/PyO3/maturin):

```bash
pip install maturin
maturin develop --release
```

Then use in Python:

```python
import kore

# Tensors
x = kore.Tensor.randn([4, 16])
y = kore.Tensor.randn([4, 16])
z = x + y

# Autograd
x.requires_grad_(True)
loss = (x * x).sum()
loss.backward()
print(x.grad)

# Layers
linear = kore.nn.Linear(16, 8)
out = linear(x)

# Quantized layers (2-bit, 16× compression)
qlinear = kore.nn.QuatLinear(16, 8)
out = qlinear(x)
print(qlinear.compression_ratio())

# Save/load
kore.save_state_dict({"weight": x}, "model.safetensors")
loaded = kore.load_state_dict("model.safetensors")
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

Copyright 2025 Ikaros Digital LLC
