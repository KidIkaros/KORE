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
| `kore-nn` | Module trait, Linear, Conv, LayerNorm, activations |
| `kore-optim` | SGD, Adam, AdamW, LR schedulers |
| `kore-btes` | Binary/Ternary/Quaternary encoding, VT-ALU, matmul |
| `kore-kernels` | CUDA kernels (cudarc + PTX), CPU SIMD fallback |
| `kore-clifford` | Geometric algebra engine |
| `kore-attention` | Flash Attention, paged KV-cache |
| `kore-serve` | Inference server (axum, OpenAI-compatible) |
| `kore-python` | PyO3 bindings → `import kore` |
| `kore-cli` | CLI: train, serve, bench, info |

## Build

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
```

## License

MIT
