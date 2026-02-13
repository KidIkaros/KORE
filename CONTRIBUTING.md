# Contributing to KORE

Thanks for your interest in contributing! Here's how to get started.

## Build & Test

```bash
# Build all crates
cargo build --workspace

# Run all tests (588+)
cargo test --workspace

# Lint
cargo clippy --workspace -- -D warnings
```

## Python Bindings

```bash
pip install maturin
maturin develop --release
python -c "import kore; print(kore.__version__)"
```

## Code Style

- **No bare `unwrap()` in production code.** Use `?` for error propagation or descriptive `expect("reason")`.
- **Return `Result`** from any function that can fail (shape mismatch, unsupported dtype, etc.).
- **Precondition asserts** for invariants that indicate programmer error (e.g., `assert!(kernel_size > 0)`).
- **`debug_assert!`** only for checks that are too expensive for release builds.
- Use `total_cmp()` instead of `partial_cmp().unwrap()` for NaN-safe float sorting.
- Guard against division by zero and `usize` underflow explicitly.

## PR Conventions

1. Branch from `main`.
2. One logical change per commit with a descriptive message: `fix:`, `feat:`, `refactor:`, `test:`, `docs:`.
3. All tests must pass (`cargo test --workspace`).
4. No clippy warnings (`cargo clippy --workspace -- -D warnings`).
5. Add tests for new functionality.

## Project Structure

```
crates/
  kore-core/        # Tensor, DType, Device, Storage, SIMD
  kore-autograd/    # Computation graph, backward pass
  kore-nn/          # Layers: Linear, Conv, Norm, Embedding, etc.
  kore-optim/       # Adam, SGD, LR schedulers, gradient clipping
  kore-btes/        # Binary/Ternary/Quaternary encoding
  kore-kernels/     # CUDA + CPU SIMD kernels
  kore-clifford/    # Geometric algebra
  kore-attention/   # Flash Attention, paged KV-cache
  kore-serve/       # Model-agnostic inference server (axum)
  kore-edge/        # No-std inference runtime
  kore-data/        # Dataset utilities
  kore-python/      # PyO3 bindings
  kore-cli/         # CLI: info, bench, train, serve, export
python/kore/        # Python package root
```

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
