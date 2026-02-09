# kore-edge

Lightweight on-device inference runtime for mobile (Android/iOS), WASM (browser/Node), and embedded targets. Competes with ExecuTorch by leveraging Kore's native ternary/quaternary quantization.

## Key Features

- **`.koref` model format** — single-file, mmap-friendly, mixed quantization
- **Arena allocator** — zero-alloc steady-state inference
- **Ternary quantization** — 1.6 bits/weight (7B model in ~1.4 GB vs ~3.5 GB INT4)
- **SIMD backends** — ARM NEON (mobile), WASM SIMD128 (browser), scalar fallback
- **C FFI** — Swift (iOS), Kotlin/JNI (Android), C# (Unity), Dart (Flutter)
- **WASM bindings** — `wasm-bindgen` JS/TS API for browser inference

## Quick Start

### Export a model
```bash
kore export --model ./llama-3.2-1b --quantize ternary --output model.koref
```

### Run on device (Rust)
```rust
use kore_edge::{KorefModel, Session};

let data = std::fs::read("model.koref")?;
let model = KorefModel::from_bytes(&data)?;
let mut session = Session::new(model);
let tokens = session.generate(&[1, 2, 3], 32);
```

### Run in browser (WASM)
```js
import init, { KoreSession } from '@kore/edge';
await init();
const bytes = new Uint8Array(await (await fetch('model.koref')).arrayBuffer());
const session = KoreSession.fromBytes(bytes);
const tokens = session.generate(new Uint32Array([1, 2, 3]), 32);
```

### Run on Android (Kotlin)
```kotlin
val session = KoreEdge.load("/data/local/tmp/model.koref")
val tokens = session.generate(intArrayOf(1, 2, 3), maxTokens = 32)
session.close()
```

### Run on iOS (Swift)
```swift
let session = try KoreEdge(path: Bundle.main.path(forResource: "model", ofType: "koref")!)
let tokens = session.generate(inputIds: [1, 2, 3], maxTokens: 32)
```

## Cross-Compilation

```bash
# Install tools
cargo install cross cargo-make wasm-pack

# Build for all targets
cargo make build-all

# Individual targets
cargo make build-android    # aarch64-linux-android
cargo make build-ios        # aarch64-apple-ios
cargo make build-wasm       # wasm32 via wasm-pack
cargo make build-rpi        # aarch64-unknown-linux-gnu
```

## Benchmark
```bash
cargo run --example bench_edge -p kore-edge --release
```

## Architecture

```
kore-edge/src/
├── format.rs          # .koref model format (read/write/builder)
├── arena.rs           # Arena allocator for zero-alloc inference
├── runtime.rs         # Inference session with KV-cache
├── plan.rs            # Execution plan + memory estimation
├── ops/               # Portable operator library (scalar)
│   ├── matmul.rs      # f32 + ternary + quaternary matmul
│   ├── norm.rs        # RMSNorm, LayerNorm
│   ├── activation.rs  # ReLU, GELU, SiLU, Sigmoid, Softmax
│   ├── embedding.rs   # Token lookup
│   ├── attention.rs   # Multi-head attention with GQA
│   ├── rope.rs        # Rotary position embeddings
│   └── elementwise.rs # Residual add, mul, scale
├── simd_dispatch.rs   # Compile-time SIMD backend selection
├── neon.rs            # ARM NEON kernels (aarch64 only)
├── wasm_simd.rs       # WASM SIMD128 kernels (wasm32 only)
├── wasm_api.rs        # wasm-bindgen JS/TS API
└── ffi.rs             # C FFI for Swift/Kotlin
```
