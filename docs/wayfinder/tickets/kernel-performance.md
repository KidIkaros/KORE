## Question

What KORE kernel work **directly raises matmul + attention throughput** on the
user's backends (CPU SIMD / CUDA / ROCm / Vulkan)?

Pin these invariants before optimizing:
- Which kernels are the bottleneck (matmul GEMM, attention, KV read) - profile, don't guess.
- Kernel fusion points (matmul + activation + quantization).
- Fused int8 tensor-core / thread-block-cluster kernels (Strata's 0.1.36 fused prompt
  kernels reported +16-22% - quantify KORE's equivalent).
- Thread-block clusters + argmax-multi for faster decode argmax.
- Beam/thread scheduling for batched throughput (not just single-request latency).

## Claimed by

(unassigned)

## Type / mode

`wayfinder:task` (AFK). Ends when a profiling run + a kernel patch + a before/after
number exists for the bottleneck kernel.
