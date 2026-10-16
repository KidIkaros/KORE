## Question

What KORE engine work is needed to run on an 8-16 G b card (the "improbable
hardware")? Decide shape + pin invariants:

- KV streaming (KV cache offloaded to RAM/SSD; only reads what attention needs in VRAM)
- 4-bit K V cache (with Hadamard rotation) as an option
- K8V4 K V cache (8-bit keys / 4-bit values) as an option
- Quantized matmul golden tests (2-bit and ternary correctness)
- Cross-backend (CPU SIMD, CUDA, ROCm, Vulkan) so "improbable hardware" is a
  portability claim, not a single-card luck claim

## Claimed by

(unassigned)

## Blocks / blocked by

- blocked by: none
- unblocked

## Type / mode

`wayfinder:task` (AFK). Ends when a working branch plus golden tests + a measured
12 G b card run exists, then the ticket records the concrete facts (branch, numbers,
card).
