## Question

How does KORE's KV-cache subsystem **cut memory-bound stalls and allow bigger
batches / longer context** (which also raises end-to-end tokens/sec)?

- Paged KV (PagedAttention idea: near-zero KV waste, 2-4x throughput).
- KV streaming (KV cache in RAM, only reads needed parts in VRAM - Strata).
- 4-bit KV cache + Hadamard rotation (Strata) / KVQuant's 3-bit KV.
- Block-granularity memory sharing (parallel sampling).

## Claimed by

(unassigned)

## Type / mode

`wayfinder:task` (AFK). Ends when a measured before/after on the user's hardware
shows a throughput gain, plus golden tests.
