# KORE v0.2.1 — Inference on a 16 GB AMD card

This release is the first one where the inference path is measured on the
target machine (RX 9060 XT, gfx1200) rather than inferred from a smaller card.
It also closes the one test that had been red on `main`.

## What is faster, and by how much

All ratios are versus the greedy arm measured in the same process on the same
cooled die, after warming every arm. Raw tok/s is what the card actually
sustained; the ratio is what the change did.

### Qwen3.6-35B-A3B (MoE, 3.99x over its own dense path at the start)

Speculative decoding, 4 prompts x 400 tokens, k=6:

| regime | greedy ms/step | spec ms/step | ratio |
|---|---|---|---|
| loop      | 117.9 | 66.8 | 1.76x |
| code      | 101.5 | 68.6 | 1.48x |
| prose     |  76.9 | 69.2 | 1.11x |
| tech      |  65.6 | 68.8 | 0.95-1.00x |

The tech regime is at parity, not a win: its greedy baseline is already cheap
enough that verification overhead cancels the gain. That is stated here rather
than averaged away, because the same table with tech excluded would read as a
much better release.

### Dense 2B

| policy | tok/s | ratio |
|---|---|---|
| greedy | 109.6 | 1.00x |
| k=6, min_match=6 | 173.0 | 1.58x |
| k=3, min_match=4 | — | 1.53x |

### Flint (40B-A2.5B MoE), host path

7.4-7.8 tok/s warm decode, with 4 prompts x 48 greedy ids identical to the
reference. This is the baseline the device port is measured against; the port
is not finished in this release, so no device number is claimed.

## What was measured and rejected

The handoff documents each of these with the numbers that killed it, so nobody
rebuilds them: guard re-probe backoff, two heads per `gdn_step` workgroup,
single-lane k-selection, hoisted softmax `exp`, vec4 router matvec, one-warp
`moe_topk`, pre-batch snapshot+replay, device-side embedding handoff, sparse
attention, full residency, dense gate+up fusion, routed gate+up fusion (a
168 MiB host-resident bank at 10 GB/s), `router_matvec`+`moe_topk` fusion, and
`min_match=8` as a default.

Three cache ideas were measured against the route profile and two were
refuted: static hot-expert pinning (fit-on-itself 90 %, held out 39.9-42.6 % at
8 GiB versus 33.6 % random) and previous-token prefetch (0.83/6, chance 0.14).
What survives is within-document recency, which is why the Flint device
architecture is "all layers resident, experts as a recency LRU" rather than a
layer split.

## Correctness work

- Sink + window decode attention and flat QK-norm/RoPE are gated against CPU
  oracles at max |Δ| 2e-5 and 1e-5 on random weights.
- `quantize_q8_rows` had a `.5` tie decided by the driver. The shader now pins
  the tie and the test asserts the achievable contract (documented in
  `docs/inference-handoff.md` section 3). Byte identity across lanes would
  require passing a host-computed reciprocal into the kernel; that is a roadmap
  item, not a fix, because the q8 KV cache dequantizes with the scale the same
  kernel returns.
- Full workspace suite green: 36/36 GPU op tests, 176 `kore-nn` lib tests, all
  crates passing. `cargo clippy --workspace -- -D warnings` is clean.

## What this release is not

- There is no end-to-end device model. The Vulkan kernels and the host-path
  Flint port exist and are tested; the device `attn_step`, routed MoE lane and
  on-device head are not wired yet.
- Flint has no external oracle yet. It agrees with itself across prompts, which
  is weaker than agreeing with llama.cpp.
- `clippy --all-targets` is error-free but still emits warnings in test code
  (unused names in cfg-gated helpers). CI does not gate on them.

Roadmap: `docs/roadmap.md`.
