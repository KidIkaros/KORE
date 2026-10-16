## Question

How does KORE **raise effective tokens/sec lossless** via speculative decoding?

- Graft a fast draft head / MTP (speculative decoding) behind the target model - the
  sudoingX MTP head flipped 26.4 -> 40.5 tok/s (1.5x) on an RTX 3060; Strata uses
  MTP too.
- Greedy/argmax sampling path for the draft head.
- Kernel speedup for the draft-path matmul.

## Claimed by

(unassigned)

## Type / mode

`wayfinder:task` (AFK). Ends when a working MTP head + before/after tokens/sec
exists, proving the lossless gain.
