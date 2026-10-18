#!/usr/bin/env python3
"""Generate the K-quant dequantization fixtures used by `k_quant`'s tests.

The point of this file is to make KORE's K-quant unpacking provable against an
*independent* implementation rather than against my reading of the format. The
oracle is `gguf.quants` -- the reference implementation shipped with `gguf-py`,
which is the same code llama.cpp's Python tooling uses. Anything KORE claims
about Q4_K/Q5_K/Q6_K block layouts is checked against it here.

For each type we emit, as `<name>.raw` (payload bytes) and `<name>.f32`
(reference dequantized f32, little-endian):

  real_<t>     first N blocks of a real tensor from a real GGUF
  real2_<t>    first N blocks of a real tensor from a second, unrelated GGUF
  rand_<t>     N blocks of pseudo-random bytes with finite f16 scales

`rand_*` is the important one: real tensors exercise the common path, but only
random bytes exercise every nibble, every high bit, and every scale/min packing
branch. The f16 scale fields are forced finite so that comparisons stay
meaningful (a random f16 can be inf/NaN, which would make the test vacuous).

Usage:  python3 gen.py <out_dir> [--models-dir ../../../../../models]
Requires: `pip install gguf numpy`
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

from gguf import GGUFReader
from gguf.quants import Q4_K, Q5_K, Q6_K

# ggml type ids, matching gguf_min::type_block.
TYPES = {12: ("q4_k", Q4_K), 13: ("q5_k", Q5_K), 14: ("q6_k", Q6_K)}

# Number of 256-weight blocks per fixture. Small: these get committed.
N_BLOCKS = 8

# The real GGUFs we slice from. Both are `qwen35` arch; the first is K-quant
# dense with Q4_K+Q5_K+Q6_K present, the second is llama-arch Q4_K+Q6_K only.
REAL_SOURCES = {
    "real": "ornith-9b/Ornith-1.0-9B-Q4_K_M.gguf",
    "real2": "minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf",
}


def first_tensor_of_type(reader: GGUFReader, type_id: int):
    """The first tensor in the file with this quant type (deterministic order)."""
    for t in reader.tensors:
        if int(t.tensor_type) == type_id:
            return t
    return None


def payload_blocks(tensor, type_size: int, n_blocks: int) -> np.ndarray:
    """First `n_blocks` raw blocks of a tensor's payload (uint8)."""
    raw = np.asarray(tensor.data).reshape(-1).view(np.uint8)
    want = n_blocks * type_size
    if raw.size < want:
        raise SystemExit(
            f"tensor {tensor.name} too small: {raw.size} < {want} bytes"
        )
    return raw[:want].reshape(n_blocks, type_size).copy()


def finite_random_blocks(type_size: int, n_blocks: int, rng: np.random.Generator,
                         scale_bytes: list[tuple[int, int]]) -> np.ndarray:
    """Random blocks with the f16 scale fields forced to finite values."""
    blocks = rng.integers(0, 256, size=(n_blocks, type_size), dtype=np.uint8)
    for lo, hi in scale_bytes:
        # f16 exponents 0x1c..0x2c are ordinary small magnitudes (2^-3 .. 2^4).
        exp = rng.integers(0x1C, 0x2C, size=(n_blocks, 1)).astype(np.uint8)
        sign = rng.integers(0, 2, size=(n_blocks, 1)).astype(np.uint8) << 7
        mant = rng.integers(0, 256, size=(n_blocks, 1)).astype(np.uint8)
        blocks[:, lo] = (sign | exp)[:, 0]
        blocks[:, hi] = mant[:, 0]
    return blocks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--models-dir", default=None)
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = pathlib.Path(args.models_dir) if args.models_dir else pathlib.Path(
        __file__).resolve().parents[5] / "models"

    rng = np.random.default_rng(0x4B51_4B33)  # "KQK3"
    manifest: dict[str, dict] = {}

    # Where the f16 scale fields live inside a block, per type.
    #   Q4_K/Q5_K: d at [0,2), dmin at [2,4)   -> force both finite
    #   Q6_K:      d at [208,210)              -> force that one
    scale_fields = {
        12: [(0, 1), (2, 3)],
        13: [(0, 1), (2, 3)],
        14: [(208, 209)],
    }

    readers: dict[str, GGUFReader] = {}
    for tag, rel in REAL_SOURCES.items():
        path = models / rel
        if not path.exists():
            print(f"skip {tag}: {path} not found", file=sys.stderr)
            continue
        readers[tag] = GGUFReader(str(path))
        print(f"loaded {tag}: {path}")

    cases: list[tuple[str, int, np.ndarray]] = []

    # Real slices.
    for tag, reader in readers.items():
        for type_id, (_, cls) in TYPES.items():
            t = first_tensor_of_type(reader, type_id)
            if t is None:
                continue
            blocks = payload_blocks(t, cls.type_size, N_BLOCKS)
            cases.append((f"{tag}_{TYPES[type_id][0]}", type_id, blocks))
            print(f"  {tag}_{TYPES[type_id][0]:6s} <- {t.name}  shape={list(t.shape)}")

    # Random blocks (synthetic, but exercising every bit pattern).
    for type_id, (name, cls) in TYPES.items():
        blocks = finite_random_blocks(
            cls.type_size, N_BLOCKS, rng, scale_fields[type_id])
        cases.append((f"rand_{name}", type_id, blocks))

    # An all-zero block: d = dmin = 0, so every output must be exactly 0.
    for type_id, (name, cls) in TYPES.items():
        cases.append((f"zero_{name}", type_id,
                      np.zeros((1, cls.type_size), dtype=np.uint8)))

    for name, type_id, blocks in sorted(cases):
        cls = TYPES[type_id][1]
        deq = cls.dequantize_blocks(blocks)  # reference implementation
        assert deq.dtype == np.float32, deq.dtype
        blocks.astype(np.uint8).tofile(out / f"{name}.raw")
        np.ascontiguousarray(deq.reshape(-1), dtype="<f4").tofile(out / f"{name}.f32")
        # The block tensor is [rows, cols] with ne[0] contiguous, so the flat
        # reference order is exactly the concatenation of the blocks.
        manifest[name] = {
            "type_id": type_id,
            "n_blocks": int(blocks.shape[0]),
            "block_bytes": int(cls.type_size),
            "block_elems": int(cls.block_size),
            "n_elems": int(deq.size),
        }

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    total = sum((out / f"{n}.{e}").stat().st_size for n in manifest for e in ("raw", "f32"))
    print(f"wrote {len(manifest)} fixtures, {total} bytes, to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
