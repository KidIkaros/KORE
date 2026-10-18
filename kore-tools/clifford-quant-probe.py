"""Falsification probe, arm B: is a rotor-structured (spherical) quantizer
better than scalar-affine at the same bit rate?

A rotor is a point on S^3 up to sign; quantizing a rotor means quantizing three
angles. So the GA-native quantizer for a 4-vector is exactly a spherical
quantizer: {3 quantized angles} x {norm}. If GA has a quantization story, this
is it, and it can be tested on real weights with numpy alone.

Four schemes, all given the same data and measured in the *original* basis:

  AFF32    per-32 asymmetric affine, b-bit codes, f16 scale+offset
           rate = b + 1.0 bits/weight
  AFF32s   per-32 symmetric affine, b-bit codes, f16 scale
           rate = b + 0.5
  ROT4     per-4 spherical: 3 angles x b_a bits + 8-bit log norm
           rate = (3*b_a + 8) / 4
  AFF32h   AFF32 applied after a 32-wide normalized Hadamard and inverted back
           rate = b + 1.0 (identical to AFF32 -- the transform is free)

Plus gguf-py's own Q4_K on the same blocks, as the anchor for "what the shipped
format gets at its rate".

Prediction stated before running: AFF32 dominates ROT4 at every matched rate
(scalar quantization of an i.i.d. near-Gaussian source is near the
rate-distortion bound; a spherical chart has a non-uniform metric that wastes
code points), and AFF32h is a statistical tie with AFF32 on Gaussian data,
winning only if the blocks are heavy-tailed.
"""

import numpy as np
import gguf

rng = np.random.default_rng(0)


def hadamard(n: int) -> np.ndarray:
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def blocks_of(x: np.ndarray, width: int) -> np.ndarray:
    flat = x.reshape(-1)
    n = (flat.size // width) * width
    return flat[:n].reshape(-1, width)


# ---------------------------------------------------------------- schemes
def aff32(blk: np.ndarray, bits: int, symmetric: bool) -> np.ndarray:
    if symmetric:
        scale = np.abs(blk).max(axis=1, keepdims=True) / (2 ** (bits - 1) - 1)
        scale = np.maximum(scale, 1e-12)
        q = np.clip(np.rint(blk / scale), -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1)
        return q * scale
    lo = blk.min(axis=1, keepdims=True)
    hi = blk.max(axis=1, keepdims=True)
    span = np.maximum(hi - lo, 1e-12)
    q = np.clip(np.rint((blk - lo) / span * (2**bits - 1)), 0, 2**bits - 1)
    return q / (2**bits - 1) * span + lo


def spherical4(blk: np.ndarray, bits_a: int) -> np.ndarray:
    """Per-4 vector: quantize direction on S^3 (3 angles) and an 8-bit log norm."""
    v = blk.reshape(-1, 4)
    nrm = np.linalg.norm(v, axis=1)
    u = v / np.maximum(nrm[:, None], 1e-12)
    a = np.arccos(np.clip(u[:, 0], -1, 1))            # [0, pi]
    s = np.maximum(np.sin(a), 1e-12)
    b = np.arccos(np.clip(u[:, 1] / s, -1, 1))        # [0, pi]
    c = np.arctan2(u[:, 3], u[:, 2]) % (2 * np.pi)    # [0, 2pi)

    def qz(ang, lo, hi):
        step = (hi - lo) / (2**bits_a - 1)
        return lo + np.rint((ang - lo) / step) * step

    a, b, c = qz(a, 0, np.pi), qz(b, 0, np.pi), qz(c, 0, 2 * np.pi)
    uq = np.stack(
        [np.cos(a), np.sin(a) * np.cos(b),
         np.sin(a) * np.sin(b) * np.cos(c),
         np.sin(a) * np.sin(b) * np.sin(c)], axis=1)
    # 8-bit log norm, one exponent byte over the block's own dynamic range
    lo, hi = np.log(nrm.min() + 1e-12), np.log(nrm.max() + 1e-12)
    ln = lo + np.rint((np.log(nrm + 1e-12) - lo) / max((hi - lo) / 255, 1e-12)) * ((hi - lo) / 255)
    return (uq * np.exp(ln)[:, None]).reshape(blk.shape)


def rmse(truth: np.ndarray, rec: np.ndarray) -> float:
    return float(np.sqrt(((truth - rec) ** 2).mean()))


def rel(truth: np.ndarray, rec: np.ndarray) -> float:
    return rmse(truth, rec) / float(np.sqrt((truth**2).mean()))


# ---------------------------------------------------------------- data
def f32_routers(path: str, layers=6, rows=96) -> np.ndarray:
    r = gguf.GGUFReader(path)
    out = []
    for name in sorted(t.name for t in r.tensors if t.name.endswith("ffn_gate_inp.weight")):
        if len(out) >= layers:
            break
        t = next(t for t in r.tensors if t.name == name)
        d = np.array(t.data, dtype=np.float32).reshape(-1, t.shape[0])
        out.append(d[:rows])
    return np.concatenate(out, axis=0)


def q6_tensor(path: str, name: str, rows=4096) -> np.ndarray:
    r = gguf.GGUFReader(path)
    t = next(t for t in r.tensors if t.name == name)
    d = gguf.quants.dequantize(np.array(t.data), t.tensor_type)
    d = d.reshape(-1, t.shape[0])
    return d[:rows]


def report(tag: str, truth: np.ndarray):
    print(f"\n== {tag}  ({truth.size} values, std {truth.std():.4f}) ==")
    print(f"{'scheme':<10}{'bits/w':>8}{'rel RMSE':>11}")
    for bits in (2, 3, 4, 5, 6):
        rec = aff32(truth, bits, symmetric=False)
        print(f"{'AFF32':<10}{bits + 1.0:>8.2f}{rel(truth, rec):>11.4%}")
    for bits in (2, 3, 4, 5, 6):
        rec = aff32(truth, bits, symmetric=True)
        print(f"{'AFF32s':<10}{bits + 0.5:>8.2f}{rel(truth, rec):>11.4%}")
    for ba in (2, 3, 4, 5, 6, 8):
        rec = spherical4(truth, ba)
        print(f"{'ROT4':<10}{(3 * ba + 8) / 4:>8.2f}{rel(truth, rec):>11.4%}")
    h = hadamard(32)
    rot = truth @ h.T
    for bits in (3, 4, 5):
        rec = aff32(rot, bits, symmetric=False) @ h
        print(f"{'AFF32h':<10}{bits + 1.0:>8.2f}{rel(truth, rec):>11.4%}")


def q4k_anchor(truth: np.ndarray):
    # gguf-py quantizes 256-element super-blocks
    flat = truth.reshape(-1)
    n = (flat.size // 256) * 256
    q = gguf.quants.quantize(flat[:n], gguf.GGMLQuantizationType.Q4_K)
    d = gguf.quants.dequantize(q, gguf.GGMLQuantizationType.Q4_K)
    rate = q.dtype.itemsize * 8 / 256
    print(f"{'Q4_K':<10}{rate:>8.2f}{rel(flat[:n], d):>11.4%}")


if __name__ == "__main__":
    pocket = "/tmp/opencode/pocket-35b-q2k.gguf"
    ornith = "/home/ikaaros/Coding/Gold/kore/models/ornith-9b/Ornith-1.0-9B-Q4_K_M.gguf"

    r = f32_routers(pocket)
    report("POCKET f32 router weights (exact pretrained weights)", blocks_of(r, 32))
    # (gguf-py can dequantize Q4_K but not quantize it, so no anchor here)

    f = q6_tensor(ornith, "blk.0.ffn_down.weight")
    report("Ornith ffn_down (Q6_K, near-lossless proxy for f32)", blocks_of(f, 32))

    # how heavy-tailed are the blocks? (a rotation can only help if they are)
    for tag, x in (("router", r), ("ffn_down", f)):
        b = blocks_of(x, 32)
        kurt = float(((b - b.mean(1, keepdims=True)) ** 4).mean() / (b.var(1) ** 2).mean())
        print(f"\n{tag}: block excess kurtosis {kurt - 3:.3f} (0 = Gaussian)")
