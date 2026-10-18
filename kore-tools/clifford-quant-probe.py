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

Arm C (report_rotations) asks the two follow-up questions the arm-B verdict
left open, at one shared rate convention (b + one f16 scale per 128-wide block,
i.e. 0.125 b/w -- every transform here is orthonormal and fixed, so its
parameters cost zero stored bits):

  AFF128s  no rotation, per-128 symmetric affine
  AFF128h  full 128-wide normalized Hadamard, then AFF128s
  AFF4q    SO(4) isoclinic T(v) = q_L v conj(q_R) per 4-chunk, then AFF128s
  AFF2p    2D Givens rotation per pair, then AFF128s
  HQMQ     Hurwitz 24-cell x S random unit quaternions (the multiplicative
           codebook of arXiv:2605.27646), radius at b_r bits

Arm C answers: (1) does rotation width matter, and (2) does HQMQ's codebook
beat rotation+scalar at a matched rate.
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


# ================================================================ arm C
# Does the *rotation width* matter, and does HQMQ's multiplicative Hurwitz
# codebook beat rotation+scalar at an exactly matched rate?
#
# Every arm below is scored at the same rate convention: b adjustable bits per
# value plus one f16 scale per 128-wide block (0.125 b/w). All transformations
# are orthonormal and fixed, so their parameters cost zero stored bits -- the
# same assumption the Hadamard arm has always made.


def to_blocks(x: np.ndarray, width: int, cap: int = 1 << 20) -> np.ndarray:
    flat = x.reshape(-1)[:cap]
    n = (flat.size // width) * width
    return flat[:n].reshape(-1, width).astype(np.float64)


def sym_aff_rows(b: np.ndarray, bits: int) -> np.ndarray:
    """Symmetric affine, one scale per row. rate = bits + 16/width."""
    hi = 2 ** (bits - 1) - 1
    s = np.maximum(np.abs(b).max(axis=1, keepdims=True) / hi, 1e-12)
    return np.clip(np.rint(b / s), -hi, hi) * s


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


_QCONJ = np.array([1.0, -1.0, -1.0, -1.0])


def hurwitz() -> np.ndarray:
    """The 24 units of the binary tetrahedral group 2T = vertices of the 24-cell."""
    pts = []
    for i in range(4):
        for s in (-1.0, 1.0):
            v = np.zeros(4)
            v[i] = s
            pts.append(v)
    for sw in (-1, 1):
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    pts.append(np.array([sw, sx, sy, sz], dtype=np.float64) * 0.5)
    return np.array(pts)


def lexi_unit(rng, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def nn_codebook(u: np.ndarray, cb: np.ndarray, chunk: int = 2048) -> np.ndarray:
    out = np.empty(u.shape[0], dtype=np.int64)
    ct = cb.T.astype(np.float32)
    for i in range(0, u.shape[0], chunk):
        out[i:i + chunk] = (u[i:i + chunk].astype(np.float32) @ ct).argmax(1)
    return out


def hqmq(b: np.ndarray, S: int, bits_r: int, seed: int = 0) -> np.ndarray:
    """HQMQ: each 4-chunk's direction -> q_p * q_s, q_p in Hurwitz 2T, q_s random.

    24S effective codewords from S stored quaternions. Radius gets bits_r bits
    with one f16 scale per 128-wide row (matching every other arm here).
    """
    n, w = b.shape
    ch = w // 4
    v = b.reshape(n, ch, 4)
    nrm = np.linalg.norm(v, axis=2)
    u = v / np.maximum(nrm[..., None], 1e-12)

    rng = np.random.default_rng(seed)
    qs = lexi_unit(rng, S)
    cb = quat_mul(hurwitz()[:, None, :], qs[None, :, :]).reshape(-1, 4)

    idx = nn_codebook(u.reshape(-1, 4), cb)
    uq = cb[idx].reshape(n, ch, 4)

    top = (2 ** bits_r - 1)
    sc = np.maximum(nrm.max(axis=1, keepdims=True) / top, 1e-12)
    rq = np.clip(np.rint(nrm / sc), 0, top) * sc
    return (uq * rq[..., None]).reshape(n, w)


def hqmq_rate(S: int, bits_r: int, width: int = 128) -> float:
    return (np.ceil(np.log2(24 * S)) + bits_r) / 4 + 16.0 / width


def rot_quat4(b: np.ndarray, seed: int = 0) -> np.ndarray:
    """SO(4) isoclinic rotation T(v) = q_L v conj(q_R), applied per 4-chunk."""
    rng = np.random.default_rng(seed)
    ql = lexi_unit(rng, 1)[0]
    qr = lexi_unit(rng, 1)[0]
    v = b.reshape(-1, 4)
    return quat_mul(quat_mul(ql, v), _QCONJ * qr).reshape(b.shape)


def rot_quat4_inv(b: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ql = lexi_unit(rng, 1)[0]
    qr = lexi_unit(rng, 1)[0]
    v = b.reshape(-1, 4)
    return quat_mul(quat_mul(_QCONJ * ql, v), qr).reshape(b.shape)


def rot_givens2(b: np.ndarray, seed: int = 0, inverse: bool = False) -> np.ndarray:
    th = np.random.default_rng(seed).uniform(0, np.pi)
    c, s = np.cos(th), np.sin(th)
    if inverse:
        s = -s
    v = b.reshape(-1, 2)
    return np.stack([c * v[:, 0] - s * v[:, 1],
                     s * v[:, 0] + c * v[:, 1]], axis=1).reshape(b.shape)


def rot_hadamard(b: np.ndarray) -> np.ndarray:
    h = hadamard(b.shape[1])
    return b @ h.T


def report_rotations(tag: str, x: np.ndarray):
    b = to_blocks(x, 128)
    print(f"\n== {tag}  ({b.size} values, {b.shape[0]} x 128) ==")
    print(f"{'scheme':<12}{'bits/w':>8}{'rel RMSE':>11}")
    rows = []

    for bits in (3, 4, 5):
        rate = bits + 16 / 128
        rows.append(("AFF128s", rate, rel(b, sym_aff_rows(b, bits))))

        h = rot_hadamard(b)
        rows.append(("AFF128h", rate, rel(b, rot_hadamard(sym_aff_rows(h, bits)))))

        q = rot_quat4(b)
        rec = rot_quat4_inv(sym_aff_rows(q, bits))
        rows.append(("AFF4q", rate, rel(b, rec)))

        p = rot_givens2(b)
        rows.append(("AFF2p", rate, rel(b, rot_givens2(sym_aff_rows(p, bits), inverse=True))))

    for S, br in ((24, 3), (96, 4), (384, 5)):
        rows.append((f"HQMQ{24 * S}", hqmq_rate(S, br), rel(b, hqmq(b, S, br))))

    for name, rate, err in sorted(rows, key=lambda r: r[1]):
        print(f"{name:<12}{rate:>8.3f}{err:>11.4%}")
    return rows


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

    report_rotations("POCKET f32 router weights", r)
    report_rotations("Ornith ffn_down (Q6_K)", f)
