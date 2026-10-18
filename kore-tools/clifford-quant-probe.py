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


def to_blocks(x: np.ndarray, width: int, cap: int | None = 1 << 20) -> np.ndarray:
    flat = x.reshape(-1)
    if cap is not None:
        flat = flat[:cap]
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


def rot_signs(b: np.ndarray, seed: int = 0) -> np.ndarray:
    """The random diagonal `D` of the usual `H D` incoherence rotation."""
    return np.random.default_rng(seed).choice(np.array([-1.0, 1.0]), size=b.shape[1])


def rot_hadamard_signed(b: np.ndarray, seed: int = 0) -> np.ndarray:
    """`H D x`: a plain WHT can *worsen* a structured input (worst case the
    transform aligns with the data), which is why QuaRot/SpinQuant randomise the
    signs. Cheap to check here whether real K/V care."""
    return (b * rot_signs(b, seed)) @ hadamard(b.shape[1]).T


def rot_hadamard_signed_inv(b: np.ndarray, seed: int = 0) -> np.ndarray:
    """`D H y` -- the exact inverse, since `H` is symmetric and self-inverse."""
    return (b @ hadamard(b.shape[1]).T) * rot_signs(b, seed)


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


# ================================================================ KV (P6.6a)
# Same arm-C question, asked of the vectors the attention kernel actually
# quantizes: post-RoPE K and raw V, dumped from the engine by
# `cargo run --release -p kore-nn --example kv_dump`.
#
# Width matters here in a way it does not for weights. For weights the rotation
# is free at any width (the transform is folded into the matmul input), so
# `report_rotations` can just compare widths. For the KV cache a wide transform
# *inside* the attention kernel costs `log2 d` per key-value pair -- which is why
# the literature sits at 4 -- and the way out is to rotate at *append*, where it
# commutes with the weighted sum and is therefore free at full width
# (sum_j a_j (R v_j) = R sum_j a_j v_j). So the numbers below are what decides
# whether the append-time route is worth the kernel work: if a 128-wide Hadamard
# does not beat a 4-wide/quaternion one on real K and V, the cheap in-kernel
# scheme wins and the append-time machinery is not justified.


def read_kv(path: str):
    """Parse the `KVDU` dump. Returns ([ (k, v), ... ], folded)."""
    import struct as _s

    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"KVDU":
            raise ValueError(f"{path}: bad magic {magic!r}")
        ver, n_layers, folded = _s.unpack("<III", f.read(12))
        if ver != 1:
            raise ValueError(f"{path}: unsupported version {ver}")
        out = []
        for _ in range(n_layers):
            kv_dim, ln = _s.unpack("<II", f.read(8))
            k = np.frombuffer(f.read(ln * kv_dim * 4), dtype="<f4").reshape(ln, kv_dim)
            v = np.frombuffer(f.read(ln * kv_dim * 4), dtype="<f4").reshape(ln, kv_dim)
            out.append((k, v))
    return out, bool(folded)


def armc(b: np.ndarray, bits: int):
    """The rotation arms at one bit rate: (rate, {name: rel RMSE}).

    `b` is `[n, width]`, and the width is the whole point -- pass 128 for the
    append-time scheme and 4 for the in-kernel compromise.
    """
    rate = bits + 16.0 / b.shape[1]
    h = rot_hadamard(b)
    hd = rot_hadamard_signed(b)
    q = rot_quat4(b)
    p = rot_givens2(b)
    return rate, {
        "AFF": rel(b, sym_aff_rows(b, bits)),
        "had": rel(b, rot_hadamard(sym_aff_rows(h, bits))),
        "hadD": rel(b, rot_hadamard_signed_inv(sym_aff_rows(hd, bits))),
        "quat4": rel(b, rot_quat4_inv(sym_aff_rows(q, bits))),
        "givens2": rel(b, rot_givens2(sym_aff_rows(p, bits), inverse=True)),
    }


def excess_kurtosis(b: np.ndarray) -> float:
    m = b.mean(axis=1, keepdims=True)
    var = ((b - m) ** 2).mean(axis=1)
    return float(((b - m) ** 4).mean() / max((var ** 2).mean(), 1e-30)) - 3.0


def report_kv(path: str, bits: int = 4, width: int = 128):
    layers, folded = read_kv(path)
    n_layers = len(layers)
    ln, kv_dim = layers[0][0].shape
    print(
        f"\n== KV dump {path}\n   {n_layers} full-attention layers, kv_dim {kv_dim}, "
        f"len {ln}, folded={folded}, {bits} adjustable bits + one f16 scale per block =="
    )

    # The width sweep is the part that is *not* the same as the weight question.
    # A wider block is also a *cheaper* block: one f16 scale per `width` values,
    # so widening lowers the rate as well as the error. The consumer decides how
    # wide it can afford to be -- for the KV cache, in-kernel costs log2(width)
    # per key-value pair and append-time costs nothing -- so this prints both the
    # quality and the price for each width and leaves the trade explicit.
    #
    # Averaged per layer rather than pooled: pooling concatenates layers and a
    # cap then silently truncates to the first layer, which hid a layer that
    # improves 2.6x behind a mean that said 1.2x.
    arms = ("AFF", "had", "hadD", "quat4", "givens2")
    widths = (32, 64, 128, 256)
    for half, which in (("K", 0), ("V", 1)):
        print(f"\n-- {half} width sweep (mean over {n_layers} layers, all values)")
        print(f"{'width':>6}{'b/w':>8}" + "".join(f"{n:>10}" for n in arms) + f"{'kurt':>8}")
        stats: dict[int, dict[str, float]] = {}
        for w in widths:
            acc: dict[str, list[float]] = {n: [] for n in arms}
            kurts = []
            for k, v in layers:
                b = to_blocks((k, v)[which], w, None)
                _, r = armc(b, bits)
                for n in acc:
                    acc[n].append(r[n])
                kurts.append(excess_kurtosis(b))
            m = {n: float(np.mean(acc[n])) for n in acc}
            stats[w] = m
            print(
                f"{w:>6}{bits + 16.0 / w:>8.3f}"
                + "".join(f"{m[n]:>10.4%}" for n in arms)
                + f"{np.mean(kurts):>8.2f}"
            )
        # The in-kernel arms are what the plan is choosing against, so price them
        # properly: of the log-error reduction a full-width append-time rotation
        # buys at this width, how much does an in-kernel one buy for the same
        # dots? A quarter of the gain is the whole reason this question is asked.
        print(f"{'':>6}{'append':>10}{'SO(4)':>10}{'pair':>10}  (share of the log-error gain)")
        for w in widths:
            m = stats[w]
            total = np.log(m["AFF"] / m["had"])
            print(
                f"{w:>6}{'100%':>10}"
                + "".join(f"{np.log(m['AFF'] / m[n]) / total:>9.0%}" for n in ("quat4", "givens2"))
            )

    print(f"\n-- per layer at {width}-wide blocks, {bits + 16.0 / width:.3f} b/w")
    for half, which in (("K", 0), ("V", 1)):
        per = [armc(to_blocks((k, v)[which], width, None), bits) for k, v in layers]
        kurt = np.mean([excess_kurtosis(to_blocks((k, v)[which], width, None)) for k, v in layers])
        print(f"\n{half}: mean excess kurtosis {kurt:+.2f}")
        names = ("AFF", "had", "hadD", "quat4", "givens2")
        print(f"{'layer':<8}" + "".join(f"{n:>10}" for n in names))
        for i, (_, r) in enumerate(per):
            print(f"{i:<8}" + "".join(f"{r[n]:>10.4%}" for n in names))
        mean = {n: float(np.mean([r[n] for _, r in per])) for n in names}
        best = min(mean, key=lambda n: mean[n])
        print(f"{'mean':<8}" + "".join(f"{mean[n]:>10.4%}" for n in names))
        print(
            f"{'vs AFF':<8}"
            + "".join(f"{mean['AFF'] / mean[n]:>9.2f}x" for n in names)
            + f"   best {best} ({mean['AFF'] / mean[best]:.2f}x)"
        )


if __name__ == "__main__":
    import sys

    # KV mode is standalone: it needs a dump, not the models the default sweep
    # dequantizes, and it is the P6.6a measurement rather than the arm-B/C weight
    # probe.
    if len(sys.argv) > 2 and sys.argv[1] == "--kv":
        report_kv(sys.argv[2], bits=int(sys.argv[3]) if len(sys.argv) > 3 else 4)
        raise SystemExit(0)

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
