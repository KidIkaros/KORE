"""Does a wider prefill chunk amortise expert bytes? Ask the real routing.

A routed FFN picks its own experts per token, so a chunk of `m` tokens costs the
**union** of its picks per layer instead of `m * n_experts_used` uploads. That
ratio is the naive case for chunked prefill, and it is misleading on its own: the
engine already keeps a slot pool per layer, so at `m = 1` it is already banking
part of the same cross-token overlap. Both mechanisms must be modelled together or
the chunk looks 3.6x better than it is.

Input is a `KORE_MOE_ROUTE_DUMP` file: one line of picked expert ids per routed
FFN call, in call order, no layer index (a token calls the routed FFN once per
routed layer in a fixed order).

  KORE_MOE_ROUTE_DUMP=/tmp/route.txt bonsai <gguf> - "" 8 gpu 4096

The layer period is recovered first by maximising same-set overlap between calls
`k` apart, because routing is layer-specific: consecutive tokens *at the same
layer* correlate, different layers do not. Recovering it is also a check -- on the
35B it lands on 40 (= `n_layers`, so every layer is routed) with 2.92/8 = 36.5%
overlap, reproducing the 36-43% the engine's prefetch probe measures.

The cache model is then calibrated against the engine's own counters, because a
projection that cannot reproduce a measurement is a guess:

  expert bank: 226794 hits, 127446 misses, 46557.5 MB uploaded
  -> 127446 / 3 banks            = 42482 expert misses
  -> 46557.5 MB / 42482          = 1.096 MB per expert (gate+up+down, Q2_K)

so `--calibrate` must reproduce 42482 misses at `chunk=1`, `cap=32` before its
other rows are worth reading.
"""

import argparse
import sys

import numpy as np

# One expert's gate+up+down at the Q2_K family's rate, from the counters above.
E_MB = 1.096


class Lru:
    """Least-recently-used set of at most `cap` experts, most recent first."""

    __slots__ = ("cap", "q")

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.q: list[int] = []

    def touch(self, e: int) -> bool:
        """True on hit; a miss inserts and evicts the least-recently-used."""
        q = self.q
        try:
            i = q.index(e)
        except ValueError:
            if len(q) >= self.cap:
                q.pop()
            q.insert(0, e)
            return False
        del q[i]
        q.insert(0, e)
        return True


def load(path: str, n_experts: int) -> np.ndarray:
    picks = np.array(
        [[int(x) for x in line.split()] for line in open(path) if line.strip()],
        dtype=np.int32,
    )
    bad = picks[picks >= n_experts]
    if bad.size:
        raise SystemExit(f"{path}: expert id {bad.max()} >= n_experts {n_experts}")
    return picks


def layer_period(picks: np.ndarray, max_k: int) -> tuple[int, list[tuple[float, int]]]:
    """The `k` at which same-set overlap peaks. Ties break to the larger `k`,
    whose larger sample is the better estimate."""
    calls = len(picks)
    masks = np.zeros((calls, int(picks.max()) + 1), dtype=bool)
    np.put_along_axis(masks, picks.astype(np.int64), True, axis=1)
    ranked = []
    for k in range(1, min(max_k, calls // 8) + 1):
        a, b = masks[:-k], masks[k:]
        ranked.append((float((a & b).sum(axis=1).mean()), k))
    ranked.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return ranked[0][1], ranked


def union_table(picks: np.ndarray, k: int, n_used: int, widths: list[int]) -> None:
    n_tok = len(picks) // k
    per = picks[: n_tok * k].reshape(n_tok, k, n_used)
    print(f"\n- union over `m` tokens, per layer, with no cache (the naive case):")
    print(f"{'m':>5}{'windows':>10}{'union/token':>13}{'vs m*n_used':>14}{'MB/token':>11}")
    for m in widths:
        if m > n_tok:
            break
        un = []
        for w in range(0, n_tok - m + 1, m):  # disjoint: each chunk runs once
            chips = per[w : w + m]
            for li in range(k):
                un.append(len(np.unique(chips[:, li, :])))
        u = float(np.mean(un))
        print(
            f"{m:>5}{len(un):>10}{u / (m * n_used):>13.3f}"
            f"{m * n_used / u:>13.2f}x{u * E_MB / m:>11.1f}"
        )


def simulate(picks: np.ndarray, k: int, chunk: int, cap: int) -> int:
    """Expert misses for a chunked pass with a `cap`-slot LRU per layer."""
    n_tok = len(picks) // k
    per = picks.reshape(n_tok, k, -1)
    lru = [Lru(cap) for _ in range(k)]
    misses = 0
    for w in range(0, n_tok, chunk):
        block = per[w : w + chunk]
        for li in range(k):
            seen, seq = set(), []
            for row in block[:, li, :]:
                for e in row:
                    e = int(e)
                    if e not in seen:
                        seen.add(e)
                        seq.append(e)
            for e in seq:
                if not lru[li].touch(e):
                    misses += 1
    return misses


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dump", help="KORE_MOE_ROUTE_DUMP file")
    ap.add_argument("--n-experts", type=int, default=256)
    ap.add_argument("--deepest-period", type=int, default=120)
    ap.add_argument("--chunks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--caps", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument(
        "--calibrate",
        metavar="MISSES",
        type=int,
        help="engine's measured expert misses; checked against chunk=1, cap=min(caps)",
    )
    args = ap.parse_args()

    picks = load(args.dump, args.n_experts)
    k, ranked = layer_period(picks, args.deepest_period)
    n_tok = len(picks) // k
    print(
        f"{args.dump}: {len(picks)} calls, {picks.shape[1]} picks/call, "
        f"{args.n_experts} experts"
    )
    print(f"\nlayer period: {k} ({n_tok} tokens if k is right)")
    for ov, kk in ranked[:4]:
        print(f"  k={kk:>3}  same-set overlap {ov:.3f}/{picks.shape[1]}  ({len(picks)/kk:.1f} tokens)")
    if len(picks) % k:
        print(f"  WARNING: {len(picks)} calls do not divide by {k}; tail dropped")

    union_table(picks, k, picks.shape[1], args.chunks)

    base = simulate(picks, k, 1, min(args.caps))
    print(f"\n- chunk x slot-pool, both mechanisms at once (per-layer LRU):")
    print(f"{'chunk':>6}{'cap':>5}{'misses':>9}{'/token':>8}{'MB/token':>10}{'vs m=1':>9}")
    if args.calibrate is not None:
        delta = base - args.calibrate
        verdict = "matches" if abs(delta) <= max(1, args.calibrate // 100) else "MISMATCH"
        print(f"  calibration: model {base} vs engine {args.calibrate}  {verdict}")
        if verdict == "MISMATCH":
            sys.exit(1)
    for cap in args.caps:
        for chunk in args.chunks:
            if chunk > n_tok:
                continue
            m = simulate(picks, k, chunk, cap)
            print(
                f"{chunk:>6}{cap:>5}{m:>9}{m / n_tok:>8.1f}"
                f"{m * E_MB / n_tok:>10.1f}{base / m:>8.2f}x"
            )


if __name__ == "__main__":
    main()
