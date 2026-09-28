import json
import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernels as kr

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "pair_bank.npz")

SEED = 20260905
SCALE = 2 ** 16
SUM_G, SUM_V = 1024, 64
MIN_EFFECT = 0.20 

G_SITES = [(2 * a, 2 * b) for a in range(8) for b in range(8)]
V_SITES = [(a, b) for a in range(3) for b in range(3)]
ARM_CODE = {"matched": 1, "control": 2, "same_kernel": 3}
N_PAIRS = {"matched": 20, "control": 10, "same_kernel": 10}


def draw_G(rng):
    g = {p: 1 for p in G_SITES}
    i0, j0 = int(rng.integers(0, 7)), int(rng.integers(0, 7))
    block = [(2 * (i0 + a), 2 * (j0 + b)) for a in range(2) for b in range(2)]
    for p, extra in zip(block, rng.multinomial(SUM_G - len(G_SITES), [0.25] * 4)):
        g[p] += int(extra)
    assert sum(g.values()) == SUM_G
    return {p: Fraction(v) for p, v in g.items()}


def draw_v(rng):
    x = [int(t) for t in rng.integers(1, 25, size=9)]
    d = SUM_V - sum(x)
    while d > 0:
        x[int(rng.integers(0, 9))] += 1
        d -= 1
    while d < 0:
        i = int(rng.integers(0, 9))
        if x[i] > 1:
            x[i] -= 1
            d += 1
    assert sum(x) == SUM_V and min(x) >= 1
    return {p: Fraction(t) for p, t in zip(V_SITES, x)}


def make_kernel(G, V):
    k = kr.pmul(G, V)
    assert kr.psum(k) == SUM_G * SUM_V
    k = {p: v / (SUM_G * SUM_V) for p, v in k.items()}
    return kr.poly_to_frame(k)

def acorr_difference(F1, F2):
    """acorr(k1) - acorr(k2)"""
    a1 = kr.pacorr(kr.frame_to_poly(F1))
    a2 = kr.pacorr(kr.frame_to_poly(F2))
    diff = {}
    for lag in set(a1) | set(a2):
        d = a1.get(lag, 0) - a2.get(lag, 0)
        if d != 0:
            diff[lag] = d
    even_even = [lag for lag in diff if lag[0] % 2 == 0 and lag[1] % 2 == 0]
    return diff, even_even


def is_shift(F1, F2):
    k1, k2 = kr.frame_to_poly(F1), kr.frame_to_poly(F2)
    for di in range(-kr.KSIZE, kr.KSIZE + 1):
        for dj in range(-kr.KSIZE, kr.KSIZE + 1):
            if {(i + di, j + dj): v for (i, j), v in k1.items()} == k2:
                return True
    return False


def admissible(F):
    k = kr.frame_to_poly(F)
    return len(k) == kr.KSIZE ** 2 and min(k.values()) > 0 and sum(k.values()) == 1 \
        and float(F.sum()) == 1.0


def log_spectral_distance(F1, F2, S):
    """RMS distance between the log low-res spectra"""
    f1 = kr.lowres_spectrum(F1, S)
    f2 = kr.lowres_spectrum(F2, S)
    d = np.log(f1 + 1e-300) - np.log(f2 + 1e-300)
    return float(np.sqrt((d ** 2).mean()))


def build_pair(arm, idx, S_white, S_tilt):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 10, ARM_CODE[arm], idx]))
    for attempt in range(20000):
        G = draw_G(rng)
        V1 = draw_v(rng)
        if arm == "matched":
            if V1[(0, 1)] == V1[(2, 1)] or V1[(1, 0)] == V1[(1, 2)]:
                continue
            V2 = dict(V1)
            V2[(0, 1)], V2[(2, 1)] = V1[(2, 1)], V1[(0, 1)]
        elif arm == "control":
            V2 = draw_v(rng)
        else:
            V2 = dict(V1)

        F1, F2 = make_kernel(G, V1), make_kernel(G, V2)
        diff, even_even = acorr_difference(F1, F2)
        if arm == "matched":
            if even_even or not diff:
                continue
        elif arm == "control":
            if not even_even:
                continue
        else:
            assert not diff
        if not (admissible(F1) and admissible(F2)):
            continue
        if arm != "same_kernel" and is_shift(F1, F2):
            continue

        d_white = log_spectral_distance(F1, F2, S_white)
        d_tilt = log_spectral_distance(F1, F2, S_tilt)
        if arm == "matched" and d_tilt < MIN_EFFECT:
            continue
        if arm == "control" and d_white < MIN_EFFECT:
            continue

        info = {"pair": f"{arm}_{idx:02d}", "arm": arm, "attempt": attempt,
                "G": {f"{p[0]},{p[1]}": int(v) for p, v in sorted(G.items())},
                "v1": [[int(V1[(i, j)]) for j in range(3)] for i in range(3)],
                "v2": [[int(V2[(i, j)]) for j in range(3)] for i in range(3)],
                "n_nonzero_lags": len(diff),
                "n_even_even_lags": len(even_even),
                "logspec_dist_white": d_white,
                "logspec_dist_tilt": d_tilt}
        return info, F1, F2
    raise#cannot do


def main():
    S_white, S_tilt = kr.white_spectrum(), kr.tilt_spectrum()
    pairs, kernels = [], {}
    for arm in ("matched", "control", "same_kernel"):
        for idx in range(N_PAIRS[arm]):
            info, F1, F2 = build_pair(arm, idx, S_white, S_tilt)
            info["k1"], info["k2"] = info["pair"] + "_k1", info["pair"] + "_k2"
            kernels[info["k1"]], kernels[info["k2"]] = F1, F2
            pairs.append(info)
            print(f"{info['pair']:15s} nonzero lags={info['n_nonzero_lags']:3d} "
                  f"even-even={info['n_even_even_lags']:3d} "
                  f"dist white={info['logspec_dist_white']:.2e} "
                  f"tilt={info['logspec_dist_tilt']:.2e}")

    np.savez_compressed(OUT, pairs=json.dumps(pairs),
                        **{"K_" + name: F for name, F in kernels.items()})
    print(f"wrote {len(pairs)} pairs to {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
