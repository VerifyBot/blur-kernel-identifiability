"""
The classifier experiment (Section 6 of the report).

For every pair and every run:
  * draw M Gaussian fields per kernel on a 128x128 torus with spectrum S_x,
    blur with the kernel, keep every second pixel  -> 64x64 low-res images
  * compute a fixed feature vector per image (log-periodogram, radial averages,
    intensity quantiles, small triple correlations)
  * train a gradient boosted classifier on the first half, test on the second half

Runs:
  matched pairs   white source   M = 200 and 800
  matched pairs   tilted source  M = 200 and 800
  control pairs   white source   M = 200
  same_kernel     white source   M = 200

Appends one line per run to results/accuracies.jsonl (safe to stop and restart).
Takes about 15 minutes on a laptop.
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernels as kr

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
BANK = os.path.join(RESULTS, "pair_bank.npz")
OUT = os.path.join(RESULTS, "accuracies.jsonl")

SEED = 20260905
N = kr.NGRID
N_RADIAL = 24
QUANTILES = np.linspace(0.05, 0.95, 19)
LAGS = [(a, b) for a in range(-2, 3) for b in range(-2, 3)]   # 25 lag vectors
IU = np.triu_indices(len(LAGS))                               # 325 unordered pairs


# ---------- features ----------

# radial bin of each rfft2 frequency of a 64x64 image (fixed, computed once)
_f1 = np.fft.fftfreq(64)[:, None]
_f2 = np.fft.rfftfreq(64)[None, :]
_r = np.sqrt(_f1 ** 2 + _f2 ** 2)
_rbin = np.minimum((_r / (_r.max() + 1e-12) * N_RADIAL).astype(int), N_RADIAL - 1).ravel()
_rcount = np.bincount(_rbin, minlength=N_RADIAL).astype(float)


def features(y):
    P = np.abs(np.fft.rfft2(y)) ** 2
    logP = np.log(P + 1e-300).ravel()                                   # 64*33 = 2112
    radial = np.bincount(_rbin, weights=logP, minlength=N_RADIAL) / _rcount   # 24
    quant = np.quantile(y, QUANTILES)                                   # 19
    z = y - y.mean()                                                    # 325 triple correlations
    Z = np.stack([np.roll(z, (-a, -b), axis=(0, 1)).ravel() for a, b in LAGS])
    C = (Z * z.ravel()[None, :]) @ Z.T / z.size
    return np.concatenate([logP, radial, quant, C[IU]])


# ---------- synthesis ----------

def transfer(kernel, S):
    """FFT of the combined filter: white noise -> source (sqrt S) -> blur (K)."""
    grid = np.zeros((N, N))
    grid[:kr.KSIZE, :kr.KSIZE] = kernel
    return np.sqrt(S) * np.fft.fft2(grid)


def make_features(T, rng, M):
    X = []
    for _ in range(M):
        w = rng.standard_normal((N, N))
        y = np.fft.ifft2(np.fft.fft2(w) * T).real[::2, ::2]
        X.append(features(y))
    return np.array(X)


# ---------- one run ----------

def run_one(r, pair, kernels, S, spec_code, M):
    n_train = M // 2
    X = []
    for w, slot in enumerate(("k1", "k2")):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 41, r, w, 0, M, spec_code]))
        X.append(make_features(transfer(kernels[pair[slot]], S), rng, M))
    Xtr = np.vstack([X[0][:n_train], X[1][:n_train]])
    ytr = np.r_[np.zeros(n_train), np.ones(n_train)]
    Xte = np.vstack([X[0][n_train:], X[1][n_train:]])
    yte = np.r_[np.zeros(M - n_train), np.ones(M - n_train)]
    clf_seed = int(np.random.SeedSequence([SEED, 42, r, 0, M, spec_code]).generate_state(1)[0] % 2 ** 31)
    clf = HistGradientBoostingClassifier(random_state=clf_seed)
    clf.fit(Xtr, ytr)
    return float((clf.predict(Xte) == yte).mean())


def main():
    with np.load(BANK, allow_pickle=True) as d:
        pairs = json.loads(str(d["pairs"]))
        kernels = {k[2:]: d[k] for k in d.files if k.startswith("K_")}
    spectra = {"white": (0, kr.white_spectrum()), "tilt": (1, kr.tilt_spectrum())}

    runs = []
    for M in (200, 800):
        for spec in ("white", "tilt"):
            for r, p in enumerate(pairs):
                if p["arm"] == "matched":
                    runs.append((r, spec, M))
    for r, p in enumerate(pairs):
        if p["arm"] != "matched":
            runs.append((r, "white", 200))

    done = set()
    if os.path.exists(OUT):
        with open(OUT) as fh:
            done = {json.loads(line)["run"] for line in fh if line.strip()}

    t0 = time.time()
    with open(OUT, "a") as fh:
        for i, (r, spec, M) in enumerate(runs):
            pair = pairs[r]
            name = f"{pair['pair']}__{spec}__M{M}"
            if name in done:
                continue
            code, S = spectra[spec]
            acc = run_one(r, pair, kernels, S, code, M)
            rec = {"run": name, "pair": pair["pair"], "arm": pair["arm"],
                   "spectrum": spec, "M": M, "test_accuracy": acc}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{i + 1}/{len(runs)}] {name:30s} acc={acc:.3f}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
