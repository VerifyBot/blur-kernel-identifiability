#Summarise results/accuracies.jsonl per arm: mean accuracy over pairs, a 95%
#t-interval (pairs are the clusters), and a sign-flip permutation p-value for
#"mean accuracy > 0.5".  Writes results/results.json and prints a table.
import json
import os

import numpy as np
from scipy import stats

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
N_PERM = 10000
SEED = 20260905


def arm_stats(accs, rng):
    a = np.asarray(accs, dtype=float)
    n = len(a)
    mean, sd = a.mean(), a.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    # permutation test
    dev = a - 0.5
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, n))
    null = (signs * np.abs(dev)).mean(axis=1)
    p = (np.sum(null >= dev.mean() - 1e-15) + 1) / (N_PERM + 1)
    return {"n_pairs": n, "mean_acc": float(mean),
            "ci95": [float(mean - half), float(mean + half)],
            "covers_chance": bool(mean - half <= 0.5 <= mean + half),
            "perm_p": float(p), "accs": [float(x) for x in a]}


def main():
    with open(os.path.join(RESULTS, "accuracies.jsonl")) as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 43]))

    arms = [("matched", "white", 200), ("matched", "white", 800),
            ("matched", "tilt", 200), ("matched", "tilt", 800),
            ("same_kernel", "white", 200), ("control", "white", 200)]
    out = {}
    print(f"{'arm':12s} {'source':7s} {'M':>4s} {'n':>3s} {'mean':>7s}  {'95% CI':18s} {'p':>7s}")
    for arm, spec, M in arms:
        sel = sorted((r for r in recs if r["arm"] == arm and r["spectrum"] == spec and r["M"] == M),
                     key=lambda r: r["pair"])
        if not sel:
            continue
        s = arm_stats([r["test_accuracy"] for r in sel], rng)
        out[f"{arm}__{spec}__M{M}"] = s
        print(f"{arm:12s} {spec:7s} {M:4d} {s['n_pairs']:3d} {s['mean_acc']:7.4f}  "
              f"[{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}] {s['perm_p']:7.4f}")

    with open(os.path.join(RESULTS, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
