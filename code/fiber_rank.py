#Rank of the observation map's Jacobian (Section 5 of the report).
#F(k) = T_S( acorr(k) ) maps a 17x17 kernel to the low-res power spectrum.
#We evaluate its derivative at a random strictly positive integer kernel,
#restrict it to the tangent space V = { sum delta = 0 } (dimension 288), and
#compute the rank.The number of invisible directions is 288.

import json
import os
import random
import sys

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
PRIMES = [2147483647, 2000000011, 1000000007]
R = 8                     # kernel is 17x17 = {-8..8}^2
LAGBOX = 2 * R            # autocorrelation

BOX = [(i, j) for i in range(-R, R + 1) for j in range(-R, R + 1)]
ALIASES = [(0, 0), (1, 0), (0, 1), (1, 1)]


def sym_lags(rad):
    """per {l, -l} in {-rad..rad}^2"""
    return [(0, 0)] + [(i, j) for i in range(-rad, rad + 1) for j in range(-rad, rad + 1)
                       if (i, j) > (0, 0)]


SYM = sym_lags(LAGBOX)
SYM_IDX = {l: i for i, l in enumerate(SYM)}


def parity(l):
    return (l[0] % 2, l[1] % 2)

def pmul(X, Y):
    out = {}
    for lx, cx in X.items():
        for ly, cy in Y.items():
            l = (lx[0] + ly[0], lx[1] + ly[1])
            out[l] = out.get(l, 0) + cx * cy
    return {l: c for l, c in out.items() if c}


def shift_by_pi(X, a):
    """the coefficient at lag l pick (-1)^{<l,a>}"""
    return {l: (c if (l[0] * a[0] + l[1] * a[1]) % 2 == 0 else -c) for l, c in X.items()}

def jacobian_of_acorr(k):
    return [[k.get((n[0] + l[0], n[1] + l[1]), 0) + k.get((n[0] - l[0], n[1] - l[1]), 0)
             for n in BOX] for l in SYM]


def restrict_to_V(M):
    j0 = BOX.index((0, 0))
    return [[row[j] - row[j0] for j in range(len(BOX)) if j != j0] for row in M]


def fold_matrix(D, q):
    """integer matrix of the fold map T for S = 1/D"""
    Ds = {a: shift_by_pi(D, a) for a in ALIASES}
    prod_except = {}
    for a in ALIASES:
        P = {(0, 0): 1}
        for a2 in ALIASES:
            if a2 != a:
                P = pmul(P, Ds[a2])
        prod_except[a] = P
    E = {}
    for b in ALIASES:
        acc = {}
        for a in ALIASES:
            s = -1 if (a[0] * b[0] + a[1] * b[1]) % 2 else 1
            for l, c in prod_except[a].items():
                acc[l] = acc.get(l, 0) + s * c
        E[b] = {l: c for l, c in acc.items() if c}
    rows = sym_lags(LAGBOX + 3 * q)
    row_idx = {l: i for i, l in enumerate(rows)}
    T = [[0] * len(SYM) for _ in rows]
    for c, l in enumerate(SYM):
        terms = [l] if l == (0, 0) else [l, (-l[0], -l[1])]
        for s in terms:
            for r, coef in E[parity(l)].items():
                t = (r[0] + s[0], r[1] + s[1])
                if t in row_idx:
                    T[row_idx[t]][c] += coef
    return T


def random_D(q, rng):
    """even trig polynomial"""
    D = {}
    for l in sym_lags(q):
        if l == (0, 0):
            continue
        c = rng.choice([x for x in range(-5, 6) if x != 0])
        D[l] = c
        D[(-l[0], -l[1])] = c
    D[(0, 0)] = 1 + sum(abs(c) for c in D.values())    # triangle inequality => D >= 1
    return D

def rank_mod_p(M, p):
    A = np.array(M, dtype=np.int64) % p
    nr, nc = A.shape
    r = 0
    for c in range(nc):
        nz = np.nonzero(A[r:, c])[0]
        if nz.size == 0:
            continue
        piv = r + int(nz[0])
        A[[r, piv]] = A[[piv, r]]
        A[r] = (A[r] * pow(int(A[r, c]), p - 2, p)) % p
        below = A[r + 1:, c].copy()
        A[r + 1:] = (A[r + 1:] - np.outer(below, A[r])) % p
        r += 1
        if r == nr:
            break
    return r


def rank_over_Q(M):
    from sympy import ZZ
    from sympy.polys.matrices import DomainMatrix
    return DomainMatrix([[ZZ(int(x)) for x in row] for row in M], (len(M), len(M[0])), ZZ).rank()


def matmul(T, L):
    T, L = np.array(T, dtype=object), np.array(L, dtype=object)   # python ints, no overflow
    return (T @ L).tolist()


def main():
    exact = "--exact" in sys.argv
    seed = 20260824
    rng = random.Random(seed)
    k = {m: rng.randint(1, 100) for m in BOX}
    L = restrict_to_V(jacobian_of_acorr(k))

    counts = {}
    for l in SYM:
        counts[parity(l)] = counts.get(parity(l), 0) + 1
    print("symmetric lags by parity class:", counts)

    out = {"seed": seed, "primes": PRIMES, "parity_counts": {f"{a}{b}": c for (a, b), c in counts.items()},
           "dim_V": 288, "ranks": {}}
    for q in (0, 1, 2, 3, 4):
        D = {(0, 0): 1} if q == 0 else random_D(q, random.Random(seed + q))
        J = matmul(fold_matrix(D, q), L)
        ranks = {p: rank_mod_p(J, p) for p in PRIMES}
        rec = {"rank_mod_p": {str(p): r for p, r in ranks.items()},
               "invisible_directions": 288 - max(ranks.values())}
        if exact:
            rec["rank_over_Q"] = rank_over_Q(J)
            rec["invisible_directions"] = 288 - rec["rank_over_Q"]
        name = "white" if q == 0 else f"q={q}"
        print(f"{name:6s} rank mod p = {sorted(set(ranks.values()))}"
              + (f", over Q = {rec['rank_over_Q']}" if exact else "")
              + f"  -> {rec['invisible_directions']} invisible directions", flush=True)
        out["ranks"][name] = rec

    with open(os.path.join(RESULTS, "fiber_rank.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
