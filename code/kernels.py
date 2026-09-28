from fractions import Fraction
import numpy as np

KSIZE = 17   
NGRID = 128

def pmul(p, q):
    """convolution of two kernels"""
    out = {}
    for (i1, j1), c1 in p.items():
        for (i2, j2), c2 in q.items():
            key = (i1 + i2, j1 + j2)
            out[key] = out.get(key, 0) + c1 * c2
    return {k: v for k, v in out.items() if v != 0}


def pflip(p):
    """k(-x)"""
    return {(-i, -j): v for (i, j), v in p.items()}


def pacorr(p):
    """autocorrelation"""
    return pmul(p, pflip(p))


def psum(p):
    return sum(p.values(), Fraction(0))


def poly_to_frame(p, size=KSIZE):
    i0 = min(i for i, _ in p); i1 = max(i for i, _ in p)
    j0 = min(j for _, j in p); j1 = max(j for _, j in p)
    h, w = i1 - i0 + 1, j1 - j0 + 1
    assert h <= size and w <= size
    oi = (size - h) // 2 - i0
    oj = (size - w) // 2 - j0
    out = np.zeros((size, size))
    for (i, j), v in p.items():
        out[i + oi, j + oj] = float(v)
    return out


def frame_to_poly(F):
    return {(int(i), int(j)): Fraction(float(F[i, j])) for i, j in zip(*F.nonzero())}


def white_spectrum(n=NGRID):
    return np.ones((n, n))

def tilt_spectrum(n=NGRID, c=0.8):
    w = 2.0 * np.pi * np.fft.fftfreq(n)
    _, w2 = np.meshgrid(w, w, indexing="ij")
    return 1.0 + c * np.cos(w2)


def kernel_power(kernel, n=NGRID):
    grid = np.zeros((n, n))
    grid[:KSIZE, :KSIZE] = kernel
    K = np.fft.fft2(grid)
    return K.real ** 2 + K.imag ** 2


def fold_s2(A):
    n = A.shape[0]
    m = n // 2
    return A.reshape(2, m, 2, m).sum(axis=(0, 2))


def lowres_spectrum(kernel, S):
    """power spectrum of the low-res image for kernel k and source spectrum S"""
    return fold_s2(kernel_power(kernel) * S)


def fold_gap(k1, k2, S):
    """max dif between the two low-res spectra && the scale to compare it to"""
    f1 = lowres_spectrum(k1, S)
    f2 = lowres_spectrum(k2, S)
    return float(np.abs(f1 - f2).max()), float(np.abs(f1).max())
