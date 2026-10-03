"""Benchmark: numpy vs ``@cm.jit`` vs ``@cm.guvectorize`` row-wise clipping.

Run from the repository root::

    .venv/bin/python -m example.benchmark_guvectorize

All three kernels compute the same row-wise clip of a matrix; the numba kernels
are precompiled (typed markers), so timings exclude compilation.
"""

import statistics
import time

import numpy as np

import cmenpy as cm

ROWS = 8192
COLS = 32
REPEATS = 50


class Kernels:
    @cm.guvectorize(
        [
            cm.Array[float, 1],
            cm.Scalar[float],
            cm.Scalar[float],
            cm.Array[float, 1],
        ],
        "(n),(),()->(n)",
    )
    def clip(row, low, up, out):
        for i in range(row.shape[0]):
            out[i] = min(max(row[i], low), up)

    @cm.jit(cm.Array[float, 2], cm.Scalar[float], cm.Scalar[float])
    def clip_jit(X, low, up):
        out = np.empty_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                out[i, j] = min(max(X[i, j], low), up)

        return out


def numpy_clip(X, low, up):
    return np.clip(X, low, up)


def bench(func, X, low, up, repeats):
    func(X, low, up)  # warm up

    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        func(X, low, up)
        times.append(time.perf_counter() - start)

    return statistics.mean(times)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.uniform(-5.0, 5.0, (ROWS, COLS))
    low, up = -1.0, 1.0

    reference = numpy_clip(X, low, up)
    np.testing.assert_allclose(Kernels.clip(X, low, up), reference)
    np.testing.assert_allclose(Kernels.clip_jit(X, low, up), reference)

    print(f"matrix {ROWS}x{COLS} ({ROWS * COLS} elements)  repeats={REPEATS}\n")

    for name, func in (
        ("numpy", numpy_clip),
        ("cm.jit", Kernels.clip_jit),
        ("cm.guvectorize", Kernels.clip),
    ):
        elapsed = bench(func, X, low, up, REPEATS)
        throughput = (ROWS * COLS) / elapsed / 1e6

        print(f"{name:<14} {elapsed * 1e3:8.3f} ms   {throughput:6.1f} Melem/s")
