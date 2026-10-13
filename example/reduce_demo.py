"""Demonstration of the ``cm.reduce`` decorator (generalized ufuncs).

The layout is inferred from the parameter annotations: ``Array[dtype, "dims"]``
names dimensions, ``Scalar[dtype]`` is a scalar core dimension, and the last
parameter is the output (allocated by numba). Pass ``only`` annotations; the
signature can also be given explicitly with ``cm.Contract[[...], ...]``.

Run with::

    .venv/bin/python example/reduce_demo.py
"""

import numpy as np

import cmenpy as cm


class RowClip:
    @cm.reduce  # (n),(),()->(n)
    def clip(
        row: "Array[float, 'n']",
        low: "Scalar[float]",
        up: "Scalar[float]",
        out: "Array[float, 'n']",
    ):
        for i in range(row.shape[0]):
            out[i] = min(max(row[i], low), up)


if __name__ == "__main__":
    matrix = np.array([[-5.0, 0.5, 5.0], [10.0, -10.0, 1.0]], dtype=np.float32)

    print("input:\n", matrix)
    print("clipped:\n", RowClip.clip(matrix, -1.0, 1.0))
    print("kernels:", cm.prepare(RowClip))
