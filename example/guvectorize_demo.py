"""Demonstration of the ``cm.guvectorize`` decorator (generalized ufuncs).

``cm.guvectorize`` follows numba's style: the first argument is a list of core
type specs and the second is the layout expression. Inputs are cast according
to their markers; outputs (right of ``->``) are allocated by numba.

Run with::

    .venv/bin/python example/guvectorize_demo.py
"""

import numpy as np

import cmenpy as cm


class RowClip:
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


if __name__ == "__main__":
    matrix = np.array([[-5.0, 0.5, 5.0], [10.0, -10.0, 1.0]], dtype=np.float32)

    print("input:\n", matrix)
    print("clipped:\n", RowClip.clip(matrix, -1.0, 1.0))
    print("kernels:", cm.prepare(RowClip))
