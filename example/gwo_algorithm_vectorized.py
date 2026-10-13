"""Grey Wolf Optimizer (GWO) built with the ``cm.template`` class API.

This is a class-based copy of ``example/gwo_algorithm.py`` that delegates the
heavy matrix math to numba through the compile decorators:

* ``@cm.vectorize``  -> elementwise ufunc with automatic casting.
* ``@cm.precompile(parallel=True)`` -> ``numba.njit(parallel=True)`` over
  ``nb.prange``; the model enables it with ``GreyWolfOptimizer(parallel=True)``.
* ``@cm.precompile`` -> ``numba.njit`` for plain matrix algebra.

The ``Array``/``Scalar`` annotations tell the decorators how to cast and type
each argument (``numpy.asarray`` / ``float``) before calling the compiled kernel.
"""

import numba as nb
import numpy as np
from opfunu.name_based import Ackley01

import cmenpy as cm


@cm.template
class GreyWolfOptimizer:
    def initialize(self, ctx) -> None:
        buff, bounds, rng = ctx.buff, ctx.bounds, ctx.rng
        pop = cm.PopulationSwarm(buff)

        pop.compute << rng.uniform(bounds.low, bounds.up, (pop.size, bounds.ndim))

    def evolve(self, e, ctx) -> None:
        buff, bounds, rng = ctx.buff, ctx.bounds, ctx.rng
        pop = cm.PopulationSwarm(buff)
        n_pop = cm.PopulationSwarm(buff.snapshot)

        # Alpha, beta and delta wolves.
        leaders = pop.X[pop.sort][:3]
        X_pop = pop.X[: pop.size]

        # Linearly decays from 2 to 0 across the run.
        a = 2.0 - 2.0 * e / ctx.epochs

        # Coefficients A = a(2r - 1) and C = 2r, vectorized with broadcasting.
        rnd = rng.random((pop.size, leaders.shape[0], bounds.ndim))
        A = self.scale(rnd, a)
        C = self.sweep(rnd)

        # Position update, parallel over the whole population.
        X = self.hunt(X_pop, leaders, A, C)

        n_pop.compute << self.clip(X, bounds.low, bounds.up)

        mask = cm.is_best((n_pop.F, pop.F), ctx.sense)
        pop.assign[mask] << n_pop.access[mask]

    @cm.vectorize
    def scale(r: "Scalar[float]", a: "Scalar[float]") -> "Scalar[float]":
        return a * (2.0 * r - 1.0)

    @cm.vectorize
    def sweep(r: "Scalar[float]") -> "Scalar[float]":
        return 2.0 * r

    @cm.precompile(parallel=True)
    def hunt(
        X: "Array[float, 2]",
        leaders: "Array[float, 2]",
        A: "Array[float, 3]",
        C: "Array[float, 3]",
    ):
        out = np.empty_like(X)

        for i in nb.prange(X.shape[0]):
            acc = np.zeros(X.shape[1])

            for j in range(leaders.shape[0]):
                D = np.abs(C[i, j] * leaders[j] - X[i])
                acc += leaders[j] - A[i, j] * D

            out[i] = acc / leaders.shape[0]

        return out

    @cm.reduce
    def clip(
        row: "Array[float, 'n']",
        low: "Scalar[float]",
        up: "Scalar[float]",
        out: "Array[float, 'n']",
    ):
        for i in range(row.shape[0]):
            out[i] = min(max(row[i], low), up)


if __name__ == "__main__":
    model = GreyWolfOptimizer(parallel=True)
    f = Ackley01(ndim=30)

    best_pop = model.solve(f.evaluate, f.bounds, 150, 75)
    print("Best population:", best_pop)
