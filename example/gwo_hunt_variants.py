"""GWO hunt variants: parallel (baseline) vs reduce (A) vs precompiled (B).

* ``GreyWolfOptimizer`` (baseline, importado de ``gwo_algorithm_vectorized``):
  ``@cm.precompile`` + ``nb.prange`` sobre ``pop``.
* ``GreyWolfGuvec`` (A): ``@cm.reduce`` por lobo, con dims anotadas
  (``'n'`` y ``'m,n'``); el layout ``"(n),(m,n),(m,n),(m,n)->(n)"`` se infiere.
* ``GreyWolfJitVec`` (B): ``@cm.precompile`` con cuerpo 100% numpy vectorizado
  (broadcast + ``np.sum(..., axis=1)``). Nota: ``np.mean(..., axis=1)`` no
  compila en nopython, por eso se usa ``sum / n_leaders``.

Run from the repository root::

    .venv/bin/python -m example.benchmark_hunt_ab
"""

import numpy as np

import cmenpy as cm


def numpy_hunt(X, leaders, A, C):
    """Pure-numpy reference: no numba, whole-population broadcast."""
    diff = np.abs(C * leaders[None, :, :] - X[:, None, :])
    return np.sum(leaders[None, :, :] - A * diff, axis=1) / leaders.shape[0]


@cm.template
class GreyWolfGuvec:
    """GWO with hunt as a per-wolf gufunc (option A)."""

    def initialize(self, ctx) -> None:
        pop = cm.PopulationSwarm(ctx.buff)
        pop.compute << ctx.rng.uniform(
            ctx.bounds.low, ctx.bounds.up, (pop.size, ctx.bounds.ndim)
        )

    def evolve(self, e, ctx) -> None:
        buff, bounds, rng = ctx.buff, ctx.bounds, ctx.rng
        pop = cm.PopulationSwarm(buff)
        n_pop = cm.PopulationSwarm(buff.snapshot)

        leaders = pop.X[pop.sort][:3]
        X_pop = pop.X[: pop.size]

        a = 2.0 - 2.0 * e / ctx.epochs

        rnd = rng.random((pop.size, leaders.shape[0], bounds.ndim))
        A = self.scale(rnd, a)
        C = self.sweep(rnd)

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

    @cm.reduce
    def hunt(
        x: "Array[float, 'n']",
        leaders: "Array[float, 'm,n']",
        A: "Array[float, 'm,n']",
        C: "Array[float, 'm,n']",
        out: "Array[float, 'n']",
    ):
        m = leaders.shape[0]

        for d in range(x.shape[0]):
            acc = 0.0

            for j in range(m):
                D = abs(C[j, d] * leaders[j, d] - x[d])
                acc += leaders[j, d] - A[j, d] * D

            out[d] = acc / m

    @cm.reduce
    def clip(
        row: "Array[float, 'n']",
        low: "Scalar[float]",
        up: "Scalar[float]",
        out: "Array[float, 'n']",
    ):
        for i in range(row.shape[0]):
            out[i] = min(max(row[i], low), up)


@cm.template
class GreyWolfJitVec:
    """GWO with hunt as jit-compiled numpy-vectorized code (option B)."""

    def initialize(self, ctx) -> None:
        pop = cm.PopulationSwarm(ctx.buff)
        pop.compute << ctx.rng.uniform(
            ctx.bounds.low, ctx.bounds.up, (pop.size, ctx.bounds.ndim)
        )

    def evolve(self, e, ctx) -> None:
        buff, bounds, rng = ctx.buff, ctx.bounds, ctx.rng
        pop = cm.PopulationSwarm(buff)
        n_pop = cm.PopulationSwarm(buff.snapshot)

        leaders = pop.X[pop.sort][:3]
        X_pop = pop.X[: pop.size]

        a = 2.0 - 2.0 * e / ctx.epochs

        rnd = rng.random((pop.size, leaders.shape[0], bounds.ndim))
        A = self.scale(rnd, a)
        C = self.sweep(rnd)

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

    @cm.precompile
    def hunt(
        X: "Array[float, 2]",
        leaders: "Array[float, 2]",
        A: "Array[float, 3]",
        C: "Array[float, 3]",
    ):
        diff = np.abs(C * leaders[None, :, :] - X[:, None, :])
        term = leaders[None, :, :] - A * diff

        return np.sum(term, axis=1) / leaders.shape[0]

    @cm.reduce
    def clip(
        row: "Array[float, 'n']",
        low: "Scalar[float]",
        up: "Scalar[float]",
        out: "Array[float, 'n']",
    ):
        for i in range(row.shape[0]):
            out[i] = min(max(row[i], low), up)
