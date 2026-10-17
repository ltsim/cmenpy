"""Adaptive Random Search (Luus–Jaakola style) built with ``cm.template``.

Whole-population parts, GWO style:

* ``perturb``      -> sample every candidate and clip (``@cm.reduce``).
* ``adapt_radius`` -> per-individual radius update (``@cm.vectorize``).

No ``@cm.precompile``. Self-learning: each individual keeps its own search radius
and grows or shrinks it from the success rate of the last window.
"""
import numpy as np

import cmenpy as cm


@cm.template
class AdaptiveRandomSearch:
    r: cm.Argument[float, (1e-4, 1.0), 0.1]
    expand: cm.Argument[float, (1.0, 2.0), 1.2]
    contract: cm.Argument[float, (0.1, 1.0), 0.5]
    window: cm.Argument[int, (10, 200), 50]
    p_target: cm.Argument[float, (0.1, 0.5), 0.2]

    radius: cm.Variable[np.ndarray]
    history: cm.Variable[np.ndarray]
    h_idx: cm.Variable[int]
    steps: cm.Variable[int]

    def initialize(self, ctx) -> None:
        buff, bounds, rng = ctx.buff, ctx.bounds, ctx.rng
        pop = cm.PopulationSwarm(buff)

        pop.compute << rng.uniform(bounds.low, bounds.up, (pop.size, bounds.ndim))

        self.radius = np.full(pop.size, self.r, dtype=float)
        self.window = int(self.window)
        self.history = np.zeros((pop.size, self.window), dtype=bool)
        self.h_idx = 0
        self.steps = 0

    def evolve(self, e, ctx) -> None:
        buff, bounds, rng = ctx.buff, ctx.bounds, ctx.rng
        pop = cm.PopulationSwarm(buff)
        n_pop = cm.PopulationSwarm(buff.snapshot)

        n = pop.size
        X = pop.X[:n]
        F = pop.F[:n]

        Z = rng.standard_normal((n, bounds.ndim))
        candidates = self.perturb(X, Z, self.radius, bounds.low, bounds.up)

        n_pop.compute << candidates
        F_cand = n_pop.F[:n]

        improved = cm.is_best((F_cand, F), ctx.sense)

        self.history[:, self.h_idx] = False
        if len(improved):
            pop.assign[improved] << n_pop.access[improved]
            self.history[improved, self.h_idx] = True

        self.h_idx = (self.h_idx + 1) % self.window
        self.steps += 1

        if self.steps % self.window == 0:
            p = np.mean(self.history, axis=1)
            self.radius = self.adapt_radius(self.radius, p, self.p_target, self.expand, self.contract)

    @cm.reduce
    def perturb(
        row: "Array[float, 'n']",
        z: "Array[float, 'n']",
        r: "Scalar[float]",
        low: "Scalar[float]",
        up: "Scalar[float]",
        out: "Array[float, 'n']",
    ):
        for i in range(row.shape[0]):
            v = row[i] + r * z[i]
            if v < low:
                v = low
            if v > up:
                v = up
            out[i] = v

    @cm.vectorize
    def adapt_radius(
        r: "Scalar[float]",
        p: "Scalar[float]",
        p_target: "Scalar[float]",
        expand: "Scalar[float]",
        contract: "Scalar[float]",
    ) -> "Scalar[float]":
        if p > p_target + 1e-12:
            return r * expand
        if p < p_target - 1e-12:
            return r * contract
        return r


def sphere(x):
    x = np.asarray(x)
    if x.ndim == 2:
        return np.sum(x ** 2, axis=1)
    return np.sum(x ** 2)


if __name__ == "__main__":
    model = AdaptiveRandomSearch(seed=42)
    bounds = cm.Bounds[(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    best_f, best_x = model.solve(sphere, bounds, epochs=500, pop_size=20, pop_range=(20, 20))
    print("Best fitness:", best_f)
    print("Best solution:", best_x)
    print("Kernels:", cm.prepare(AdaptiveRandomSearch))
