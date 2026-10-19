"""Differential Evolution (jDE-style self-adaptive) built with ``cm.template``.

The generation is broken into small, whole-population kernels, in the same
spirit as the GWO example (``hunt``/``sweep``/``clip``):

* ``partners``  -> three distinct donors per individual (NumPy index algebra).
* ``resample``  -> jDE per-individual F/CR draws (vectorized).
* ``mutate``    -> DE/rand/1 donor matrix (``@cm.precompile``).
* ``crossover`` -> binomial trial + forced gene (``@cm.precompile``).
* ``clip``      -> row-wise clipping to the bounds (``@cm.reduce``).

No ``@cm.precompile``. Self-learning: successful ``(F_i, CR_i)`` are kept for the
next generation; unsuccessful ones are redrawn around the jDE range.
"""
import numpy as np

import cmenpy as cm


@cm.template
class DifferentialEvolution:
    F0: cm.Argument[float, (0.1, 1.0), 0.5]
    CR0: cm.Argument[float, (0.0, 1.0), 0.9]
    tau1: cm.Argument[float, (0.0, 1.0), 0.1]
    tau2: cm.Argument[float, (0.0, 1.0), 0.1]
    Fl: cm.Argument[float, (0.1, 0.5), 0.1]
    Fu: cm.Argument[float, (0.9, 2.0), 0.9]

    F: cm.Variable[np.ndarray]
    CR: cm.Variable[np.ndarray]

    def initialize(self, ctx) -> None:
        buff, bounds, rng = ctx.buff, ctx.bounds, ctx.rng
        pop = cm.PopulationSwarm(buff)

        pop.compute << rng.uniform(bounds.low, bounds.up, (pop.size, bounds.ndim))

        self.F = np.full(pop.size, self.F0, dtype=float)
        self.CR = np.full(pop.size, self.CR0, dtype=float)

    def evolve(self, e, ctx) -> None:
        buff, bounds, rng = ctx.buff, ctx.bounds, ctx.rng
        pop = cm.PopulationSwarm(buff)
        n_pop = cm.PopulationSwarm(buff.snapshot)

        n = pop.size
        if n < 4:
            return

        X = pop.X[:n]

        # Parts of one generation: donors -> parameters -> trial -> clip -> select.
        r1, r2, r3 = self.partners(n, rng)
        F_new, CR_new = self.resample(self.F, self.CR, rng)

        donors = self.mutate(X[r1], X[r2], X[r3], F_new)
        jrand = rng.integers(0, bounds.ndim, n)
        randn = rng.random((n, bounds.ndim))
        trials = self.crossover(X, donors, CR_new, jrand, randn)

        n_pop.compute << self.clip(trials, bounds.low, bounds.up)

        improved = cm.is_best((n_pop.F, pop.F), ctx.sense)
        if len(improved):
            pop.assign[improved] << n_pop.access[improved]
            # jDE: keep the freshly drawn parameters only where the trial won.
            self.F[improved] = F_new[improved]
            self.CR[improved] = CR_new[improved]

    def partners(self, n, rng):
        """Return three distinct partner indices per row, none equal to self."""
        keys = rng.random((n, n))
        np.fill_diagonal(keys, np.inf)
        order = np.argsort(keys, axis=1)

        return order[:, 0], order[:, 1], order[:, 2]

    def resample(self, F, CR, rng):
        """jDE draws for every individual at once (NumPy, no Python loop)."""
        n = F.shape[0]

        redraw_F = rng.random(n) < self.tau1
        F_new = np.where(redraw_F, self.Fl + rng.random(n) * (self.Fu - self.Fl), F)

        redraw_CR = rng.random(n) < self.tau2
        CR_new = np.where(redraw_CR, rng.random(n), CR)

        return F_new, np.clip(CR_new, 0.0, 1.0)

    @cm.precompile
    def mutate(
        xr1: "Array[float, 2]",
        xr2: "Array[float, 2]",
        xr3: "Array[float, 2]",
        F: "Array[float, 1]",
    ):
        return xr1 + F[:, None] * (xr2 - xr3)

    @cm.precompile
    def crossover(
        x: "Array[float, 2]",
        v: "Array[float, 2]",
        CR: "Array[float, 1]",
        jrand: "Array[int, 1]",
        randn: "Array[float, 2]",
    ):
        cols = np.arange(x.shape[1])
        take = (randn < CR[:, None]) | (cols[None, :] == jrand[:, None])

        return np.where(take, v, x)

    @cm.reduce
    def clip(
        row: "Array[float, 'n']",
        low: "Scalar[float]",
        up: "Scalar[float]",
        out: "Array[float, 'n']",
    ):
        for i in range(row.shape[0]):
            v = row[i]
            if v < low:
                v = low
            if v > up:
                v = up
            out[i] = v


def ackley(x):
    x = np.asarray(x)
    n = x.shape[-1] if x.ndim > 0 else 1
    if x.ndim == 2:
        s1 = np.sum(x ** 2, axis=1)
        s2 = np.sum(np.cos(2.0 * np.pi * x), axis=1)
        return -20.0 * np.exp(-0.2 * np.sqrt(s1 / n)) - np.exp(s2 / n) + 20.0 + np.exp(1.0)
    s1 = np.sum(x ** 2)
    s2 = np.sum(np.cos(2.0 * np.pi * x))
    return -20.0 * np.exp(-0.2 * np.sqrt(s1 / n)) - np.exp(s2 / n) + 20.0 + np.exp(1.0)


if __name__ == "__main__":
    model = DifferentialEvolution(seed=5)
    bounds = cm.Bounds[(-32.0, 32.0), (-32.0, 32.0), (-32.0, 32.0), (-32.0, 32.0), (-32.0, 32.0)]
    best_f, best_x = model.solve(ackley, bounds, epochs=400, pop_size=40, pop_range=(40, 40))
    print("Best fitness:", best_f)
    print("Best solution:", best_x)
    print("Kernels:", cm.prepare(DifferentialEvolution))
