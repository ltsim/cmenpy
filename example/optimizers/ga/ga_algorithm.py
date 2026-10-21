"""Real-coded Genetic Algorithm (self-adaptive) built with ``cm.template``.

Generational operators are whole-population kernels, GWO style:

* ``tournament`` -> vectorized tournament selection for the mating pool.
* ``crossover``  -> BLX-α over every child at once (``@cm.reduce``).
* ``mutate``     -> Gaussian mutation + clip (``@cm.reduce``).
* ``clip``       -> row clipping (``@cm.reduce``).
* ``adapt_sigma``-> 1/5 success-rule update of the mutation strength.

Survival is an elitist ``(mu + lambda)`` step: the next generation keeps the best
``n`` solutions of parents plus offspring, so the best is never lost.

No ``@cm.precompile``.
"""
import numpy as np
from opfunu.name_based import Griewank

import cmenpy as cm


@cm.template
class GeneticAlgorithm:
    pc: cm.Argument[float, (0.5, 1.0), 0.8]
    pm: cm.Argument[float, (0.01, 0.5), 0.1]
    alpha: cm.Argument[float, (0.3, 0.8), 0.5]
    sigma0: cm.Argument[float, (0.01, 0.5), 0.1]
    tsize: cm.Argument[int, (2, 8), 3]
    window: cm.Argument[int, (20, 500), 100]
    p_target: cm.Argument[float, (0.05, 0.3), 0.1]

    sigma: cm.Variable[float]
    success_hist: cm.Variable[np.ndarray]
    h_idx: cm.Variable[int]
    steps: cm.Variable[int]

    def initialize(self, ctx) -> None:
        buff, bounds, rng = ctx.buff, ctx.bounds, ctx.rng
        pop = cm.PopulationSwarm(buff)

        pop.compute << rng.uniform(bounds.low, bounds.up, (pop.size, bounds.ndim))

        self.sigma = float(self.sigma0)
        self.window = int(self.window)
        self.tsize = int(self.tsize)
        self.success_hist = np.zeros(self.window, dtype=float)
        self.h_idx = 0
        self.steps = 0

    def evolve(self, e, ctx) -> None:
        buff, bounds, rng = ctx.buff, ctx.bounds, ctx.rng
        pop = cm.PopulationSwarm(buff)
        n_pop = cm.PopulationSwarm(buff.snapshot)

        n = pop.size
        if n < 2:
            return

        X = pop.X[:n]
        F = pop.F[:n]

        # Parents -> BLX-α crossover -> Gaussian mutation -> clip -> survive.
        p1, p2 = self.tournament(F, n, rng, ctx.sense)

        r = rng.random((n, bounds.ndim))
        crossed = self.crossover(X[p1], X[p2], self.alpha, r)
        pick = np.where((rng.random(n) < 0.5)[:, None], X[p1], X[p2])
        children = np.where((rng.random(n) < self.pc)[:, None], crossed, pick)

        z = rng.standard_normal((n, bounds.ndim))
        mutated = self.mutate(children, z, self.sigma, bounds.low, bounds.up)
        gene_mask = rng.random((n, bounds.ndim)) < self.pm
        kept = np.where(gene_mask, mutated, self.clip(children, bounds.low, bounds.up))

        n_pop.compute << kept
        F_off = n_pop.F[:n]

        # Elitist (mu + lambda): keep the best ``n`` of parents + offspring.
        combined_f = np.concatenate([F, F_off])
        combined_x = np.vstack([X, kept])
        order = np.argsort(combined_f)
        if ctx.sense == "max":
            order = order[::-1]
        keep = order[:n]

        rows = np.empty((n, bounds.ndim + 1))
        rows[:, 0] = combined_f[keep]
        rows[:, 1:] = combined_x[keep]
        pop.assign[np.arange(n)] << rows

        self.success_hist[self.h_idx] = int(np.sum(keep >= n)) / n
        self.h_idx = (self.h_idx + 1) % self.window
        self.steps += 1

        if self.steps % self.window == 0:
            p_succ = float(np.mean(self.success_hist))
            self.sigma = self.adapt_sigma(self.sigma, p_succ, self.p_target)

    def tournament(self, F, n, rng, sense):
        """Select two parents per child with a vectorized tournament."""
        i = np.arange(n)
        cand1 = rng.integers(0, n, (n, self.tsize))
        cand2 = rng.integers(0, n, (n, self.tsize))

        if sense == "min":
            p1 = cand1[i, np.argmin(F[cand1], axis=1)]
            p2 = cand2[i, np.argmin(F[cand2], axis=1)]
        else:
            p1 = cand1[i, np.argmax(F[cand1], axis=1)]
            p2 = cand2[i, np.argmax(F[cand2], axis=1)]

        p2 = np.where(p2 == p1, (p2 + 1) % n, p2)

        return p1, p2

    @cm.reduce
    def crossover(
        a: "Array[float, 'n']",
        b: "Array[float, 'n']",
        alpha: "Scalar[float]",
        r: "Array[float, 'n']",
        out: "Array[float, 'n']",
    ):
        for i in range(a.shape[0]):
            x1 = min(a[i], b[i])
            x2 = max(a[i], b[i])
            span = x2 - x1
            low = x1 - alpha * span
            up = x2 + alpha * span
            out[i] = low + (up - low) * r[i]

    @cm.reduce
    def mutate(
        c: "Array[float, 'n']",
        z: "Array[float, 'n']",
        sigma: "Scalar[float]",
        low: "Scalar[float]",
        up: "Scalar[float]",
        out: "Array[float, 'n']",
    ):
        for i in range(c.shape[0]):
            v = c[i] + sigma * z[i]
            if v < low:
                v = low
            if v > up:
                v = up
            out[i] = v

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

    @cm.vectorize
    def adapt_sigma(
        sigma: "Scalar[float]",
        p_succ: "Scalar[float]",
        p_target: "Scalar[float]",
    ) -> "Scalar[float]":
        if p_succ < p_target:
            sigma_new = sigma * 0.95
        else:
            sigma_new = sigma * 1.05

        if sigma_new < 1e-4:
            sigma_new = 1e-4
        if sigma_new > 0.5:
            sigma_new = 0.5

        return sigma_new


if __name__ == "__main__":
    model = GeneticAlgorithm(seed=7)
    f = Griewank(ndim=5)
    best_f, best_x = model.solve(f.evaluate, f.bounds, epochs=400, pop_size=40, pop_range=(40, 40))
    print("Best fitness:", best_f)
    print("Best solution:", best_x)
    print("Kernels:", cm.prepare(GeneticAlgorithm))
