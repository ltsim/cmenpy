"""JADE (self-adaptive) built with ``cm.template``.

current-to-pbest/1 with an external archive and adaptive control parameters.
The generation is decomposed into whole-population parts, GWO style:

* ``sample_params`` -> F ~ Cauchy(mu_F, .1), CR ~ Normal(mu_CR, .1), clipped.
* ``pbest``         -> per-individual p-best donor index.
* ``partners``      -> r1 from the population and r2 from population+archive.
* ``mutate``        -> current-to-pbest/1 donor matrix (``@cm.precompile``).
* ``crossover``     -> binomial trial (``@cm.precompile``).
* ``clip``          -> row clipping (``@cm.reduce``).
* ``archive_put``   -> vectorized archive append/replace.
* ``lehmer``        -> success-history mean used to adapt ``mu_F``/``mu_CR``.

No ``@cm.precompile``.
"""
import numpy as np

import cmenpy as cm


@cm.template
class JADE:
    p: cm.Argument[float, (0.05, 0.5), 0.05]
    c: cm.Argument[float, (0.0, 0.5), 0.1]
    archive_max: cm.Argument[int, (10, 200), 50]

    mu_F: cm.Variable[float]
    mu_CR: cm.Variable[float]
    archive: cm.Variable[np.ndarray]
    archive_size: cm.Variable[int]

    def initialize(self, ctx) -> None:
        buff, bounds, rng = ctx.buff, ctx.bounds, ctx.rng
        pop = cm.PopulationSwarm(buff)

        pop.compute << rng.uniform(bounds.low, bounds.up, (pop.size, bounds.ndim))

        self.mu_F = 0.5
        self.mu_CR = 0.5
        self.archive_max = int(self.archive_max)
        self.archive = np.zeros((self.archive_max, bounds.ndim), dtype=float)
        self.archive_size = 0

    def evolve(self, e, ctx) -> None:
        buff, bounds, rng = ctx.buff, ctx.bounds, ctx.rng
        pop = cm.PopulationSwarm(buff)
        n_pop = cm.PopulationSwarm(buff.snapshot)

        n = pop.size
        if n < 4:
            return

        X = pop.X[:n]

        F_new, CR_new = self.sample_params(self.mu_F, self.mu_CR, n, rng)
        pbest = self.pbest(pop, n, self.p, rng)

        pool = np.vstack([X, self.archive[: self.archive_size]]) if self.archive_size else X
        r1, r2 = self.partners(n, pool.shape[0], rng)

        donors = self.mutate(X, X[pbest], X[r1], pool[r2], F_new)
        jrand = rng.integers(0, bounds.ndim, n)
        randn = rng.random((n, bounds.ndim))
        trials = self.crossover(X, donors, CR_new, jrand, randn)

        n_pop.compute << self.clip(trials, bounds.low, bounds.up)

        improved = cm.is_best((n_pop.F, pop.F), ctx.sense)
        if len(improved):
            pop.assign[improved] << n_pop.access[improved]

            self.archive_put(X[improved], rng)

            S_F = F_new[improved]
            S_CR = CR_new[improved]
            self.mu_F = (1.0 - self.c) * self.mu_F + self.c * self.lehmer(S_F)
            self.mu_CR = (1.0 - self.c) * self.mu_CR + self.c * float(np.mean(S_CR))

    def sample_params(self, mu_F, mu_CR, n, rng):
        """Vectorized Cauchy/normal draws around the running means."""
        mu_F = np.broadcast_to(np.asarray(mu_F, dtype=float), (n,))
        mu_CR = np.broadcast_to(np.asarray(mu_CR, dtype=float), (n,))

        F = mu_F + 0.1 * rng.standard_cauchy(n)
        bad = F <= 0.0
        while np.any(bad):
            F[bad] = mu_F[bad] + 0.1 * rng.standard_cauchy(int(bad.sum()))
            bad = F <= 0.0

        CR = mu_CR + 0.1 * rng.standard_normal(n)

        return np.minimum(F, 1.0), np.clip(CR, 0.0, 1.0)

    def pbest(self, pop, n, p, rng):
        """Draw one p-best parent per individual (vectorized gather)."""
        count = max(1, int(np.ceil(p * n)))

        return rng.choice(pop.sort[:count], size=n)

    def partners(self, n, pool_size, rng):
        """r1 in the population and r2 in population+archive, excluding i and r1."""
        i = np.arange(n)

        r1 = rng.integers(0, n - 1, n)
        r1 = r1 + (r1 >= i)

        k = rng.integers(0, pool_size - 2, n)
        lo = np.minimum(i, r1)
        hi = np.maximum(i, r1)
        r2 = k + (k >= lo) + (k >= hi - 1)

        return r1, r2

    def archive_put(self, x_succ, rng):
        """Append successful parents; replace random slots once full."""
        cap = self.archive.shape[0]
        k = x_succ.shape[0]
        if k == 0 or cap == 0:
            return

        take = min(cap - self.archive_size, k)
        if take > 0:
            self.archive[self.archive_size : self.archive_size + take] = x_succ[:take]
            self.archive_size += take

        overflow = k - take
        if overflow > 0:
            self.archive[rng.integers(0, cap, overflow)] = x_succ[take:]

    @cm.precompile
    def mutate(
        x: "Array[float, 2]",
        xpbest: "Array[float, 2]",
        xr1: "Array[float, 2]",
        xr2: "Array[float, 2]",
        F: "Array[float, 1]",
    ):
        return x + F[:, None] * (xpbest - x) + F[:, None] * (xr1 - xr2)

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

    @cm.precompile
    def lehmer(sf: "Array[float, 1]"):
        den = np.sum(sf)
        if den <= 0.0:
            return 0.0

        return np.sum(sf * sf) / den


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
    model = JADE(seed=11)
    bounds = cm.Bounds[(-32.0, 32.0), (-32.0, 32.0), (-32.0, 32.0), (-32.0, 32.0), (-32.0, 32.0)]
    best_f, best_x = model.solve(ackley, bounds, epochs=500, pop_size=50, pop_range=(50, 50))
    print("Best fitness:", best_f)
    print("Best solution:", best_x)
    print("Kernels:", cm.prepare(JADE))
