"""LSHADE (self-adaptive) built with ``cm.template``.

Success-history parameter adaptation plus Linear Population Size Reduction
(LPSR). Same decomposed, whole-population kernels as JADE:

* ``sample_params``   -> F/CR drawn per individual from its own memory slot.
* ``pbest``           -> p-best donor index (``p`` shrinks linearly with time).
* ``partners``        -> r1 in the population and r2 in population+archive.
* ``mutate``          -> current-to-pbest/1 matrix (``@cm.precompile``).
* ``crossover``/``clip`` -> binomial trial then row clipping.
* ``lehmer_weighted`` -> weighted success-history mean for the memory.
* ``archive_put``     -> vectorized archive append/replace.
* ``shrink``          -> LPSR by deactivating the worst rows in bulk.

No ``@cm.precompile``.
"""
import numpy as np
from opfunu.name_based import Ackley01

import cmenpy as cm


@cm.template
class LSHADE:
    p_min: cm.Argument[float, (0.01, 0.2), 0.05]
    p_max: cm.Argument[float, (0.1, 0.5), 0.25]
    H: cm.Argument[int, (5, 20), 5]
    arc_rate: cm.Argument[float, (1.0, 5.0), 2.0]

    M_F: cm.Variable[np.ndarray]
    M_CR: cm.Variable[np.ndarray]
    memory_index: cm.Variable[int]
    archive: cm.Variable[np.ndarray]
    archive_size: cm.Variable[int]
    N_init: cm.Variable[int]
    N_min: cm.Variable[int]

    def initialize(self, ctx) -> None:
        buff, bounds, rng = ctx.buff, ctx.bounds, ctx.rng
        pop = cm.PopulationSwarm(buff)

        pop.compute << rng.uniform(bounds.low, bounds.up, (pop.size, bounds.ndim))

        self.N_init = pop.size
        self.N_min = max(4, buff.size.min)
        self.H = int(self.H)
        self.M_F = np.full(self.H, 0.5, dtype=float)
        self.M_CR = np.full(self.H, 0.5, dtype=float)
        self.memory_index = 0
        self.archive = np.zeros((int(self.arc_rate * self.N_init), bounds.ndim), dtype=float)
        self.archive_size = 0

    def evolve(self, e, ctx) -> None:
        buff, bounds, rng = ctx.buff, ctx.bounds, ctx.rng
        pop = cm.PopulationSwarm(buff)

        active = np.array([*~pop])
        n = len(active)
        if n < 4:
            return

        # Physical row -> compacted position, so sort/gather stay consistent.
        pos = np.full(buff.size.max, -1, dtype=int)
        pos[active] = np.arange(n)

        X = cm.cast(pop.X)[active]
        F_cur = cm.cast(pop.F)[active]

        slot = rng.integers(0, self.H, n)
        F_new, CR_new = self.sample_params(self.M_F[slot], self.M_CR[slot], n, rng)

        p = self.p_min + (self.p_max - self.p_min) * (e / ctx.epochs)
        pbest = self.pbest(pop, n, p, rng, pos)

        pool = np.vstack([X, self.archive[: self.archive_size]]) if self.archive_size else X
        r1, r2 = self.partners(n, pool.shape[0], rng)

        donors = self.mutate(X, X[pbest], X[r1], pool[r2], F_new)
        jrand = rng.integers(0, bounds.ndim, n)
        randn = rng.random((n, bounds.ndim))
        trials = self.crossover(X, donors, CR_new, jrand, randn)

        # Evaluate the trials in a snapshot that shares the active mask.
        snap = buff.snapshot
        snap.mask[:] = buff.mask[:]
        n_pop = cm.PopulationSwarm(snap)
        n_pop.compute << self.clip(trials, bounds.low, bounds.up)

        F_off = cm.cast(n_pop.F)[active]
        better = F_off < F_cur if ctx.sense == "min" else F_off > F_cur
        rows = np.flatnonzero(better)

        if len(rows):
            succ = active[rows]
            pop.assign[succ] << n_pop.access[succ]
            self.archive_put(X[rows], rng)

            w = np.abs(F_cur[rows] - F_off[rows])
            self.M_F[self.memory_index], self.M_CR[self.memory_index] = self.lehmer_weighted(
                w, F_new[rows], CR_new[rows]
            )
            self.memory_index = (self.memory_index + 1) % self.H

        target = int(round(self.N_init - (self.N_init - self.N_min) * (e + 1) / ctx.epochs))
        target = max(self.N_min, target)
        if target < n:
            self.shrink(buff, active, target, ctx.sense)

    def sample_params(self, mu_F, mu_CR, n, rng):
        mu_F = np.broadcast_to(np.asarray(mu_F, dtype=float), (n,))
        mu_CR = np.broadcast_to(np.asarray(mu_CR, dtype=float), (n,))

        F = mu_F + 0.1 * rng.standard_cauchy(n)
        bad = F <= 0.0
        while np.any(bad):
            F[bad] = mu_F[bad] + 0.1 * rng.standard_cauchy(int(bad.sum()))
            bad = F <= 0.0

        CR = mu_CR + 0.1 * rng.standard_normal(n)

        return np.minimum(F, 1.0), np.clip(CR, 0.0, 1.0)

    def pbest(self, pop, n, p, rng, pos):
        count = max(1, int(np.ceil(p * n)))
        top = pos[pop.sort[:count]]

        return rng.choice(top, size=n)

    def partners(self, n, pool_size, rng):
        i = np.arange(n)

        r1 = rng.integers(0, n - 1, n)
        r1 = r1 + (r1 >= i)

        k = rng.integers(0, pool_size - 2, n)
        lo = np.minimum(i, r1)
        hi = np.maximum(i, r1)
        r2 = k + (k >= lo) + (k >= hi - 1)

        return r1, r2

    def archive_put(self, x_succ, rng):
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

    def shrink(self, buff, active, target, sense):
        """LPSR: deactivate the worst ``active`` rows down to ``target`` rows."""
        fitness = buff.raw[active, 0]
        order = np.argsort(fitness)
        if sense == "max":
            order = order[::-1]

        drop = active[order[target:]]
        buff.mask[drop] = False
        buff.raw[drop, :] = np.nan

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
    def lehmer_weighted(
        w: "Array[float, 1]",
        sf: "Array[float, 1]",
        scr: "Array[float, 1]",
    ):
        ws = np.sum(w)
        if ws <= 0.0:
            return 0.5, 0.5

        mean_f = np.sum(w * sf * sf) / np.sum(w * sf)
        mean_cr = np.sum(w * scr) / ws

        return mean_f, mean_cr


if __name__ == "__main__":
    model = LSHADE(seed=13)
    f = Ackley01(ndim=10)
    best_f, best_x = model.solve(f.evaluate, f.bounds, epochs=600, pop_size=50, pop_range=(10, 50))
    print("Best fitness:", best_f)
    print("Best solution:", best_x)
    print("Kernels:", cm.prepare(LSHADE))
