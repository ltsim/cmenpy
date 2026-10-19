"""Simulated Annealing (self-adaptive) built with ``cm.template``.

Run as a population of independent chains so every step is a matrix operation,
GWO style:

* ``neighbor``    -> perturb every chain at once (``@cm.reduce``).
* ``metropolis``  -> vectorized acceptance test (``@cm.precompile``).
* ``cool``        -> per-chain temperature update toward a target accept rate.
* ``adapt_step``  -> per-chain step-size update (``@cm.vectorize``).

No ``@cm.precompile``. Self-learning: each chain reshapes its own temperature and
step size from its recent acceptance ratio instead of a fixed schedule.
"""
import numpy as np

import cmenpy as cm


@cm.template
class SimulatedAnnealing:
    T0: cm.Argument[float, (1.0, 100.0), 30.0]
    alpha: cm.Argument[float, (0.8, 0.999), 0.98]
    step0: cm.Argument[float, (0.01, 1.0), 0.7]
    target_accept: cm.Argument[float, (0.1, 0.5), 0.3]
    window: cm.Argument[int, (10, 200), 50]

    T: cm.Variable[np.ndarray]
    step: cm.Variable[np.ndarray]
    accept_hist: cm.Variable[np.ndarray]
    h_idx: cm.Variable[int]
    steps: cm.Variable[int]
    best_f: cm.Variable[float]
    best_x: cm.Variable[np.ndarray]

    def initialize(self, ctx) -> None:
        buff, bounds, rng = ctx.buff, ctx.bounds, ctx.rng
        pop = cm.PopulationSwarm(buff)

        pop.compute << rng.uniform(bounds.low, bounds.up, (pop.size, bounds.ndim))

        self.window = int(self.window)
        self.T = np.full(pop.size, self.T0, dtype=float)
        self.step = np.full(pop.size, self.step0, dtype=float)
        self.accept_hist = np.zeros((pop.size, self.window), dtype=bool)
        self.h_idx = 0
        self.steps = 0
        # SA accepts worsening moves, so the best-ever solution is kept aside.
        self.best_f = float(pop.F[pop.best])
        self.best_x = pop.X[pop.best].copy()

    def evolve(self, e, ctx) -> None:
        buff, bounds, rng = ctx.buff, ctx.bounds, ctx.rng
        pop = cm.PopulationSwarm(buff)
        n_pop = cm.PopulationSwarm(buff.snapshot)

        n = pop.size
        if n < 1:
            return

        X = pop.X[:n]
        F_cur = pop.F[:n]

        Z = rng.standard_normal((n, bounds.ndim))
        candidates = self.neighbor(X, Z, self.step, bounds.low, bounds.up)

        n_pop.compute << candidates
        F_off = n_pop.F[:n]

        delta = F_off - F_cur if ctx.sense == "min" else F_cur - F_off
        accepted = self.metropolis(delta, self.T, rng.random(n))

        rows = np.flatnonzero(accepted)
        if len(rows):
            pop.assign[rows] << n_pop.access[rows]

        # Keep the best-ever solution visible to ``solve`` (which reads the buffer).
        best_idx = int(pop.best)
        best_f = float(pop.F[best_idx])
        improved = best_f < self.best_f if ctx.sense == "min" else best_f > self.best_f

        if improved:
            self.best_f = best_f
            self.best_x = pop.X[best_idx].copy()
        else:
            worst = int(pop.worst)
            pop.assign[worst] << np.concatenate(([self.best_f], self.best_x))

        self.accept_hist[:, self.h_idx] = accepted
        self.h_idx = (self.h_idx + 1) % self.window
        self.steps += 1

        if self.steps % self.window == 0:
            p_acc = np.mean(self.accept_hist, axis=1)
            self.T = self.cool(self.T, self.alpha, p_acc, self.target_accept)
            self.step = self.adapt_step(self.step, p_acc, self.target_accept)

    @cm.reduce
    def neighbor(
        row: "Array[float, 'n']",
        z: "Array[float, 'n']",
        step: "Scalar[float]",
        low: "Scalar[float]",
        up: "Scalar[float]",
        out: "Array[float, 'n']",
    ):
        for i in range(row.shape[0]):
            v = row[i] + step * z[i]
            if v < low:
                v = low
            if v > up:
                v = up
            out[i] = v

    @cm.precompile
    def metropolis(
        delta: "Array[float, 1]",
        T: "Array[float, 1]",
        u: "Array[float, 1]",
    ):
        return u < np.exp(-delta / T)

    @cm.vectorize
    def cool(
        T: "Scalar[float]",
        alpha: "Scalar[float]",
        p_acc: "Scalar[float]",
        target: "Scalar[float]",
    ) -> "Scalar[float]":
        T_new = T * alpha

        if p_acc < target:
            T_new = T_new * (1.0 + 0.1 * (target - p_acc))
        else:
            T_new = T_new / (1.0 + 0.1 * (p_acc - target))

        if T_new < 1e-8:
            T_new = 1e-8

        return T_new

    @cm.vectorize
    def adapt_step(
        step: "Scalar[float]",
        p_acc: "Scalar[float]",
        target: "Scalar[float]",
    ) -> "Scalar[float]":
        if p_acc < target:
            step_new = step * 0.95
        else:
            step_new = step * 1.05

        if step_new < 1e-3:
            step_new = 1e-3
        if step_new > 1.0:
            step_new = 1.0

        return step_new


def rastrigin(x):
    x = np.asarray(x)
    n = x.shape[-1] if x.ndim > 0 else 1
    terms = x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)
    if x.ndim == 2:
        return 10.0 * n + np.sum(terms, axis=1)
    return 10.0 * n + np.sum(terms)


if __name__ == "__main__":
    model = SimulatedAnnealing(seed=123)
    bounds = cm.Bounds[(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)]
    best_f, best_x = model.solve(rastrigin, bounds, epochs=1500, pop_size=20, pop_range=(20, 20))
    print("Best fitness:", best_f)
    print("Best solution:", best_x)
    print("Kernels:", cm.prepare(SimulatedAnnealing))
