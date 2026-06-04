import numpy as np
import cmenpy as cm

from opfunu.name_based import Ackley01


@cm.declare(...)
def gwo(args, epoch, ctx):
    pop, bounds, rng = ctx.population, ctx.bounds, ctx.rng

    pop.compute << rng.uniform(bounds.low, bounds.up, (len(pop), bounds.ndim))
    X = np.full((pop.size, bounds.ndim), np.nan)

    for e in epoch:
        sorted_pop = cm.sort_agents(pop, ctx.sense)

        n_pop = pop.view
        b_pop = sorted_pop[:3]

        a = 2 - 2 * e / epoch.max

        A = a * (2 * rng.random(size=(pop.size, len(b_pop), bounds.ndim)) - 1)
        C = 2 * rng.random(size=(pop.size, len(b_pop), bounds.ndim))

        for i, p in enumerate(sorted_pop):
            D = [
                b.solution - A[i][j] * np.abs(C[i][j] * b.solution - p.solution)
                for j, b in enumerate(b_pop)
            ]

            X[i] = np.sum(D, axis=0) / len(b_pop)

        n_pop.compute << np.clip(
            X,
            bounds.low,
            bounds.up,
        )

        for i, (a, b) in enumerate(zip(n_pop, sorted_pop)):
            pop.assign[i] << cm.best_of((a, b), ctx.sense).buff


if __name__ == "__main__":
    model = gwo()
    f = Ackley01(ndim=30)

    b_pop = model.solve(f.evaluate, f.bounds, 150, 75)
    print(b_pop)
