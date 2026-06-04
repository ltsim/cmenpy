import numpy as np
import cmenpy as cm

from opfunu.name_based import Ackley01


@cm.declare(...)
def gwo(args, epoch, ctx):
    pop, bounds, rng = ctx.population, ctx.bounds, ctx.rng

    pop.compute << rng.uniform(bounds.low, bounds.up, (len(pop), bounds.ndim))
    X_n = np.full((pop.size, bounds.ndim), np.nan)

    for e in epoch:
        all_pop = cm.sort_agents(pop, ctx.sense)
        new_pop = pop.view

        best_pop = all_pop[:3]

        a = 2 - 2 * e / epoch.max

        for i, p in enumerate(all_pop):
            A = a * (2 * rng.random(size=(len(best_pop), bounds.ndim)) - 1)
            C = 2 * rng.random(size=(len(best_pop), bounds.ndim))

            D = [
                b.solution - A[j] * np.abs(C[j] * b.solution - p.solution)
                for j, b in enumerate(best_pop)
            ]

            X_n[i] = np.sum(D, axis=0) / len(best_pop)

        new_pop.compute << np.clip(
            X_n,
            bounds.low,
            bounds.up,
        )

        for i, (a, b) in enumerate(zip(new_pop, all_pop)):
            pop.assign[i] << cm.best_of((a, b), ctx.sense).buff


if __name__ == "__main__":
    model = gwo()
    f = Ackley01(ndim=30)

    b_pop = model.solve(f.evaluate, f.bounds, 150, 75)
    print(b_pop)
