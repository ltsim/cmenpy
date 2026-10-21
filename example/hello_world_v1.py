import numpy as np
from opfunu.name_based import Ackley01

import cmenpy as cm


@cm.declare(x=cm.Argument[int, (0, 3), 2])
def my_algorithm(args, epoch, ctx):
    buff, bounds, rng = ctx.buff, ctx.bounds, ctx.rng

    pop = cm.PopulationSwarm(buff)

    pop.compute << rng.uniform(bounds.low, bounds.up, (pop.size, bounds.ndim))
    b_pop = pop.F[pop.best]

    for _ in epoch:
        pop.compute << np.clip(
            ~pop.X + rng.uniform(-1, 1, (pop.size, bounds.ndim)), -1, 1
        )

        if pop.F[pop.best] < b_pop and pop.size > 1:
            pop.remove(pop.worst)
        else:
            if pop.free_space:
                pop.insert(rng.uniform(-1, 1, bounds.ndim))


if __name__ == "__main__":
    model = my_algorithm(x=1)

    f = Ackley01(ndim=3)

    best_pop = my_algorithm.solve(f.evaluate, f.bounds, 150, 15, (1, 25))
    print("Best population:", best_pop)
