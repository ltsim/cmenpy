import numpy as np

import cmenpy as cm


@cm.declare(x=cm.Argument[int, (0, 3), 2])
def my_algorithm(args, pop, bounds, epoch, rng):
    pop @= rng.uniform(-1, 1, (len(pop), bounds.ndim))

    b_pop = pop.best

    for _ in epoch:
        pop @= np.clip(~pop + rng.uniform(-1, 1, (len(pop), bounds.ndim)), -1, 1)

        if pop.best < b_pop and pop.size > 1:
            b_pop = pop.best
            w_pop = pop.worst

            pop.remove(w_pop.id)
        else:
            if pop.free_space:
                n_pop = pop.append(rng.uniform(-1, 1, bounds.ndim))


if __name__ == "__main__":
    model = my_algorithm(x=1, seed=1)

    def sphere(x):
        return np.sum(x**2)

    def rastrigin(x):
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    def ackley(x):
        n = len(x)
        s1 = np.sum(x**2)
        s2 = np.sum(np.cos(2 * np.pi * x))

        return -20 * np.exp(-0.2 * np.sqrt(s1 / n)) - np.exp(s2 / n) + 20 + np.e

    def peaks(i):
        x = i[0]
        y = i[1]

        a = 3 * (1 - x) ** 2 * np.exp(-(x**2) - (y + 1) ** 2)
        b = -10 * (x / 5 - x**3 - y**5) * np.exp(-(x**2) - y**2)
        c = -1 / 3 * np.exp(-((x + 1) ** 2) - y**2)

        return a + b + c

    best_pop = my_algorithm.solve(
        sphere, cm.Bounds([(-1, 1), (-1, 1)]), 1500, 15, (1, 20)
    )
    print("Best population:", best_pop)
