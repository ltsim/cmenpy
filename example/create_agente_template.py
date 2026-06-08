import numpy as np
import cmenpy as cm
from cmenpy.population.agent import AgentTemplate


class MyAgent(AgentTemplate):
    v: cm.Scalar[int]
    w: cm.Vector[float]


@cm.template
class MyAlgorithm:
    agent: cm.DefineAgent[MyAgent]

    a: cm.Argument[int, (1, 100)]
    b: cm.Argument[int, (1, 100)]

    w: cm.Variable[np.array]

    def initialize(self, population, bounds) -> None:
        population @= np.random.uniform(-1, 1, (len(population), bounds.ndim))

    def evolve(self, e: int, population: cm.PopulationSwarm, bounds: cm.Bounds) -> None:
        b_pop = population.best

        population @= np.clip(
            ~population + np.random.uniform(-1, 1, (len(population), bounds.ndim)),
            -1,
            1,
        )

        if population.best < b_pop and population.size > 1:
            b_pop = population.best
            w_pop = population.worst
            id_pop = w_pop.id

            population.remove(id_pop)
        else:
            if population.free_space:
                n_pop = population.append(np.random.uniform(-1, 1, bounds.ndim))


if __name__ == "__main__":
    model = MyAlgorithm(a=3, b=2)

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

    best_pop = model.solve(sphere, cm.Bounds([(-1, 1), (-1, 1)]), 1500, 15, (1, 20))
    print("Best population:", best_pop)
