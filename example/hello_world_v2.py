import numpy as np
import cmenpy as cm


@cm.declare(strict=False)
class MyAlgorithm:
    epoch: cm.Argument[int, (1, 100)]
    population: cm.Argument[int, (1, 100)]

    def initialize(self, population, bounds) -> None:
        ...

    def evolve(self, e: int, population, bounds) -> None:
        ...


if __name__ == "__main__":
    model = MyAlgorithm(epoch=50, population=100)

    def sphere(x):
        return np.sum(x ** 2)


    def rastrigin(x):
        return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


    def ackley(x):
        n = len(x)
        s1 = np.sum(x ** 2)
        s2 = np.sum(np.cos(2 * np.pi * x))

        return -20 * np.exp(-0.2 * np.sqrt(s1 / n)) - np.exp(s2 / n) + 20 + np.e


    def peaks(i):
        x = i[0]
        y = i[1]

        a = 3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2)
        b = -10 * (x / 5 - x ** 3 - y ** 5) * np.exp(-x ** 2 - y ** 2)
        c = -1 / 3 * np.exp(-(x + 1) ** 2 - y ** 2)

        return a + b + c

    model.solve(sphere, [(-1, 1), (-1, 1)], 1500, 15, (1, 20))
