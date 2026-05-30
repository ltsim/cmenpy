import functools

from cmenpy import low
from cmenpy.bounds import Bounds
from cmenpy.epoch import EpochIteration
from cmenpy.target import TargetFunction


class AlgorithmModel:
    def __init__(self, alias: str, seed=None):
        self.__alias = alias
        self.__inner = None
        self.__buffer = None
        self.__epoch = None

    def __del__(self):
        pass

    def define(self, func):
        @functools.wraps(func)
        def inner(f, bounds, n_it, n_pop, r_pop=None):
            if r_pop is None:
                r_pop = (n_pop, n_pop)

            min_pop, max_pop = r_pop

            if min_pop > max_pop or n_pop < min_pop:
                raise IndexError("Population size is too small.")
            elif n_pop > max_pop:
                raise IndexError("Population size is too large.")

            bounds = Bounds(bounds)
            target = TargetFunction(f, bounds)

            self.__buffer = low.init_buffer(max_pop, bounds.ndim)
            self.__epoch = EpochIteration(self.__buffer, target, n_it, n_pop, r_pop)

            return func(self.__epoch.population, bounds, self.__epoch)

        self.__inner = inner

        return inner

    def __call__(self, *args, **kwargs):
        return self.__inner(*args, **kwargs)


def define(alias: str):
    model = AlgorithmModel(alias)

    def inner(func):
        return model.define(func)

    return inner
