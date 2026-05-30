import typing

from cmenpy import low
from cmenpy.bounds import Bounds
from cmenpy.epoch import EpochIteration
from cmenpy.target import TargetFunction, Target
from cmenpy.model.protocol import ModelProtocol


class OptimizerModel:
    def __init__(self, cls_model: typing.Type[ModelProtocol]) -> None:
        self.__alias = str(cls_model.__name__)
        self.__cls_model: typing.Type[ModelProtocol] = cls_model
        self.__model: ModelProtocol | None = None

        self.__inner = None
        self.__buffer = None
        self.__epoch = None

    def __del__(self) -> None:
        pass

    @property
    def alias(self) -> str:
        return self.__alias

    def solve(self, f: Target, bounds, n_it, n_pop, r_pop=None):
        if self.__model is None:
            raise NotImplementedError

        if r_pop is None:
            r_pop = n_pop, n_pop

        min_pop, max_pop = r_pop

        if min_pop > max_pop or n_pop < min_pop:
            raise IndexError("Population size is too small.")
        elif n_pop > max_pop:
            raise IndexError("Population size is too large.")

        bounds = Bounds(bounds)
        target = TargetFunction(f, bounds)

        self.__buffer = low.init_buffer(
            max_pop, bounds.ndim
        )

        self.__epoch = EpochIteration(
            self.__buffer, target, n_it, n_pop, r_pop
        )

        if hasattr(self, "initialize"):
            self.__model.initialize(
                population=self.__epoch.population,
                bounds=bounds,
            )

        for e in range(self.__model.epoch):
            self.__model.evolve(
                e=e,
                population=self.__epoch.population,
                bounds=bounds,
            )

    def __call__(self, *args, **kwargs) -> "OptimizerModel":
        self.__model = self.__cls_model(
            *args, **kwargs
        )

        return self
