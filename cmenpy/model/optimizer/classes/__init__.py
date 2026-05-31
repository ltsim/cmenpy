import typing

from cmenpy.agent import Agent
from cmenpy.bounds import Bounds, SequenceStructure, create_bounds
from cmenpy.model.optimizer.functions import CallableFunction
from cmenpy.model.optimizer.base import BaseOptimizer
from cmenpy.model.optimizer.classes.protocol import ModelProtocol
from cmenpy.resource import OptimizerResourceManager
from cmenpy.target import Target
from cmenpy.types import DType


class ClassOptimizerModel(BaseOptimizer):
    def __init__(self, cls_model: typing.Type[ModelProtocol]) -> None:
        self.__alias = str(cls_model.__name__)
        self.__cls_model: typing.Type[ModelProtocol] = cls_model
        self.__inner: typing.Optional[CallableFunction] = None
        self.__resource: typing.Optional[OptimizerResourceManager] = None

    @property
    def alias(self) -> str:
        return self.__alias

    def solve(
        self,
        f: Target,
        bounds: Bounds | SequenceStructure[DType],
        epochs: int,
        pop_size: int,
        pop_range: typing.Optional[tuple[int, int]] = None,
    ) -> Agent:
        if self.__model is None:
            raise NotImplementedError()

        if pop_range is None:
            pop_range = pop_size, pop_size

        min_pop, max_pop = pop_range

        if min_pop > max_pop or pop_size < min_pop:
            raise IndexError("Population size is too small.")
        elif pop_size > max_pop:
            raise IndexError("Population size is too large.")

        self.__resource = OptimizerResourceManager(
            f, create_bounds(bounds), epochs, pop_size, pop_range
        )

        if self.__resource is None:
            raise NotImplementedError("Function not implemented.")

        if hasattr(self.__model, "initialize"):
            self.__model.initialize(
                population=self.__resource.population,
                bounds=self.__resource.bounds,
            )

        for e in self.__resource.epoch_it:
            self.__model.evolve(
                e=e,
                population=self.__resource.population,
                bounds=self.__resource.bounds,
            )

        return self.__resource.population.best

    def __call__(self, *args, **kwargs) -> "ClassOptimizerModel":
        self.__model = self.__cls_model(*args, **kwargs)

        return self
