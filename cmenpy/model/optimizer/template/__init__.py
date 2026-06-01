import typing

from cmenpy.agent import Agent
from cmenpy.bounds import Bounds, SequenceStructure, create_bounds
from cmenpy.model.optimizer.base import BaseOptimizer
from cmenpy.model.optimizer.template.protocols import ModelProtocol
from cmenpy.model.optimizer.functions import CallableFunction
from cmenpy.resource import MainContextManager, Context
from cmenpy.target import Target
from cmenpy.types import DType


class TemplateOptimizerModel(BaseOptimizer):
    def __init__(self, cls_model: typing.Type[ModelProtocol]) -> None:
        self.__alias = str(cls_model.__name__)
        self.__cls_model: typing.Type[ModelProtocol] = cls_model
        self.__inner: typing.Optional[CallableFunction] = None
        self.__resource: typing.Optional[MainContextManager] = None
        self.__seed: typing.Optional[int] = None

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

        self.__resource = MainContextManager(
            f, create_bounds(bounds), epochs, pop_size, pop_range, self.__seed
        )

        if self.__resource is None:
            raise NotImplementedError("Function not implemented.")

        if hasattr(self.__model, "initialize"):
            self.__model.initialize(
                ctx=Context(
                    f=self.__resource.target,
                    bounds=self.__resource.bounds,
                    population=self.__resource.population,
                    generator=self.__resource.default_generator,
                ),
            )

        for e in self.__resource.epoch_it:
            self.__model.evolve(
                e=e,
                ctx=Context(
                    f=self.__resource.target,
                    bounds=self.__resource.bounds,
                    population=self.__resource.population,
                    generator=self.__resource.default_generator,
                ),
            )

        return self.__resource.population.best

    def __call__(
        self, seed: typing.Optional[int] = None, *args, **kwargs
    ) -> "BaseOptimizer":
        self.__seed = seed
        self.__model = self.__cls_model(*args, **kwargs)

        return self
