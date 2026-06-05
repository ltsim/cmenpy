import typing

from cmenpy.bounds import Bounds, SequenceStructure, create_bounds
from cmenpy.context import MainContextManager, Context
from cmenpy.model.optimizer.base import BaseOptimizer
from cmenpy.model.optimizer.functions import CallableFunction
from cmenpy.model.optimizer.template.protocols import ModelProtocol
from cmenpy.population.agent import Agent
from cmenpy.target import Target
from cmenpy.tracker import EpochHistory
from cmenpy.types import DType
from cmenpy.types.option import SenseType


class TemplateOptimizerModel(BaseOptimizer):
    def __init__(self, cls_model: typing.Type[ModelProtocol]) -> None:
        self.__alias = str(cls_model.__name__)
        self.__cls_model: typing.Type[ModelProtocol] = cls_model
        self.__inner: typing.Optional[CallableFunction] = None
        self.__resource: typing.Optional[MainContextManager] = None
        self.__seed: typing.Optional[int] = None
        self.__debug = False
        self.__sense: SenseType = "min"

    @property
    def alias(self) -> str:
        return self.__alias

    @property
    def tracker(self) -> list[EpochHistory]:
        if self.__resource is None:
            return []

        return self.__resource.tracker.history

    def solve(
        self,
        f: Target,
        bounds: Bounds | SequenceStructure[DType],
        epochs: int,
        pop_size: int,
        pop_range: typing.Optional[tuple[int, int]] = None,
        debug: bool = False,
        sense: SenseType = "min",
    ) -> Agent:
        if self.__model is None:
            raise NotImplementedError()

        self.__debug = debug

        if pop_range is None:
            pop_range = pop_size, pop_size

        min_pop, max_pop = pop_range

        if min_pop > max_pop or pop_size < min_pop:
            raise IndexError("Population size is too small.")
        elif pop_size > max_pop:
            raise IndexError("Population size is too large.")

        self.__sense = sense
        self.__resource = MainContextManager(
            f,
            create_bounds(bounds),
            epochs,
            pop_size,
            pop_range,
            self.__seed,
            self.__debug,
            self.__sense,
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
                    sense=self.__sense,
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
                    sense=self.__sense,
                ),
            )

        return self.__resource.population.agents.best

    def __call__(
        self, seed: typing.Optional[int] = None, *args: typing.Any, **kwargs: typing.Any
    ) -> "BaseOptimizer":
        self.__seed = seed
        self.__model = self.__cls_model(*args, **kwargs)

        return self
