import typing

from cmenpy import low
from cmenpy.agent import Agent
from cmenpy.bounds import Bounds, SequenceStructure, create_bounds
from cmenpy.epoch import EpochIteration
from cmenpy.model.optimizer.base import BaseOptimizer
from cmenpy.model.optimizer.classes.protocol import ModelProtocol
from cmenpy.target import TargetFunction, Target
from cmenpy.types import NDArrayType, DType


class ClassOptimizerModel(BaseOptimizer):
    def __init__(self, cls_model: typing.Type[ModelProtocol]) -> None:
        self.__alias = str(cls_model.__name__)
        self.__cls_model: typing.Type[ModelProtocol] = cls_model
        self.__model: ModelProtocol | None = None
        self.__inner = None
        self.__buffer: NDArrayType | None = None
        self.__epoch: EpochIteration | None = None

    @property
    def alias(self) -> str:
        return self.__alias

    def solve(self, f: Target, bounds: Bounds | SequenceStructure[DType], n_it: int, n_pop: int,
              r_pop: typing.Optional[tuple[int, int]] = None) -> Agent:
        if self.__model is None:
            raise NotImplementedError()

        if r_pop is None:
            r_pop = n_pop, n_pop

        min_pop, max_pop = r_pop

        if min_pop > max_pop or n_pop < min_pop:
            raise IndexError("Population size is too small.")
        elif n_pop > max_pop:
            raise IndexError("Population size is too large.")

        bounds = create_bounds(bounds)
        target = TargetFunction(f, bounds)

        buffer = low.init_buffer(
            max_pop, bounds.ndim
        )

        epoch = EpochIteration(buffer, target, n_it, n_pop, r_pop)
        population = epoch.population

        self.__epoch = epoch
        self.__buffer = buffer

        if hasattr(self.__model, "initialize"):
            self.__model.initialize(
                population=population,
                bounds=bounds,
            )

        for e in epoch:
            self.__model.evolve(
                e=e,
                population=population,
                bounds=bounds,
            )

        return population.best

    def __call__(self, *args, **kwargs) -> "ClassOptimizerModel":
        self.__model = self.__cls_model(
            *args,
            **kwargs
        )

        return self
