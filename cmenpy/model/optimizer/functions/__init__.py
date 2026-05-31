import functools
import typing

from cmenpy import low
from cmenpy.agent import Agent
from cmenpy.bounds import Bounds, SequenceStructure, create_bounds
from cmenpy.epoch import EpochIteration
from cmenpy.model.optimizer.base import BaseOptimizer
from cmenpy.population import Population
from cmenpy.target import TargetFunction, Target
from cmenpy.types import DType, NDArrayType


class AlgorithmFunction(typing.Protocol):
    __name__: str

    def __call__(self, pop: Population, bounds: Bounds, epoch: EpochIteration) -> None:
        ...


class CallableFunction(typing.Protocol):
    def __call__(
            self,
            f: Target,
            bounds: Bounds | SequenceStructure[DType],
            epochs: int,
            pop_size: int,
            pop_range: typing.Optional[tuple[int, int]] = None
    ) -> Agent:
        ...


class FunctionOptimizerModel(BaseOptimizer):
    def __init__(self, alias: str | None = None, seed=None):
        self.__alias = alias if isinstance(alias, str) else alias
        self.__inner: typing.Optional[CallableFunction] = None
        self.__buffer: NDArrayType | None = None
        self.__epoch: EpochIteration | None = None

    def define(self, func: AlgorithmFunction):
        @functools.wraps(func)
        def wrapper(
                f: Target,
                bounds: Bounds | SequenceStructure[DType],
                epochs: int,
                pop_size: int,
                pop_range: typing.Optional[tuple[int, int]] = None
        ) -> Agent:
            if pop_range is None:
                pop_range = pop_size, pop_size

            min_pop, max_pop = pop_range

            if min_pop > max_pop or pop_size < min_pop:
                raise IndexError("Population size is too small.")
            elif pop_size > max_pop:
                raise IndexError("Population size is too large.")

            bounds = create_bounds(bounds)
            target = TargetFunction(f, bounds)

            buffer = low.init_buffer(max_pop, bounds.ndim)
            epoch = EpochIteration(buffer, target, epochs, pop_size, pop_range)
            population = epoch.population

            self.__buffer = buffer
            self.__epoch = epoch

            func(
                pop=population,
                bounds=bounds,
                epoch=epoch
            )

            return epoch.population.best

        self.__alias = func.__name__
        self.__inner = lambda *args, **kwargs: wrapper(*args, **kwargs)

        return self

    def solve(self, f: Target, bounds: Bounds | SequenceStructure[DType], epochs: int, pop_size: int,
              pop_range: typing.Optional[tuple[int, int]] = None):
        if self.__inner is not None:
            return self.__inner(
                f=f,
                bounds=bounds,
                epochs=epochs,
                pop_size=pop_size,
                pop_range=pop_range
            )

        raise NotImplementedError()

    def __call__(self, f: Target, bounds: Bounds | SequenceStructure[DType], epochs: int, pop_size: int,
                 pop_range: typing.Optional[tuple[int, int]] = None):
        if self.__inner is not None:
            return self.__inner(
                f=f,
                bounds=bounds,
                epochs=epochs,
                pop_size=pop_size,
                pop_range=pop_range
            )

        raise NotImplementedError()
