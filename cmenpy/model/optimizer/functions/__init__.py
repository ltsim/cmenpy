import functools
import typing

from cmenpy.agent import Agent
from cmenpy.bounds import Bounds, SequenceStructure, create_bounds
from cmenpy.model.optimizer.base import BaseOptimizer
from cmenpy.model.optimizer.functions.params import FunctionParamsArguments
from cmenpy.model.optimizer.functions.protocols import (
    CallableFunction,
    AlgorithmFunction,
)
from cmenpy.resource import OptimizerResourceManager
from cmenpy.target import Target
from cmenpy.types import DType


class FunctionOptimizerModel(BaseOptimizer):
    def __init__(self, alias: str = "Optimizer", **kwargs):
        self.__alias = alias if isinstance(alias, str) else alias
        self.__inner: typing.Optional[CallableFunction] = None
        self.__resource: typing.Optional[OptimizerResourceManager] = None
        self.__arguments: FunctionParamsArguments = FunctionParamsArguments(kwargs)

    @property
    def alias(self) -> str:
        return self.__alias

    def define(self, func: AlgorithmFunction):
        @functools.wraps(func)
        def wrapper(
            f: Target,
            bounds: Bounds | SequenceStructure[DType],
            epochs: int,
            pop_size: int,
            pop_range: typing.Optional[tuple[int, int]] = None,
        ) -> Agent:
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

            func(
                pop=self.__resource.population,
                bounds=self.__resource.bounds,
                epoch=self.__resource.epoch_it,
                args=self.__arguments,
            )

            return self.__resource.population.best

        self.__alias = func.__name__
        self.__inner = lambda *args, **kwargs: wrapper(*args, **kwargs)

        return self

    def solve(
        self,
        f: Target,
        bounds: Bounds | SequenceStructure[DType],
        epochs: int,
        pop_size: int,
        pop_range: typing.Optional[tuple[int, int]] = None,
    ):
        if self.__inner is not None:
            return self.__inner(
                f=f,
                bounds=bounds,
                epochs=epochs,
                pop_size=pop_size,
                pop_range=pop_range,
            )

        raise NotImplementedError()

    def __call__(self, *args, **kwargs) -> BaseOptimizer:
        self.__arguments.load(kwargs)

        if self.__inner is not None:
            return self

        raise NotImplementedError()
