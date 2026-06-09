import functools
import typing

from cmenpy.bounds import Bounds, SequenceStructure
from cmenpy.context import MainContextManager, Context
from cmenpy.kernel import KernelSize
from cmenpy.model.optimizer.base import BaseOptimizer
from cmenpy.model.optimizer.functions.params import (
    FunctionParamsArguments,
    ExceptionValueParams,
    ExceptionTypeParams,
)
from cmenpy.model.optimizer.functions.protocols import (
    CallableFunction,
    AlgorithmFunction,
)
from cmenpy.target import Target
from cmenpy.tracker import EpochHistory
from cmenpy.hints import ScalarType, NDArrayType
from cmenpy.hints.option import SenseType


class FunctionOptimizerModel(BaseOptimizer):
    def __init__(
        self,
        alias: str = "Optimizer",
        seed: typing.Optional[int] = None,
        **kwargs: typing.Any,
    ):
        self.__alias = alias if isinstance(alias, str) else alias
        self.__inner: typing.Optional[CallableFunction | typing.Any] = None
        self.__resource: typing.Optional[MainContextManager] = None
        self.__arguments: FunctionParamsArguments = FunctionParamsArguments(kwargs)
        self.__seed: typing.Optional[int] = seed
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

    def define(self, func: AlgorithmFunction):
        @functools.wraps(func)
        def wrapper(
            f: Target,
            bounds: Bounds | SequenceStructure[ScalarType],
            epochs: int,
            pop_size: int,
            pop_range: typing.Optional[tuple[int, int]] = None,
            sense: SenseType = "min",
        ) -> tuple[float, NDArrayType]:
            if pop_range is None:
                pop_range = pop_size, pop_size

            min_pop, max_pop = pop_range

            if min_pop > max_pop or pop_size < min_pop:
                raise IndexError("Population size is too small.")
            elif pop_size > max_pop:
                raise IndexError("Population size is too large.")

            self.__sense = sense
            self.__size = KernelSize(pop_size, min_size=min_pop, max_size=max_pop)
            self.__bounds = Bounds[bounds]
            self.__resource = MainContextManager(
                f,
                self.__bounds,
                epochs,
                self.__size,
                self.__seed,
                self.__debug,
                self.__sense,
            )

            if self.__resource is None:
                raise NotImplementedError("Function not implemented.")

            func(
                args=self.__arguments,
                epoch=self.__resource.epoch_it,
                ctx=Context(
                    f=self.__resource.target,
                    bounds=self.__resource.bounds,
                    buffer=self.__resource.buffer,
                    generator=self.__resource.default_generator,
                    sense=self.__sense,
                ),
            )

            X = self.__resource.buffer.raw_data[self.__resource.buffer.idx.best]

            return float(X[0]), X[1:]

        self.__alias = func.__name__
        self.__inner = wrapper

        return self

    def solve(
        self,
        f: Target,
        bounds: Bounds | SequenceStructure[ScalarType],
        epochs: int,
        pop_size: int,
        pop_range: typing.Optional[tuple[int, int]] = None,
        debug: bool = False,
        sense: SenseType = "min",
    ):
        self.__debug = debug

        if self.__inner is not None:
            return self.__inner(
                f=f,
                bounds=bounds,
                epochs=epochs,
                pop_size=pop_size,
                pop_range=pop_range,
            )

        raise NotImplementedError()

    def __call__(
        self, seed: typing.Optional[int] = None, *args: typing.Any, **kwargs: typing.Any
    ) -> BaseOptimizer:
        self.__seed = seed

        try:
            self.__arguments.load(kwargs)
        except ExceptionValueParams as e:
            raise ValueError(f"Invalid value in arguments: {str(e)}")
        except ExceptionTypeParams as e:
            raise TypeError(f"Invalid type in arguments: {str(e)}")

        if self.__inner is not None:
            return self

        raise NotImplementedError()
