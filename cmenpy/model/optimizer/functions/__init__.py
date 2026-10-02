import functools
import typing

from cmenpy.bounds import Bounds, SequenceStructure
from cmenpy.context import Context, MainContextManager
from cmenpy.hints import NDArrayType, ScalarType
from cmenpy.hints.option import SenseType
from cmenpy.model.optimizer.base import BaseOptimizer, build_context_manager
from cmenpy.model.optimizer.functions.params import (
    ExceptionTypeParams,
    ExceptionValueParams,
    FunctionParamsArguments,
)
from cmenpy.model.optimizer.functions.protocols import (
    AlgorithmFunction,
    CallableFunction,
)
from cmenpy.target import Target
from cmenpy.tracker import EpochHistory


class FunctionOptimizerModel(BaseOptimizer):
    def __init__(
        self,
        alias: str = "Optimizer",
        seed: int | None = None,
        **kwargs: typing.Any,
    ):
        self.__alias = alias
        self.__inner: CallableFunction | None = None
        self.__resource: MainContextManager | None = None
        self.__arguments: FunctionParamsArguments = FunctionParamsArguments(kwargs)
        self.__seed: int | None = seed
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

    def define(self, func: AlgorithmFunction) -> "FunctionOptimizerModel":
        @functools.wraps(func)
        def wrapper(
            f: Target,
            bounds: Bounds | SequenceStructure[ScalarType],
            epochs: int,
            pop_size: int,
            pop_range: tuple[int, int] | None = None,
            sense: SenseType = "min",
        ) -> tuple[float, NDArrayType]:
            self.__sense = sense
            self.__resource = build_context_manager(
                f, bounds, epochs, pop_size, pop_range, self.__seed, self.__debug, sense
            )

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

            X = self.__resource.buffer.raw[self.__resource.buffer.idx.best]

            return float(X[0]), X[1:]

        self.__alias = func.__name__
        self.__inner = typing.cast(CallableFunction, wrapper)

        return self

    def solve(
        self,
        f: Target,
        bounds: Bounds | SequenceStructure[ScalarType],
        epochs: int,
        pop_size: int,
        pop_range: tuple[int, int] | None = None,
        debug: bool = False,
        sense: SenseType = "min",
    ) -> tuple[float, NDArrayType]:
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
        self, seed: int | None = None, *args: typing.Any, **kwargs: typing.Any
    ) -> BaseOptimizer:
        self.__seed = seed

        try:
            self.__arguments.load(kwargs)
        except ExceptionValueParams as e:
            raise ValueError(f"Invalid value in arguments: {e!s}")
        except ExceptionTypeParams as e:
            raise TypeError(f"Invalid type in arguments: {e!s}")

        if self.__inner is not None:
            return self

        raise NotImplementedError()


__all__ = [
    "AlgorithmFunction",
    "CallableFunction",
    "ExceptionTypeParams",
    "ExceptionValueParams",
    "FunctionOptimizerModel",
    "FunctionParamsArguments",
]
