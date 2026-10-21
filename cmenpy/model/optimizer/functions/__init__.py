"""The function-style model (`FunctionOptimizerModel`) built by ``@cm.declare``."""

import functools
import typing

from cmenpy.bounds import Bounds, SequenceStructure
from cmenpy.context import MainContextManager
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
    """Model built by ``@cm.declare`` from a function.

    The wrapped function receives ``(args, epoch, ctx)`` and drives the
    whole run. Bind parameters by calling the model, then `solve`.

    Args:
        alias: Display name; overwritten by the function name on ``define``.
        seed: Random seed forwarded to the run.
        **kwargs: ``name=cm.Argument[...]`` field declarations.

    Attributes:
        alias: The model name.
        tracker: Per-epoch history (empty before the first ``solve``).
    """

    def __init__(
        self,
        alias: str = "Optimizer",
        seed: int | None = None,
        **kwargs: typing.Any,
    ):
        """Initialize the model state (alias, arguments, seed, debug, sense)."""
        self.__alias = alias
        self.__inner: CallableFunction | None = None
        self.__resource: MainContextManager | None = None
        self.__arguments: FunctionParamsArguments = FunctionParamsArguments(kwargs)
        self.__seed: int | None = seed
        self.__debug = False

    @property
    def alias(self) -> str:
        """The model name."""
        return self.__alias

    @property
    def tracker(self) -> list[EpochHistory]:
        """Per-epoch history (empty before the first ``solve``)."""
        if self.__resource is None:
            return []

        return self.__resource.tracker.history

    def define(self, func: AlgorithmFunction) -> "FunctionOptimizerModel":
        """Bind ``func`` as the run driver, building the solve closure.

        Args:
            func: The ``(args, epoch, ctx) -> None`` algorithm function.

        Returns:
            The model itself, for chaining.
        """

        @functools.wraps(func)
        def wrapper(
            f: Target,
            bounds: Bounds | SequenceStructure[ScalarType],
            epochs: int,
            pop_size: int,
            pop_range: tuple[int, int] | None = None,
            sense: SenseType = "min",
        ) -> tuple[float, NDArrayType]:
            self.__resource = build_context_manager(
                f, bounds, epochs, pop_size, pop_range, self.__seed, self.__debug, sense
            )

            func(
                args=self.__arguments,
                epoch=self.__resource.epoch_it,
                ctx=self.__resource.context(),
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
        """Run the optimization and return the best ``(fitness, solution)``.

        Args:
            f: The objective callable.
            bounds: The search space (or bounds sequence).
            epochs: Total number of epochs.
            pop_size: Population size.
            pop_range: ``(min, max)`` allowed population sizes.
            debug: Record per-epoch history.
            sense: ``"min"`` (default) or ``"max"``.

        Returns:
            The best fitness and the best solution vector found.

        Raises:
            NotImplementedError: If no function was bound via ``define``.
        """
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
        """Bind the model parameters (and optionally the random ``seed``).

        Raises:
            ValueError: If an argument value is invalid.
            TypeError: If an argument type is invalid.
            NotImplementedError: If no function was bound via ``define``.
        """
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
