import typing

from cmenpy.bounds import Bounds
from cmenpy.context import Context
from cmenpy.epoch import EpochIteration
from cmenpy.model.optimizer.functions import FunctionParamsArguments
from cmenpy.population.agent.base import BaseAgent
from cmenpy.target import Target
from cmenpy.hints import ScalarType


class AlgorithmFunction(typing.Protocol):
    __name__: str

    def __call__(
        self,
        args: FunctionParamsArguments,
        epoch: EpochIteration,
        ctx: Context,
    ) -> None: ...


class CallableFunction(typing.Protocol):
    def __call__(
        self,
        f: Target,
        bounds: Bounds,
        epochs: int,
        pop_size: int,
        pop_range: typing.Optional[tuple[int, int]] = None,
    ) -> BaseAgent: ...
