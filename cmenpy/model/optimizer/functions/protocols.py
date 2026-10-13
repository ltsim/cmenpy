import typing

from cmenpy.bounds import Bounds, SequenceStructure
from cmenpy.context import Context
from cmenpy.epoch import EpochIteration
from cmenpy.hints import NDArrayType, ScalarType
from cmenpy.model.optimizer.functions import FunctionParamsArguments
from cmenpy.target import Target


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
        bounds: Bounds | SequenceStructure[ScalarType],
        epochs: int,
        pop_size: int,
        pop_range: tuple[int, int] | None = None,
    ) -> tuple[float, NDArrayType]: ...
