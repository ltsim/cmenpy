import typing

import numpy as np

from cmenpy.agent import Agent
from cmenpy.bounds import Bounds, SequenceStructure
from cmenpy.epoch import EpochIteration
from cmenpy.model.optimizer.functions import FunctionParamsArguments
from cmenpy.population import Population
from cmenpy.target import Target
from cmenpy.types import DType


class AlgorithmFunction(typing.Protocol):
    __name__: str

    def __call__(
        self,
        pop: Population,
        bounds: Bounds,
        epoch: EpochIteration,
        args: typing.Optional[FunctionParamsArguments] = None,
        rng: typing.Optional[np.random.Generator] = None,
    ) -> None: ...


class CallableFunction(typing.Protocol):
    def __call__(
        self,
        f: Target,
        bounds: Bounds | SequenceStructure[DType],
        epochs: int,
        pop_size: int,
        pop_range: typing.Optional[tuple[int, int]] = None,
    ) -> Agent: ...
