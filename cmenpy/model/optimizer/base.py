import abc
import typing

from cmenpy.agent import Agent
from cmenpy.bounds import Bounds, SequenceStructure
from cmenpy.target import Target
from cmenpy.types import DType


class BaseOptimizer(abc.ABC):
    @abc.abstractmethod
    def solve(
        self,
        f: Target,
        bounds: Bounds | SequenceStructure[DType],
        epochs: int,
        pop_size: int,
        pop_range: typing.Optional[tuple[int, int]],
    ) -> Agent: ...

    @abc.abstractmethod
    def __call__(self, *args, **kwargs): ...
