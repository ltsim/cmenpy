import abc
import typing

from cmenpy.bounds import Bounds, SequenceStructure
from cmenpy.population.agent import Agent
from cmenpy.target import Target
from cmenpy.tracker import EpochHistory
from cmenpy.types import DType, NDArrayType
from cmenpy.types.option import SenseType


class BaseOptimizer(abc.ABC):
    @abc.abstractmethod
    def solve(
        self,
        f: Target,
        bounds: Bounds | SequenceStructure[DType],
        epochs: int,
        pop_size: int,
        pop_range: typing.Optional[tuple[int, int]],
        debug: typing.Optional[bool],
        sense: typing.Optional[SenseType],
    ) -> tuple[DType, NDArrayType]: ...

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> "BaseOptimizer": ...

    @property
    @abc.abstractmethod
    def alias(self) -> str: ...

    def __repr__(self):
        return f"{self.__class__.__name__}(alias={self.alias})"

    @property
    @abc.abstractmethod
    def tracker(self) -> list[EpochHistory]: ...
