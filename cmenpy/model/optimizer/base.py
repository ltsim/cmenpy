import abc
import typing

from cmenpy.bounds import Bounds, SequenceStructure
from cmenpy.target import Target
from cmenpy.tracker import EpochHistory
from cmenpy.hints import ScalarType, ArrayType
from cmenpy.hints.option import SenseType


class BaseOptimizer(abc.ABC):
    @abc.abstractmethod
    def solve(
        self,
        f: Target,
        bounds: Bounds | SequenceStructure[ScalarType],
        epochs: int,
        pop_size: int,
        pop_range: typing.Optional[tuple[int, int]] = None,
        debug: typing.Optional[bool] = None,
        sense: typing.Optional[SenseType] = None,
    ) -> tuple[float, ArrayType]: ...

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
