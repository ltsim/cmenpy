import abc

from cmenpy.agent import Agent
from cmenpy.types import NDArrayType


class BasePopulation(abc.ABC):
    @abc.abstractmethod
    def __iter__(self): ...

    @abc.abstractmethod
    def __repr__(self): ...

    @property
    @abc.abstractmethod
    def size(self) -> int: ...

    @property
    @abc.abstractmethod
    def all(self) -> list[Agent]: ...

    @property
    @abc.abstractmethod
    def fitnesses(self) -> NDArrayType: ...

    @property
    @abc.abstractmethod
    def solutions(self) -> NDArrayType: ...
