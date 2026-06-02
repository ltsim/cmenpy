import abc

from cmenpy.agent import Agent
from cmenpy.types import NDArrayType


class BasePopulation(abc.ABC):
    @abc.abstractmethod
    def __iter__(self): ...

    @abc.abstractmethod
    def __repr__(self): ...

    @abc.abstractmethod
    def __getitem__(self, key: int): ...

    @abc.abstractmethod
    def __setitem__(self, key: int, value: NDArrayType): ...

    @abc.abstractmethod
    def __matmul__(self, other: NDArrayType): ...

    @abc.abstractmethod
    def __imatmul__(self, other: NDArrayType): ...

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
