import abc
from collections.abc import MutableSequence

from cmenpy.agent import MemoryAgent
from cmenpy.population.agents.collection.sequence import AgentSequenceCollection
from cmenpy.types import NDArrayType


class AgentMutableCollection(abc.ABC, MutableSequence):
    @abc.abstractmethod
    def append(self, value: NDArrayType) -> None: ...

    @abc.abstractmethod
    def pop(self, index: int = -1) -> None: ...

    @property
    @abc.abstractmethod
    def max(self) -> MemoryAgent: ...

    @property
    @abc.abstractmethod
    def size(self) -> int: ...

    @property
    @abc.abstractmethod
    def free_space(self) -> bool: ...

    @property
    @abc.abstractmethod
    def sort(self) -> AgentSequenceCollection: ...
