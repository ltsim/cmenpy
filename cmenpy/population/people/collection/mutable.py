import abc
from collections.abc import MutableSequence

from cmenpy.population.agent import MutableAgent
from cmenpy.population.people.collection.sequence import PeopleSequenceCollection
from cmenpy.types import NDArrayType


class PeopleMutableCollection(abc.ABC, MutableSequence):
    @abc.abstractmethod
    def append(self, value: NDArrayType) -> None: ...

    @abc.abstractmethod
    def pop(self, index: int = -1) -> None: ...

    @property
    @abc.abstractmethod
    def max(self) -> MutableAgent: ...

    @property
    @abc.abstractmethod
    def size(self) -> int: ...

    @property
    @abc.abstractmethod
    def free_space(self) -> bool: ...

    @property
    @abc.abstractmethod
    def sort(self) -> PeopleSequenceCollection: ...
