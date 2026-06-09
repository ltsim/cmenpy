from collections.abc import MutableSequence

from cmenpy.kernel import KernelBuffer
from cmenpy.types import NDArrayType
from cmenpy.population.agent import MutableAgent
from cmenpy.population.people.collection.sequence import PeopleSequenceCollection


class PeopleMutableCollection(MutableSequence):
    def __init__(self, buffer: KernelBuffer) -> None:
        self.__buffer = buffer

    def __getitem__(self, index):
        pass

    def __setitem__(self, index, value):
        pass

    def __delitem__(self, index):
        pass

    def __len__(self):
        pass

    def append(self, value: NDArrayType) -> None: ...

    def pop(self, index: int = -1) -> None: ...

    def insert(self, index, value):
        pass

    @property
    def max(self) -> MutableAgent: ...

    @property
    def size(self) -> int: ...

    @property
    def free_space(self) -> bool: ...

    @property
    def sort(self) -> PeopleSequenceCollection: ...
