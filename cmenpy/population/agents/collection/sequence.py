import abc
import typing
from collections.abc import Sequence

from cmenpy.agent import MemoryAgent


class AgentSequenceCollection(abc.ABC, Sequence):
    @abc.abstractmethod
    def __getitem__(self, index) -> MemoryAgent | typing.List[MemoryAgent]: ...

    @abc.abstractmethod
    def __iter__(self) -> typing.Iterator[MemoryAgent]: ...

    @abc.abstractmethod
    def __len__(self) -> int: ...
