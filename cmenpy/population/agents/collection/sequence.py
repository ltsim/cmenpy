import abc
import typing
from collections.abc import Sequence

from cmenpy.agent import MutableAgent


class AgentSequenceCollection(abc.ABC, Sequence):
    @abc.abstractmethod
    def __getitem__(self, index) -> MutableAgent | typing.List[MutableAgent]: ...

    @abc.abstractmethod
    def __iter__(self) -> typing.Iterator[MutableAgent]: ...

    @abc.abstractmethod
    def __len__(self) -> int: ...
