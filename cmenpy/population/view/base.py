import abc

from cmenpy.agent import Agent
from cmenpy.types import NDArrayType


class ViewBase(abc.ABC):
    @abc.abstractmethod
    def __repr__(self): ...

    @property
    @abc.abstractmethod
    def size(self) -> int: ...
