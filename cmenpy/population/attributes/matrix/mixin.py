import abc

from cmenpy.hints import NDArrayType
import typing


class BufferMatrixAccess(abc.ABC):
    @abc.abstractmethod
    def __invert__(self) -> NDArrayType: ...


class BufferMatrixCompute(abc.ABC):
    @abc.abstractmethod
    def __lshift__(self, other: NDArrayType) -> typing.Self: ...
