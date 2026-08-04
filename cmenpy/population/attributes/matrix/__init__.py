import abc

from cmenpy.hints import NDArrayType
from cmenpy.population.attributes.matrix.mixin import BufferMatrixCompute, BufferMatrixAccess


class BufferMatrixAbstract(abc.ABC):
    @abc.abstractmethod
    def __getitem__(self, item: int) -> NDArrayType: ...

    @abc.abstractmethod
    def __array__(self, dtype=None, copy=None) -> NDArrayType: ...

    @abc.abstractmethod
    def __repr__(self) -> str: ...

    @abc.abstractmethod
    def __str__(self) -> str: ...


__all__ = ["BufferMatrixAbstract", "BufferMatrixCompute", "BufferMatrixAccess"]
