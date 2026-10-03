import numpy.typing as npt

from cmenpy.hints import NDArrayType
from cmenpy.kernel import KernelBuffer
from cmenpy.kernel.index import BufferIndex
from cmenpy.population.attributes.matrix import (
    BufferMatrixAbstract,
    BufferMatrixAccess,
    BufferMatrixCompute,
)


class BufferMatrixFitness(BufferMatrixAbstract, BufferMatrixAccess):
    def __init__(self, buffer: KernelBuffer):
        self.__buffer = buffer

    def __getitem__(self, item: int | slice) -> NDArrayType:
        return self.__buffer.raw[:, 0].reshape(-1)[item]

    def __array__(
        self, dtype: "npt.DTypeLike | None" = None, copy: bool | None = None
    ) -> NDArrayType:
        return self.__buffer.raw[:, 0].reshape(-1)

    def __repr__(self) -> str:
        _F = ", ".join(map(str, self.__buffer.raw[:, 0].reshape(-1)))

        return f"F [{_F}]"

    __str__ = __repr__

    def __invert__(self) -> NDArrayType:
        return self.__buffer.raw[:, 0].reshape(-1)[self.__buffer.mask.founds]


class BufferMatrixSolutions(BufferMatrixAbstract, BufferMatrixAccess, BufferMatrixCompute):
    def __init__(self, buffer: KernelBuffer):
        self.__buffer = buffer

    def __getitem__(self, item: int | slice) -> NDArrayType:
        return self.__buffer.raw[:, 1:][item]

    def __array__(
        self, dtype: "npt.DTypeLike | None" = None, copy: bool | None = None
    ) -> NDArrayType:
        return self.__buffer.raw[:, 1:]

    def __repr__(self) -> str:
        _X = ", ".join(map(str, self.__buffer.raw[:, 1:]))

        return f"X [{_X}]"

    __str__ = __repr__

    def __invert__(self) -> NDArrayType:
        return self.__buffer.raw[:, 1:][self.__buffer.mask.founds]

    def __lshift__(self, other: NDArrayType) -> "BufferMatrixSolutions":
        self.__buffer.apply(other)
        return self


class BufferMatrixOperations:
    def __init__(self, buffer: KernelBuffer):
        self.__buffer = buffer

    @property
    def F(self) -> BufferMatrixFitness:
        return BufferMatrixFitness(self.__buffer)

    @property
    def X(self) -> BufferMatrixSolutions:
        return BufferMatrixSolutions(self.__buffer)

    @property
    def idx(self) -> BufferIndex:
        return self.__buffer.idx

    @property
    def size(self) -> int:
        return self.__buffer.mask.active
