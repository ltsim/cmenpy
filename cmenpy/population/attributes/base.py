from cmenpy.kernel import KernelBuffer
from cmenpy.kernel.index import BufferIndex
from cmenpy.hints import NDArrayType
from cmenpy.population.attributes.matrix import (
    BufferMatrixAbstract,
    BufferMatrixCompute,
    BufferMatrixAccess,
)


class BufferMatrixFitness(BufferMatrixAbstract, BufferMatrixAccess):
    def __init__(self, buffer: KernelBuffer):
        self.__buffer = buffer

    def __getitem__(self, item: int) -> NDArrayType:
        return self.__buffer.raw[:, 0].reshape(-1)[item]

    def __array__(self, dtype=None, copy=None) -> NDArrayType:
        return self.__buffer.raw[:, 0].reshape(-1)

    def __repr__(self) -> str:
        _F = ", ".join(map(str, self.__buffer.raw[:, 0].reshape(-1)))

        return f"F [{_F}]"

    def __str__(self) -> str:
        _F = ", ".join(map(str, self.__buffer.raw[:, 0].reshape(-1)))

        return f"F [{_F}]"

    def __invert__(self) -> NDArrayType:
        return self.__buffer.raw[:, 0].reshape(-1)[self.__buffer.mask.founds]


class BufferMatrixSolutions(BufferMatrixAbstract, BufferMatrixAccess, BufferMatrixCompute):
    def __init__(self, buffer: KernelBuffer):
        self.__buffer = buffer

    def __getitem__(self, item: int) -> NDArrayType:
        return self.__buffer.raw[:, 1:][item]

    def __array__(self, dtype=None, copy=None) -> NDArrayType:
        return self.__buffer.raw[:, 1:]

    def __repr__(self) -> str:
        _X = ", ".join(map(str, self.__buffer.raw[:, 1:]))

        return f"X [{_X}]"

    def __str__(self) -> str:
        _X = ", ".join(map(str, self.__buffer.raw[:, 1:]))

        return f"X [{_X}]"

    def __invert__(self) -> NDArrayType:
        return self.__buffer.raw[:, 1:][self.__buffer.mask.founds]

    def __lshift__(self, other):
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
    def size(self):
        return self.__buffer.mask.active
