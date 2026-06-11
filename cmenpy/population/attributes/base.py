from cmenpy.hints.interface import TensorABC
from cmenpy.kernel import KernelBuffer
from cmenpy.kernel.index import BufferIndex
from cmenpy.hints import NDArrayType


class TensorFitness(TensorABC):
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


class TensorSolutions(TensorABC):
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


class BaseTensorAttributes:
    def __init__(self, buffer: KernelBuffer):
        self.__buffer = buffer

    @property
    def F(self) -> TensorFitness:
        return TensorFitness(self.__buffer)

    @property
    def X(self) -> TensorSolutions:
        return TensorSolutions(self.__buffer)

    @property
    def idx(self) -> BufferIndex:
        return self.__buffer.idx

    @property
    def size(self):
        return self.__buffer.mask.active
