from cmenpy.kernel import KernelBuffer
from cmenpy.kernel.index import BufferIndex
from cmenpy.types import NDArrayType


class BaseProperty:
    def __init__(self, buffer: KernelBuffer):
        self.__buffer = buffer

    @property
    def F(self) -> NDArrayType:
        return self.__buffer.buffer[[*self.__buffer.mask], 0].reshape(-1)

    @property
    def X(self) -> NDArrayType:
        return self.__buffer.buffer[[*self.__buffer.mask], 1:]

    @property
    def idx(self) -> BufferIndex:
        return self.__buffer.idx
