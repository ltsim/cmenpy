from cmenpy.kernel import KernelBuffer
from cmenpy.kernel.index import BufferIndex
from cmenpy.hints import NDArrayType


class BaseProperty:
    def __init__(self, buffer: KernelBuffer):
        self.__buffer = buffer

    @property
    def F(self) -> NDArrayType:
        return self.__buffer.raw_data[:, 0].reshape(-1)

    @property
    def X(self) -> NDArrayType:
        return self.__buffer.raw_data[:, 1:]

    @property
    def idx(self) -> BufferIndex:
        return self.__buffer.idx

    @property
    def size(self):
        return self.__buffer.mask.active
