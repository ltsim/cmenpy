from cmenpy.kernel import KernelBuffer
from cmenpy.kernel.index import BufferIndex
from cmenpy.hints import NDArrayType


class BaseProperty:
    def __init__(self, buffer: KernelBuffer):
        self.__buffer = buffer

    @property
    def F(self) -> NDArrayType:
        _F = self.__buffer.raw[:, 0].reshape(-1)
        return _F

    @property
    def X(self) -> NDArrayType:
        _X = self.__buffer.raw[:, 1:]
        return _X

    @property
    def idx(self) -> BufferIndex:
        return self.__buffer.idx

    @property
    def size(self):
        return self.__buffer.mask.active
