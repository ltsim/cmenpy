import numpy as np

from cmenpy.kernel.mask import KernelMask
from cmenpy.hints import ArrayType
from cmenpy.hints.option import SenseType


class BufferIndex:
    def __init__(self, buffer: ArrayType, mask: KernelMask, sense: SenseType):
        self.__buffer = buffer
        self.__sense = sense
        self.__mask = mask

    def __iter__(self):
        pop_size, _ = self.__buffer.shape

        return iter(range(0, pop_size))

    def __invert__(self):
        return np.where(~np.isnan(self.__buffer[:, 0]))[0].astype(int)

    @property
    def sort(self):
        _idx = np.argsort(self.__buffer[:, 0])

        if self.__sense == "max":
            return _idx[::-1]

        return _idx

    @property
    def best(self):
        if self.__sense == "max":
            return self.sort[-1]

        return self.sort[0]

    @property
    def worst(self):
        if self.__sense == "max":
            return self.sort[0]

        return self.sort[-1]
