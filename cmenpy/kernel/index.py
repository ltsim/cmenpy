import numpy as np

from cmenpy.kernel.mask import KernelMask
from cmenpy.hints import NDArrayType
from cmenpy.hints.option import SenseType


class BufferIndex:
    def __init__(self, buffer: NDArrayType, mask: KernelMask, sense: SenseType):
        self.__buffer = buffer
        self.__sense = sense
        self.__mask = mask

    @property
    def sort(self):
        idx = np.argsort(self.__buffer[:, 0])
        idx = idx[~np.isnan(self.__buffer[idx, 0])]

        if self.__sense == "max":
            return idx[::-1]

        return idx

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
