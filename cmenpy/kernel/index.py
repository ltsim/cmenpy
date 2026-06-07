import numpy as np

from cmenpy.types import NDArrayType
from cmenpy.types.option import SenseType


class BufferIndex:
    def __init__(self, buffer: NDArrayType, sense: SenseType):
        self.__buffer = buffer
        self.__sense = sense

    @property
    def sort(self):
        return np.argsort(self.__buffer[:, 0])

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
