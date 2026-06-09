import numpy as np

from cmenpy.kernel import KernelBuffer
from cmenpy.hints import NDArrayType


class DynamicBase:
    def __init__(self, buffer: KernelBuffer):
        self.__buffer = buffer

    @property
    def free_space(self):
        return sum(self.__buffer.mask) < self.__buffer.size.max

    def remove(self, idx: int) -> None:
        self.__buffer.mask[idx] = False
        self.__buffer.raw_data[idx, :] = np.nan

    def insert(self, x: NDArrayType) -> int:
        founds = [i for i, x in enumerate(self.__buffer.mask.founds) if not x]

        if not len(founds) > 0:
            raise ValueError("Max agents in memory.")

        idx = founds[0]

        self.__buffer.mask[idx] = True
        self.__buffer.apply(x, idx)

        return idx
