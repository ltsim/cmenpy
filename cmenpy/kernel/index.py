import typing

import numpy as np
import numpy.typing as npt

from cmenpy.hints import NDArrayType
from cmenpy.hints.option import SenseType
from cmenpy.kernel.mask import KernelMask


class BufferIndex:
    def __init__(self, buffer: NDArrayType, mask: KernelMask, sense: SenseType):
        self.__buffer = buffer
        self.__sense = sense
        self.__mask = mask

    def __iter__(self) -> typing.Iterator[int]:
        pop_size, _ = self.__buffer.shape

        return iter(range(pop_size))

    def __invert__(self) -> npt.NDArray[np.intp]:
        return np.where(~np.isnan(self.__buffer[:, 0]))[0].astype(int)

    @property
    def sort(self) -> npt.NDArray[np.intp]:
        _idx = np.argsort(self.__buffer[:, 0])

        if self.__sense == "max":
            return _idx[::-1]

        return _idx

    @property
    def best(self) -> np.intp:
        if self.__sense == "max":
            return typing.cast(np.intp, self.sort[-1])

        return typing.cast(np.intp, self.sort[0])

    @property
    def worst(self) -> np.intp:
        if self.__sense == "max":
            return typing.cast(np.intp, self.sort[0])

        return typing.cast(np.intp, self.sort[-1])
