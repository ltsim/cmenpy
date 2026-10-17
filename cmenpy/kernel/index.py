"""Best/worst/sort row lookups over the buffer's fitness column."""

import typing

import numpy as np
import numpy.typing as npt

from cmenpy.hints import NDArrayType
from cmenpy.hints.option import SenseType
from cmenpy.kernel.mask import KernelMask


class BufferIndex:
    """Ranking and best/worst lookups over the buffer's fitness column.

    Args:
        buffer: The raw ``(pop, ndim + 1)`` buffer (fitness in column 0).
        mask: The active-row mask.
        sense: ``"min"`` or ``"max"``; selects the direction of "best".

    Attributes:
        sort: Row indices ordered best-first (respecting ``sense``).
        best: Row index of the best individual.
        worst: Row index of the worst individual.
    """

    def __init__(self, buffer: NDArrayType, mask: KernelMask, sense: SenseType):
        """Initialize the lookups over ``buffer`` and ``mask``."""
        self.__buffer = buffer
        self.__sense = sense
        self.__mask = mask

    def __iter__(self) -> typing.Iterator[int]:
        """Iterate over all row indices of the buffer."""
        pop_size, _ = self.__buffer.shape

        return iter(range(pop_size))

    def __invert__(self) -> npt.NDArray[np.intp]:
        """Return the indices of rows whose fitness is set (not NaN)."""
        return np.where(~np.isnan(self.__buffer[:, 0]))[0].astype(int)

    @property
    def sort(self) -> npt.NDArray[np.intp]:
        """Row indices ordered best-first (respecting ``sense``)."""
        _idx = np.argsort(self.__buffer[:, 0])

        if self.__sense == "max":
            return _idx[::-1]

        return _idx

    @property
    def best(self) -> np.intp:
        """Row index of the best individual."""
        if self.__sense == "max":
            return typing.cast(np.intp, self.sort[-1])

        return typing.cast(np.intp, self.sort[0])

    @property
    def worst(self) -> np.intp:
        """Row index of the worst individual."""
        if self.__sense == "max":
            return typing.cast(np.intp, self.sort[0])

        return typing.cast(np.intp, self.sort[-1])
