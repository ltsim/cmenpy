"""Iteration and sense-aware ranking over the population's active rows."""

import typing

import numpy as np
import numpy.typing as npt

from cmenpy.kernel import KernelBuffer


class BaseIterator:
    """Iteration and ranking over the population's active rows.

    Iterating a population (or its ``~`` form) yields the active row
    indices; ``best``/``worst``/``sort`` expose the sense-aware ranking.

    Args:
        buffer: The buffer to iterate.

    Attributes:
        best: Row index of the best individual.
        worst: Row index of the worst individual.
        sort: Row indices ordered best-first.
    """

    def __init__(self, buffer: KernelBuffer):
        """Initialize the iterator over ``buffer``."""
        self.__buffer = buffer

    @property
    def best(self) -> np.intp:
        """Row index of the best individual."""
        return self.__buffer.idx.best

    @property
    def worst(self) -> np.intp:
        """Row index of the worst individual."""
        return self.__buffer.idx.worst

    @property
    def sort(self) -> npt.NDArray[np.intp]:
        """Row indices ordered best-first."""
        return self.__buffer.idx.sort

    def __iter__(self) -> typing.Iterator[int]:
        """Iterate over all row indices of the buffer."""
        return iter(self.__buffer.idx)

    def __invert__(self) -> typing.Iterator[np.intp]:
        """Iterate over the indices of rows with a scored fitness."""
        return iter(~self.__buffer.idx)
