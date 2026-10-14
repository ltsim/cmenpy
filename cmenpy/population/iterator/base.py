import typing

import numpy as np
import numpy.typing as npt

from cmenpy.kernel import KernelBuffer


class BaseIterator:
    """Iteration and ranking over the population's active rows.

    Iterating a population (or its ``~`` form) yields the active row
    indices; ``best``/``worst``/``sort`` expose the sense-aware ranking.

    Parameters
    ----------
    buffer : KernelBuffer
        The buffer to iterate.

    Attributes
    ----------
    best : numpy.intp
        Row index of the best individual.
    worst : numpy.intp
        Row index of the worst individual.
    sort : numpy.ndarray
        Row indices ordered best-first.
    """

    def __init__(self, buffer: KernelBuffer):
        self.__buffer = buffer

    @property
    def best(self) -> np.intp:
        return self.__buffer.idx.best

    @property
    def worst(self) -> np.intp:
        return self.__buffer.idx.worst

    @property
    def sort(self) -> npt.NDArray[np.intp]:
        return self.__buffer.idx.sort

    def __iter__(self) -> typing.Iterator[int]:
        return iter(self.__buffer.idx)

    def __invert__(self) -> typing.Iterator[np.intp]:
        return iter(~self.__buffer.idx)
