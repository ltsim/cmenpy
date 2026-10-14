import typing

import numpy as np

from cmenpy.hints import NDArrayType
from cmenpy.kernel import KernelBuffer
from cmenpy.population.agent.base import BaseAgent


class ImmutableAgent(BaseAgent):
    """A snapshot agent: copies its row out of the buffer at construction.

    Parameters
    ----------
    n : int
        Row index to snapshot.
    buffer : KernelBuffer
        The buffer to read from.
    """

    def __init__(self, n: int, buffer: KernelBuffer) -> None:
        self.__i = n
        self.__buffer = buffer.raw[n, :].copy()

    def __iter__(self) -> typing.Iterator[np.number[typing.Any]]:
        return iter(self.__buffer[1:].copy())

    def __getitem__(self, key: int) -> np.number[typing.Any]:
        return typing.cast(np.number[typing.Any], self.__buffer[key + 1])

    @property
    def i(self) -> int:
        return self.__i

    @property
    def solution(self) -> NDArrayType:
        return self.__buffer[1:]

    @property
    def fitness(self) -> np.number[typing.Any]:
        return typing.cast(np.number[typing.Any], self.__buffer[0])

    def __float__(self) -> float:
        return float(self.__buffer[0])

    @property
    def x(self) -> NDArrayType:
        return self.__buffer
