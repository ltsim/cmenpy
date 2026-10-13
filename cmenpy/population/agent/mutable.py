import typing

import numpy as np

from cmenpy.hints import NDArrayType
from cmenpy.kernel import KernelBuffer
from cmenpy.population.agent.base import BaseAgent


class MutableAgent(BaseAgent):
    def __init__(self, n: int, buffer: KernelBuffer) -> None:
        self.__i = n
        self.__buffer = buffer

    def __iter__(self) -> typing.Iterator[np.number[typing.Any]]:
        return iter(self.__buffer.raw[self.__i, 1:].copy())

    def __getitem__(self, key: int) -> np.number[typing.Any]:
        return typing.cast(np.number[typing.Any], self.__buffer.raw[self.__i, key + 1])

    @property
    def i(self) -> int:
        return self.__i

    @property
    def solution(self) -> NDArrayType:
        return self.__buffer.raw[self.i, 1:]

    @solution.setter
    def solution(self, value: NDArrayType) -> None:
        self.__buffer.apply(value, self.__i)

    @property
    def fitness(self) -> np.number[typing.Any]:
        return typing.cast(np.number[typing.Any], self.__buffer.raw[self.i, 0])

    def __float__(self) -> float:
        return float(self.__buffer.raw[self.i, 0])

    @property
    def x(self) -> NDArrayType:
        return self.__buffer.raw[self.i, :]
