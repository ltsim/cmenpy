import numpy as np

from cmenpy.agent import MemoryAgent
from cmenpy.target import TargetFunction
from cmenpy.types import NDArrayType
from cmenpy.types.option import SenseType


class GlobalPopulationProperty:
    def __init__(
        self,
        mask: list[bool],
        buffer: NDArrayType,
        target: TargetFunction,
        sense: SenseType = "min",
    ):
        self.__mask = mask
        self.__buffer = buffer
        self.__target = target
        self.__sense = sense

    @property
    def fitnesses(self):
        return self.__buffer[self.__mask, 0].copy()

    @property
    def solutions(self):
        return self.__buffer[self.__mask, 1:].copy()

    @property
    def best(self) -> MemoryAgent:
        idx = np.argsort(self.__buffer[:, 0])

        if self.__sense == "min":
            return MemoryAgent(self.__buffer, idx[0], self.__target)

        return MemoryAgent(self.__buffer, idx[-1], self.__target)

    @property
    def worst(self) -> MemoryAgent:
        idx = np.argsort(self.__buffer[:, 0])

        if self.__sense == "min":
            return MemoryAgent(self.__buffer, idx[-1], self.__target)

        return MemoryAgent(self.__buffer, idx[0], self.__target)

    @property
    def idx_sort(self):
        idx = np.argsort(self.__buffer[self.__mask, 0])

        if self.__sense == "max":
            return idx[::-1]
        else:
            return idx
