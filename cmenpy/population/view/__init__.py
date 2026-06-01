import typing

import numpy as np

from cmenpy.agent import Agent, VirtualAgent, MemoryAgent
from cmenpy.population.utils import sorted_population
from cmenpy.target import TargetFunction
from cmenpy.types import NDArrayType


class ViewPopulation:
    def __init__(
        self,
        buffer: NDArrayType,
        target: TargetFunction,
        agents: list[Agent],
    ):
        self.__buffer = buffer
        self.__target = target
        self.__agents = agents

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.__buffer[key]

    def __setitem__(self, key, value):
        pass

    @property
    def all(self) -> list[Agent]:
        return self.__agents

    @property
    def F(self) -> NDArrayType:
        return self.__buffer[1:]

    @property
    def S(self) -> NDArrayType:
        return self.__buffer[:1]
