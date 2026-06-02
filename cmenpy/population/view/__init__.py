import typing

import numpy as np

from cmenpy.agent import Agent, VirtualAgent, MemoryAgent
from cmenpy.population.utils import sorted_population
from cmenpy.population.view.base import BasePopulation
from cmenpy.target import TargetFunction
from cmenpy.types import NDArrayType


class ViewPopulation(BasePopulation):
    def __init__(
        self,
        mask: list[int],
        agents: list[Agent],
        buffer: NDArrayType,
        target: TargetFunction,
    ):
        self.__mask = mask
        self.__agents = agents
        self.__buffer = buffer
        self.__target = target

    def __iter__(self):
        return iter(self.all)

    def __repr__(self):
        return ""

    def __getitem__(self, key):
        return self.__buffer[self.__mask, :][key].copy()

    def __setitem__(self, key, value):
        if not self.__mask[key]:
            return

        self.__buffer[key, 0] = self.__target.f(value)
        self.__buffer[key, 1:] = value

    def __matmul__(self, other):
        buff = self.__buffer.copy()
        buff[self.__mask, 1:] = other
        buff[self.__mask, 0] = np.apply_along_axis(
            self.__target, 1, self.__buffer[self.__mask, 1:]
        )

        return buff[self.__mask, 2:]

    def __imatmul__(self, other):
        if self.size > 0:
            self.__buffer[self.__mask, 1:] = other
            self.__buffer[self.__mask, 0] = np.apply_along_axis(
                self.__target, 1, self.__buffer[self.__mask, 1:]
            )

        return self

    @property
    def size(self):
        return self.__buffer.shape[0]

    @property
    def all(self) -> list[Agent]:
        return [VirtualAgent(self.__buffer, i) for i, a in enumerate(self.__agents)]

    @property
    def solutions(self) -> NDArrayType:
        return self.__buffer[self.__mask, 1:].copy()

    @property
    def fitnesses(self) -> NDArrayType:
        return self.__buffer[self.__mask, :1].reshape(-1).copy()
