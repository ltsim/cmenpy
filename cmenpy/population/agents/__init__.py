import abc
import typing

import numpy as np

from cmenpy.agent import Agent, MutableAgent
from cmenpy.population.agents.collection.mutable import AgentMutableCollection
from cmenpy.population.agents.collection.sequence import AgentSequenceCollection
from cmenpy.population.agents.properties import GlobalPopulationProperty
from cmenpy.target import TargetFunction
from cmenpy.types import NDArrayType
from cmenpy.types.option import SenseType


class AgentSequence(AgentSequenceCollection, GlobalPopulationProperty):
    def __init__(
        self,
        mask: list[bool],
        buffer: NDArrayType,
        target: TargetFunction,
        agents: typing.List[Agent],
    ):
        super().__init__(mask, buffer, target)
        self.__buffer = buffer
        self.__agents = agents

    def __getitem__(self, index) -> Agent | typing.List[Agent]:
        return self.__agents[index]

    def __iter__(self) -> typing.Iterator[Agent]:
        return iter(self.__agents)

    def __len__(self) -> int:
        return len(self.__agents)


class AgentMutable(AgentMutableCollection, GlobalPopulationProperty):
    def __init__(
        self,
        mask: list[bool],
        buffer: NDArrayType,
        target: TargetFunction,
        r_pop: tuple[int, int],
        sense: SenseType,
        d_class: typing.Type[Agent],
    ):
        super().__init__(mask, buffer, target)
        self.__mask = mask
        self.__buffer = buffer
        self.__target = target
        self.__r_pop = r_pop
        self.__sense = sense
        self.__d_class = d_class

    def append(self, value: NDArrayType) -> None:
        founds = [i for i, x in enumerate(self.__mask) if not x]

        if not len(founds) > 0:
            raise ValueError("Max agents in memory.")

        i = founds[0]

        self.__mask[i] = True
        self.__buffer[i, 1:] = value
        self.__buffer[i, 0] = np.apply_along_axis(self.__target, 0, value)

    def pop(self, index: int = -1) -> None:
        self.__mask[index] = True
        self.__buffer[index, :] = np.nan

    def insert(self, index: int, value: NDArrayType) -> None:
        self.__mask[index] = True
        self.__buffer[index, 1:] = value
        self.__buffer[index, 0] = np.apply_along_axis(self.__target, 0, value)

    def __getitem__(self, index: int) -> MutableAgent:
        return MutableAgent(self.__buffer, index, self.__target)

    def __setitem__(self, index: int, value: NDArrayType) -> None:
        self.insert(index, value)

    def __delitem__(self, index: int) -> None:
        self.pop(index)

    def __len__(self):
        return sum(self.__mask)

    @property
    def min(self):
        return self.__r_pop[0]

    @property
    def max(self):
        return self.__r_pop[1]

    @property
    def size(self) -> int:
        return sum(self.__mask)

    @property
    def free_space(self):
        return sum(self.__mask) < self.max

    @property
    def sort(self) -> AgentSequence:
        idx = np.argsort(self.__buffer[self.__mask, 0])
        agents = [MutableAgent(self.__buffer, i, self.__target) for i in idx]

        if self.__sense == "max":
            agents = agents[::-1]

        return AgentSequence(
            self.__mask,
            self.__buffer,
            self.__target,
            agents,
        )


class PopulationGenerator(abc.ABC):
    @property
    @abc.abstractmethod
    def agents(self) -> AgentMutableCollection: ...
