import typing

import numpy as np

from cmenpy.agent import Agent, VirtualAgent, MemoryAgent
from cmenpy.population.utils import sorted_population
from cmenpy.target import TargetFunction
from cmenpy.types import NDArrayType


class Population:
    def __init__(
        self,
        buffer: NDArrayType,
        target: TargetFunction,
        agents: list[Agent],
        r_pop: tuple[int, int],
        d_class: typing.Type[Agent] | None = None,
    ):
        if d_class is None:
            d_class = VirtualAgent

        n_pop = len(agents)

        self.__min_pop, self.__max_pop = r_pop
        self.__buffer = buffer
        self.__target = target
        self.__agents = agents
        self.__mask = list(True for _ in range(0, n_pop)) + list(
            False for _ in range(n_pop, self.__max_pop)
        )

        self.__d_class = d_class

    def __len__(self):
        return len(self.__agents)

    def __getitem__(self, i: int) -> VirtualAgent:
        a = self.__agents[i]

        return VirtualAgent(self.__buffer, a.id)

    def __setitem__(self, i, value):
        if isinstance(value, Agent):
            self.__agents[i] = value
        elif any(isinstance(value, n) for n in (list, tuple, np.ndarray)):
            self.__buffer[i, :] = value

    def remove(self, i: int):
        founds = [*filter(lambda a: a.id == i, self.__agents)]

        if not len(founds) > 0:
            raise ValueError("Agent not found.")

        agent = founds[0]

        self.__mask[i] = False
        self.__buffer[i, :] = np.nan
        self.__agents.remove(agent)

    def append(self, solution=None):
        founds = [i for i, x in enumerate(self.__mask) if not x]

        if not len(founds) > 0:
            raise ValueError("Max agents in memory.")

        i = founds[0]

        self.__mask[i] = True
        self.__agents.append(MemoryAgent(self.__buffer, i))
        self.__buffer[i, 1:] = solution
        self.__buffer[i, 0] = np.apply_along_axis(self.__target, 0, solution)

        return VirtualAgent(self.__buffer, i)

    def __invert__(self):
        return self.__buffer[self.__mask, 1:].copy()

    def __matmul__(self, other):
        buff = self.__buffer.copy()
        buff[self.__mask, 1:] = other
        buff[self.__mask, 0] = np.apply_along_axis(
            self.__target, 1, self.__buffer[self.__mask, 2:]
        )

        return buff[self.__mask, 2:]

    def __imatmul__(self, other):
        if self.size > 0:
            self.__buffer[self.__mask, 1:] = other
            self.__buffer[self.__mask, 0] = np.apply_along_axis(
                self.__target, 1, self.__buffer[self.__mask, 2:]
            )

        return self

    @property
    def best(self):
        b_pop = sorted_population(self.__agents)[0]

        return VirtualAgent(self.__buffer, b_pop.id)

    @property
    def worst(self):
        w_pop = sorted_population(self.__agents)[-1]

        return VirtualAgent(self.__buffer, w_pop.id)

    @property
    def free_space(self):
        return len(self.__agents) < self.max_pop

    @property
    def min_pop(self):
        return self.__min_pop

    @property
    def max_pop(self):
        return self.__max_pop

    @property
    def size(self):
        return len(self.__agents)

    @property
    def all(self) -> list[VirtualAgent]:
        return [VirtualAgent(self.__buffer, i) for i, a in enumerate(self.__agents)]

    @property
    def sorted(self) -> list[VirtualAgent]:
        return sorted_population(self.all)


def create_range_population(n_pop: int, r_pop: typing.Optional[tuple[int, int]]):
    if r_pop is None:
        r_pop = n_pop, n_pop

    min_pop, max_pop = r_pop

    return min_pop, max_pop, r_pop
