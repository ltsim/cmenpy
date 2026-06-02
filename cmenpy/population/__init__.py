import typing

import numpy as np

from cmenpy.agent import Agent, VirtualAgent, MemoryAgent
from cmenpy.population.operations.swap import SwapOperation
from cmenpy.population.functools.utils import sort_agents
from cmenpy.population.view import ViewPopulation, BasePopulation
from cmenpy.target import TargetFunction
from cmenpy.types import NDArrayType


class PopulationManager:
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

    def __iter__(self):
        return iter(self.all)

    def __repr__(self):
        return f"Population(size={self.size})"

    def __len__(self):
        return len(self.__agents)

    def remove(self, i: int):
        founds = [*filter(lambda a: a.id == i, self.__agents)]

        if not len(founds) > 0:
            raise ValueError("Agent not found.")

        agent = founds[0]

        self.__mask[i] = False
        self.__buffer[i, :] = np.nan
        self.__agents.remove(agent)

    def append(self, solution: NDArrayType):
        founds = [i for i, x in enumerate(self.__mask) if not x]

        if not len(founds) > 0:
            raise ValueError("Max agents in memory.")

        i = founds[0]

        self.__mask[i] = True
        self.__agents.append(MemoryAgent(self.__buffer, i))
        self.__buffer[i, 1:] = solution
        self.__buffer[i, 0] = np.apply_along_axis(self.__target, 0, solution)

        return VirtualAgent(self.__buffer, i)

    def __iadd__(self, other: NDArrayType):
        self.append(other)

    def __isub__(self, other: int):
        self.remove(other)

    def __invert__(self):
        return self.__buffer[self.__mask, 1:].copy()

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
    def best(self):
        b_pop = sort_agents(self.__agents)[0]

        return VirtualAgent(self.__buffer, b_pop.id)

    @property
    def worst(self):
        w_pop = sort_agents(self.__agents)[-1]

        return VirtualAgent(self.__buffer, w_pop.id)

    @property
    def free_space(self):
        return len(self.__agents) < self.max

    @property
    def min(self):
        return self.__min_pop

    @property
    def max(self):
        return self.__max_pop

    @property
    def size(self):
        return len(self.__agents)

    @property
    def all(self) -> list[VirtualAgent]:
        return [VirtualAgent(self.__buffer, i) for i, a in enumerate(self.__agents)]

    @property
    def sorted(self) -> list[VirtualAgent]:
        return sort_agents(self.all)

    @property
    def fitnesses(self):
        return self.__buffer[self.__mask, 0].copy()

    @property
    def solutions(self):
        return self.__buffer[self.__mask, 1:].copy()

    @property
    def view(self) -> ViewPopulation:
        buffer = self.__buffer.copy()

        return ViewPopulation(
            mask=self.__mask,
            buffer=buffer,
            target=self.__target,
        )

    @view.setter
    def view(self, v: ViewPopulation):
        self.__buffer[self.__mask, 1:] = v.solutions
        self.__buffer[self.__mask, 0] = v.fitnesses

    @property
    def swap(self) -> SwapOperation:
        return SwapOperation(self.__buffer)
