import typing

import numpy as np

from cmenpy.population.agent import Agent, ImmutableAgent
from cmenpy.population.base import PopulationOperationBase
from cmenpy.operations import AssignOperator
from cmenpy.operations import ComputeOperator
from cmenpy.operations import ExtractOperation
from cmenpy.operations import SwapOperation
from cmenpy.population.people import (
    PeopleGenerator,
    PeopleMutableCollection,
    PeopleMutable,
)
from cmenpy.population.view import ViewPopulation, ViewBase
from cmenpy.target import TargetFunction
from cmenpy.types import NDArrayType
from cmenpy.types.option import SenseType


class PopulationManager(PopulationOperationBase, PeopleGenerator):
    def __init__(
        self,
        buffer: NDArrayType,
        target: TargetFunction,
        size: int,
        r_pop: tuple[int, int],
        sense: SenseType = "min",
        d_class: typing.Type[Agent] | None = None,
    ):
        if d_class is None:
            d_class = ImmutableAgent

        n_pop = size

        self.__r_pop = r_pop
        self.__buffer = buffer
        self.__target = target

        max_pop = self.__r_pop[1]
        self.__mask = list(True for _ in range(0, n_pop)) + list(
            False for _ in range(n_pop, max_pop)
        )

        self.__sense = sense
        self.__d_class = d_class

    def __repr__(self):
        return f"Population()"

    def __len__(self):
        return sum(self.__mask)

    @property
    def size(self):
        return sum(self.__mask)

    @property
    def view(self) -> ViewPopulation:
        buffer = np.copy(self.__buffer)

        return ViewPopulation(
            mask=self.__mask,
            buffer=buffer,
            target=self.__target,
            r_pop=self.__r_pop,
            sense=self.__sense,
            d_class=self.__d_class,
        )

    @view.setter
    def view(self, v: ViewPopulation):
        self.__buffer[self.__mask, 1:] = v.agents.solutions
        self.__buffer[self.__mask, 0] = v.agents.fitnesses

    @property
    def swap(self) -> SwapOperation:
        return SwapOperation(self.__buffer)

    @property
    def compute(self) -> ComputeOperator:
        return ComputeOperator(self.__mask, self.__buffer, self.__target)

    @property
    def assign(self) -> AssignOperator:
        return AssignOperator(self.__mask, self.__buffer)

    @property
    def extract(self) -> ExtractOperation:
        return ExtractOperation(self.__mask, self.__buffer)

    @property
    def agents(self) -> PeopleMutableCollection:
        return PeopleMutable(
            self.__mask,
            self.__buffer,
            self.__target,
            self.__r_pop,
            self.__sense,
            self.__d_class,
        )
