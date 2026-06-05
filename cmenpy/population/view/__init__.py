import typing

from cmenpy.agent import Agent
from cmenpy.population import PopulationOperationBase
from cmenpy.population.agents import (
    PopulationGenerator,
    AgentMutableCollection,
    AgentMutable,
)
from cmenpy.population.operations.assign import AssignOperator
from cmenpy.population.operations.compute import ComputeOperator
from cmenpy.population.operations.swap import SwapOperation
from cmenpy.population.view.base import ViewBase
from cmenpy.target import TargetFunction
from cmenpy.types import NDArrayType
from cmenpy.types.option import SenseType


class ViewPopulation(ViewBase, PopulationOperationBase, PopulationGenerator):
    def __init__(
        self,
        mask: list[bool],
        buffer: NDArrayType,
        target: TargetFunction,
        r_pop: tuple[int, int],
        sense: SenseType,
        d_class: typing.Type[Agent],
    ):
        self.__mask = mask
        self.__buffer = buffer
        self.__target = target
        self.__r_pop = r_pop
        self.__sense = sense
        self.__d_class = d_class

    def __repr__(self):
        return "View()"

    @property
    def size(self):
        return self.__buffer.shape[0]

    @property
    def swap(self) -> SwapOperation:
        return SwapOperation(self.__buffer)

    @property
    def compute(self) -> ComputeOperator:
        return ComputeOperator(self.__mask, self.__buffer, self.__target)

    @property
    def assign(self) -> AssignOperator:
        return AssignOperator(self.__mask, self.__buffer, self.__target)

    @property
    def agents(self) -> AgentMutableCollection:
        return AgentMutable(
            self.__mask,
            self.__buffer,
            self.__target,
            self.__r_pop,
            self.__sense,
            self.__d_class,
        )
