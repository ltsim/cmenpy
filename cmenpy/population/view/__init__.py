import typing

from cmenpy.population import OperationBase, ExtractOperation
from cmenpy.population.agent import Agent
from cmenpy.operations import AssignOperator
from cmenpy.operations import ComputeOperator
from cmenpy.operations import SwapOperation
from cmenpy.population.people import (
    PeopleGenerator,
    PeopleMutableCollection,
    PeopleMutable,
)
from cmenpy.population.view.base import ViewBase
from cmenpy.target import TargetFunction
from cmenpy.types import NDArrayType
from cmenpy.types.option import SenseType


class ViewPopulation(ViewBase, OperationBase, PeopleGenerator):
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
