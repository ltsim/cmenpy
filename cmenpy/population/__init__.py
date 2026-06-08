import typing

from cmenpy.kernel import KernelBuffer
from cmenpy.operations import AssignOperator
from cmenpy.operations import ComputeOperator
from cmenpy.operations import ExtractOperation
from cmenpy.operations import SwapOperation
from cmenpy.operations.base import OperationBase
from cmenpy.population.agent import Agent, ImmutableAgent
from cmenpy.population.people import (
    PeopleGenerator,
    PeopleMutableCollection,
    PeopleMutable,
)
from cmenpy.population.view import ViewPopulation, ViewBase
from cmenpy.properties.base import BaseProperty
from cmenpy.types.option import SenseType


class PopulationSwarm(OperationBase, PeopleGenerator, BaseProperty):
    def __init__(
        self,
        buffer: KernelBuffer,
        sense: SenseType = "min",
        d_class: typing.Optional[typing.Type[Agent]] = None,
    ):
        if d_class is None:
            d_class = ImmutableAgent

        super().__init__(buffer)

        self.__buffer = buffer
        self.__sense = sense
        self.__d_class = d_class

    def __repr__(self):
        return f"Population()"

    @property
    def swap(self) -> SwapOperation:
        return SwapOperation(self.__buffer)

    @property
    def compute(self) -> ComputeOperator:
        return ComputeOperator(self.__buffer)

    @property
    def assign(self) -> AssignOperator:
        return AssignOperator(self.__buffer)

    @property
    def extract(self) -> ExtractOperation:
        return ExtractOperation(self.__buffer)

    @property
    def agents(self) -> PeopleMutableCollection:
        raise NotImplementedError("PopulationManager does not implement this method")
