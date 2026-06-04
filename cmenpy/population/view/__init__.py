import numpy as np

from cmenpy.agent import Agent, VirtualAgent
from cmenpy.population.operations.assign import AssignOperator
from cmenpy.population.operations.compute import ComputeOperator
from cmenpy.population.operations.swap import SwapOperation
from cmenpy.population.view.base import BasePopulation
from cmenpy.target import TargetFunction
from cmenpy.types import NDArrayType


class ViewPopulation(BasePopulation):
    def __init__(
        self,
        mask: list[int],
        buffer: NDArrayType,
        target: TargetFunction,
    ):
        self.__mask = mask
        self.__buffer = buffer
        self.__target = target

    def __iter__(self):
        return iter(self.all)

    def __repr__(self):
        return ""

    @property
    def size(self):
        return self.__buffer.shape[0]

    @property
    def all(self) -> list[Agent]:
        return [VirtualAgent(self.__buffer, i) for i in range(self.size)]

    @property
    def solutions(self) -> NDArrayType:
        return self.__buffer[self.__mask, 1:].copy()

    @property
    def fitnesses(self) -> NDArrayType:
        return self.__buffer[self.__mask, :1].reshape(-1).copy()

    @property
    def swap(self) -> SwapOperation:
        return SwapOperation(self.__buffer)

    @property
    def compute(self) -> ComputeOperator:
        return ComputeOperator(self.__mask, self.__buffer, self.__target)

    @property
    def assign(self) -> AssignOperator:
        return AssignOperator(self.__mask, self.__buffer, self.__target)
