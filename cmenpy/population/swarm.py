import typing

from cmenpy.kernel import KernelBuffer
from cmenpy.population.agent import AgentGenerator
from cmenpy.population.attributes.base import BaseTensorAttributes
from cmenpy.population.operations.base import OperationBase
from cmenpy.population.iterator.base import BaseIterator
from cmenpy.population.dynamic.base import DynamicBase
from cmenpy.population.operations import (
    ExtractOperation,
    AssignOperator,
    ComputeOperator,
    SwapOperation,
    AccessOperator,
)


class PopulationSwarm(
    OperationBase, BaseTensorAttributes, BaseIterator, DynamicBase, AgentGenerator
):
    def __init__(self, buffer: KernelBuffer) -> None:
        BaseTensorAttributes.__init__(self, buffer)
        BaseIterator.__init__(self, buffer)
        DynamicBase.__init__(self, buffer)
        AgentGenerator.__init__(self, buffer)

        self.__buffer = buffer

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
    def access(self) -> AccessOperator:
        return AccessOperator(self.__buffer)
