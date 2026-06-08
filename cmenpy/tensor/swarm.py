from cmenpy.dynamic.base import DynamicBase
from cmenpy.iterator.base import BaseIterator
from cmenpy.kernel import KernelBuffer
from cmenpy.operations import (
    ExtractOperation,
    AssignOperator,
    ComputeOperator,
    SwapOperation,
)
from cmenpy.operations.access import AccessOperator
from cmenpy.operations.base import OperationBase
from cmenpy.properties.base import BaseProperty
from cmenpy.types.option import SenseType


class TensorSwarm(OperationBase, BaseProperty, BaseIterator, DynamicBase):
    def __init__(
        self,
        buffer: KernelBuffer,
        sense: SenseType = "min",
    ):
        BaseProperty.__init__(self, buffer)
        BaseIterator.__init__(self, buffer)
        DynamicBase.__init__(self, buffer)

        self.__buffer = buffer
        self.__sense = sense

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
