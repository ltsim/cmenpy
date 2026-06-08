from cmenpy.kernel import KernelBuffer
from cmenpy.operations.base import OperationBase
from cmenpy.properties.base import BaseProperty
from cmenpy.types.option import SenseType
from cmenpy.operations import (
    ExtractOperation,
    AssignOperator,
    ComputeOperator,
    SwapOperation,
)


class TensorSwarm(OperationBase, BaseProperty):
    def __init__(
        self,
        buffer: KernelBuffer,
        sense: SenseType = "min",
    ):
        super(BaseProperty).__init__(buffer)
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
