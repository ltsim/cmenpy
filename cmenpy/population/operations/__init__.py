from cmenpy.kernel import KernelBuffer
from cmenpy.population.operations.access import AccessOperator
from cmenpy.population.operations.assign import AssignOperator
from cmenpy.population.operations.compute import ComputeOperator
from cmenpy.population.operations.extract import ExtractOperation
from cmenpy.population.operations.swap import SwapOperation


class StreamOperations:
    """The ``compute``/``assign``/``access``/``extract``/``swap`` operators.

    Each property returns a fresh operator bound to the buffer; they are
    the verbs of the population DSL (``pop.compute << values``).

    Parameters
    ----------
    buffer : KernelBuffer
        The backing buffer.

    Attributes
    ----------
    swap : SwapOperation
        Row swap writer.
    compute : ComputeOperator
        Write-and-score operator.
    assign : AssignOperator
        Raw row writer (no scoring).
    extract : ExtractOperation
        Solution-column reader.
    access : AccessOperator
        Whole-row reader.
    """

    def __init__(self, buffer: KernelBuffer):
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


__all__ = [
    "StreamOperations",
]
