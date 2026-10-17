"""The population stream operators (``compute``/``assign``/``access``/...)."""

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

    Args:
        buffer: The backing buffer.

    Attributes:
        swap: Row swap writer.
        compute: Write-and-score operator.
        assign: Raw row writer (no scoring).
        extract: Solution-column reader.
        access: Whole-row reader.
    """

    def __init__(self, buffer: KernelBuffer):
        """Initialize the operators over ``buffer``."""
        self.__buffer = buffer

    @property
    def swap(self) -> SwapOperation:
        """Row swap writer."""
        return SwapOperation(self.__buffer)

    @property
    def compute(self) -> ComputeOperator:
        """Write-and-score operator."""
        return ComputeOperator(self.__buffer)

    @property
    def assign(self) -> AssignOperator:
        """Raw row writer (no scoring)."""
        return AssignOperator(self.__buffer)

    @property
    def extract(self) -> ExtractOperation:
        """Solution-column reader."""
        return ExtractOperation(self.__buffer)

    @property
    def access(self) -> AccessOperator:
        """Whole-row reader."""
        return AccessOperator(self.__buffer)


__all__ = [
    "StreamOperations",
]
