"""Row swap writer operator (``pop.swap``)."""

from cmenpy.hints import NDArrayType
from cmenpy.kernel import KernelBuffer
from cmenpy.population.operations.stream import WriterOperator


class SwapOperation(WriterOperator):
    """Row swap writer: ``pop.swap[i] = row`` (moves the old row to the end).

    Args:
        buffer: The backing buffer.
    """

    def __init__(self, buffer: KernelBuffer):
        """Initialize the writer over ``buffer``."""
        super().__init__(buffer.shape)
        self.__buffer = buffer

    def __setitem__(self, key: int | slice, value: NDArrayType) -> None:
        """Write ``value`` at ``key`` and move the displaced row to the end."""
        cp_mem_buff = self.__buffer.raw[key].copy()
        self.__buffer.raw[key] = value
        self.__buffer.raw[:] = cp_mem_buff

    def __repr__(self) -> str:
        """Return a compact representation of the operator."""
        return "Swap<WriterOperator>()"
