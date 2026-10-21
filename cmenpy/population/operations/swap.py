"""Row swap writer operator (``pop.swap``)."""

from cmenpy.casting import unwrap_array
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
        # FIXME: the broadcast below overwrites *every* row of the buffer with
        # the single saved row (`raw[key]` is 1-D for an int key), which does
        # not match the documented "move the displaced row to the end"
        # behavior. Suspected bug; the operator is unused by the test suite
        # and the examples, so it is left as-is pending a decision.
        cp_mem_buff = self.__buffer.raw[key].copy()
        self.__buffer.raw[key] = unwrap_array(value)
        self.__buffer.raw[:] = cp_mem_buff

    def __repr__(self) -> str:
        """Return a compact representation of the operator."""
        return "Swap<WriterOperator>()"
