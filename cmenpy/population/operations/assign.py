"""Raw row writer operator (``pop.assign``), without scoring."""

from cmenpy.hints import NDArrayType
from cmenpy.kernel import KernelBuffer
from cmenpy.population.operations.stream.vectorizable import (
    PipelineOperator,
    VectorizableOperator,
)


class AssignPipeline(PipelineOperator):
    """Per-row assign target: ``pop.assign[i] << row``.

    Args:
        idx: The row to write.
        buffer: The backing buffer.
    """

    def __init__(self, idx: int, buffer: KernelBuffer):
        """Bind the pipeline to row ``idx`` of ``buffer``."""
        self.__idx = idx
        self.__buffer = buffer

    def __lshift__(self, value: NDArrayType) -> None:
        """Write ``value`` (fitness + solution) into the bound row."""
        self.__buffer.raw[self.__idx] = value


class AssignOperator(VectorizableOperator):
    """Raw row writer (no scoring): ``pop.assign << rows`` / ``pop.assign[i] << row``.

    Args:
        buffer: The backing buffer.

    Raises:
        IndexError: If the assigned block's shape does not match the buffer.
    """

    def __init__(
        self,
        buffer: KernelBuffer,
    ):
        """Initialize the writer over ``buffer``."""
        super().__init__(buffer.shape)
        self.__buffer = buffer

    def __lshift__(self, value: NDArrayType) -> "VectorizableOperator":
        """Overwrite the whole buffer with ``value`` (no scoring)."""
        if self.shape != value.shape:
            raise IndexError(
                "The buffer cannot be assigned a value; the sizes do not match."
            )

        self.__buffer.raw[:] = value

        return self

    def __getitem__(self, item: int) -> "PipelineOperator":
        """Return a per-row assign target for row ``item``."""
        return AssignPipeline(item, self.__buffer)

    def __repr__(self) -> str:
        """Return a compact representation of the operator."""
        return "Assign<VectorizableOperator>()"
