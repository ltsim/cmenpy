"""Write-and-score operator (``pop.compute``)."""

from cmenpy.hints import NDArrayType
from cmenpy.kernel import KernelBuffer
from cmenpy.population.operations.stream.vectorizable import (
    PipelineOperator,
    VectorizableOperator,
)


class ComputePipeline(PipelineOperator):
    """Per-row compute target: ``pop.compute[i] << row`` (writes and scores).

    Args:
        idx: The row to write and evaluate.
        buffer: The backing buffer.
    """

    def __init__(self, idx: int, buffer: KernelBuffer) -> None:
        """Bind the pipeline to row ``idx`` of ``buffer``."""
        self.__idx = idx
        self.__buffer = buffer

    def __lshift__(self, value: NDArrayType) -> None:
        """Write ``value`` into the bound row and score it."""
        self.__buffer.apply(value, self.__idx)


class ComputeOperator(VectorizableOperator):
    """Write-and-score operator: ``pop.compute << rows`` / ``pop.compute[i] << row``.

    Writing through ``compute`` inserts the solutions *and* evaluates them
    against the objective.

    Args:
        buffer: The backing buffer.
    """

    def __init__(self, buffer: KernelBuffer) -> None:
        """Initialize the operator over ``buffer``."""
        super().__init__(buffer.shape)
        self.__buffer = buffer

    def __lshift__(self, value: NDArrayType) -> None:
        """Insert and score ``value`` against the objective."""
        self.__buffer.apply(value)

    def __getitem__(self, item: int) -> "PipelineOperator":
        """Return a per-row compute target for row ``item``."""
        return ComputePipeline(item, self.__buffer)

    def __repr__(self) -> str:
        """Return a compact representation of the operator."""
        return "Compute<VectorizableOperator>()"
