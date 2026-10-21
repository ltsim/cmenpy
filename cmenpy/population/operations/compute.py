"""Write-and-score operator (``pop.compute``)."""

from cmenpy.hints import NDArrayType
from cmenpy.kernel import KernelBuffer, RowSelection
from cmenpy.population.operations.stream.vectorizable import (
    PipelineOperator,
    VectorizableOperator,
)


class ComputePipeline(PipelineOperator):
    """Per-selection compute target: ``pop.compute[idx] << rows``.

    The selection may be one row (``pop.compute[3]`` or
    ``pop.compute[pop.best]``), several rows (a slice, an index array, a
    boolean mask) or the active roster (``pop.compute[~pop]``). Each row
    written through it spends exactly one evaluation.

    Args:
        idx: The rows to write and evaluate.
        buffer: The backing buffer.
    """

    def __init__(self, idx: RowSelection, buffer: KernelBuffer) -> None:
        """Bind the pipeline to the rows selected by ``idx``."""
        self.__idx = idx
        self.__buffer = buffer

    def __lshift__(self, value: NDArrayType) -> None:
        """Write ``value`` into the selected rows and score them."""
        self.__buffer.apply(value, self.__idx)


class ComputeOperator(VectorizableOperator):
    """Write-and-score operator: ``pop.compute << rows`` / ``pop.compute[idx] << rows``.

    Writing through ``compute`` inserts the solutions *and* evaluates them
    against the objective. The evaluation counter is charged one function
    evaluation per row scored: the whole active block costs its active rows, a
    mask costs the rows it selects, a single row costs one.

    Args:
        buffer: The backing buffer.
    """

    def __init__(self, buffer: KernelBuffer) -> None:
        """Initialize the operator over ``buffer``."""
        super().__init__(buffer.shape)
        self.__buffer = buffer

    def __lshift__(self, value: NDArrayType) -> None:
        """Insert and score ``value`` into the active rows against the objective."""
        self.__buffer.apply(value)

    def __getitem__(self, item: RowSelection) -> "PipelineOperator":
        """Return a compute target for the rows selected by ``item``."""
        return ComputePipeline(item, self.__buffer)

    def __repr__(self) -> str:
        """Return a compact representation of the operator."""
        return "Compute<VectorizableOperator>()"
