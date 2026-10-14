from cmenpy.hints import NDArrayType
from cmenpy.kernel import KernelBuffer
from cmenpy.population.operations.stream.vectorizable import (
    PipelineOperator,
    VectorizableOperator,
)


class ComputePipeline(PipelineOperator):
    """Per-row compute target: ``pop.compute[i] << row`` (writes and scores).

    Parameters
    ----------
    idx : int
        The row to write and evaluate.
    buffer : KernelBuffer
        The backing buffer.
    """
    def __init__(self, idx: int, buffer: KernelBuffer) -> None:
        self.__idx = idx
        self.__buffer = buffer

    def __lshift__(self, value: NDArrayType) -> None:
        self.__buffer.apply(value, self.__idx)


class ComputeOperator(VectorizableOperator):
    """Write-and-score operator: ``pop.compute << rows`` / ``pop.compute[i] << row``.

    Writing through ``compute`` inserts the solutions *and* evaluates them
    against the objective.

    Parameters
    ----------
    buffer : KernelBuffer
        The backing buffer.
    """

    def __init__(self, buffer: KernelBuffer) -> None:
        super().__init__(buffer.shape)
        self.__buffer = buffer

    def __lshift__(self, value: NDArrayType) -> None:
        self.__buffer.apply(value)

    def __getitem__(self, item: int) -> "PipelineOperator":
        return ComputePipeline(item, self.__buffer)

    def __repr__(self) -> str:
        return "Compute<VectorizableOperator>()"
