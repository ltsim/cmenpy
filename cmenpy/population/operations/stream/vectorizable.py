import abc

from cmenpy.hints import NDArrayType
from cmenpy.population.operations.stream.base import StreamOperator


class PipelineOperator(abc.ABC):
    """Per-row target of a ``<<`` write (e.g. ``pop.compute[i] << row``)."""
    @abc.abstractmethod
    def __lshift__(self, value: NDArrayType) -> None: ...


class VectorizableOperator(StreamOperator):
    """Whole-block operator supporting ``<<`` and per-row indexing."""
    @abc.abstractmethod
    def __lshift__(self, value: NDArrayType) -> "VectorizableOperator | None": ...

    @abc.abstractmethod
    def __getitem__(self, item: int) -> "PipelineOperator": ...
