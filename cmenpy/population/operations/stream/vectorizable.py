import abc

from cmenpy.hints import NDArrayType
from cmenpy.population.operations.stream.base import StreamOperator


class PipelineOperator(abc.ABC):
    @abc.abstractmethod
    def __lshift__(self, value: NDArrayType) -> None: ...


class VectorizableOperator(StreamOperator):
    @abc.abstractmethod
    def __lshift__(self, value: NDArrayType) -> "VectorizableOperator | None": ...

    @abc.abstractmethod
    def __getitem__(self, item: int) -> "PipelineOperator": ...
