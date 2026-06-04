import abc

from cmenpy.population.operations.stream.base import StreamOperator
from cmenpy.types import NDArrayType


class PipelineOperator(abc.ABC):
    @abc.abstractmethod
    def __lshift__(self, value: NDArrayType) -> "PipelineOperator": ...


class VectorizableOperator(StreamOperator):
    @abc.abstractmethod
    def __lshift__(self, value: NDArrayType) -> "VectorizableOperator": ...

    @abc.abstractmethod
    def __getitem__(self, item: int) -> "PipelineOperator": ...
