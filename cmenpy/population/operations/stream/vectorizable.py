import abc

from cmenpy.population.operations.stream.base import StreamOperator
from cmenpy.hints import ArrayType


class PipelineOperator(abc.ABC):
    @abc.abstractmethod
    def __lshift__(self, value: ArrayType) -> "PipelineOperator": ...


class VectorizableOperator(StreamOperator):
    @abc.abstractmethod
    def __lshift__(self, value: ArrayType) -> "VectorizableOperator": ...

    @abc.abstractmethod
    def __getitem__(self, item: int) -> "PipelineOperator": ...
