import abc

from cmenpy.population.operations.stream.base import StreamOperator
from cmenpy.hints import ArrayType


class WriterOperator(StreamOperator):
    @abc.abstractmethod
    def __setitem__(self, item: int | slice, value: ArrayType) -> None: ...
