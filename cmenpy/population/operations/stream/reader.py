import abc

from cmenpy.population.operations.stream.base import StreamOperator
from cmenpy.hints import ArrayType


class ReaderOperator(StreamOperator):
    @abc.abstractmethod
    def __getitem__(self, item: int | slice) -> ArrayType: ...

    @abc.abstractmethod
    def __invert__(self) -> ArrayType: ...

    @abc.abstractmethod
    def __rshift__(self, other: ArrayType) -> ArrayType: ...
