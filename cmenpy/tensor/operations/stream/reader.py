import abc

from cmenpy.tensor.operations.stream.base import StreamOperator
from cmenpy.hints import NDArrayType


class ReaderOperator(StreamOperator):
    @abc.abstractmethod
    def __getitem__(self, item: int | slice) -> NDArrayType: ...

    @abc.abstractmethod
    def __invert__(self) -> NDArrayType: ...

    @abc.abstractmethod
    def __rshift__(self, other: NDArrayType) -> NDArrayType: ...
