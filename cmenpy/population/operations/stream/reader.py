import abc

from cmenpy.hints import NDArrayType
from cmenpy.population.operations.stream.base import StreamOperator


class ReaderOperator(StreamOperator):
    """Read-only operator: indexing, ``~`` (active rows) and ``>>``."""
    @abc.abstractmethod
    def __getitem__(self, item: int | slice) -> NDArrayType: ...

    @abc.abstractmethod
    def __invert__(self) -> NDArrayType: ...

    @abc.abstractmethod
    def __rshift__(self, other: NDArrayType) -> NDArrayType: ...
