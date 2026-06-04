import abc

from cmenpy.population.operations.stream.base import StreamOperator
from cmenpy.types import NDArrayType


class ReaderOperator(StreamOperator):
    @abc.abstractmethod
    def __getitem__(self, item: int | slice) -> NDArrayType: ...


class WriterOperator(StreamOperator):
    @abc.abstractmethod
    def __setitem__(self, item: int | slice, value: NDArrayType) -> None: ...


class VectorizableOperator(StreamOperator):
    @abc.abstractmethod
    def __imatmul__(self, other: NDArrayType) -> "VectorizableOperator": ...

    @abc.abstractmethod
    def __matmul__(self, other: NDArrayType) -> None: ...

    @abc.abstractmethod
    def __setitem__(
        self, item: int | slice | list[int] | tuple[int, ...], value: NDArrayType
    ) -> None: ...


class AllOperator(StreamOperator):
    @abc.abstractmethod
    def __getitem__(self, item) -> None: ...

    @abc.abstractmethod
    def __setitem__(self, item: int | slice, value: NDArrayType) -> None: ...
