import abc

from cmenpy.operations.stream.base import StreamOperator
from cmenpy.types import NDArrayType


class AllOperator(StreamOperator):
    @abc.abstractmethod
    def __getitem__(self, item) -> None: ...

    @abc.abstractmethod
    def __setitem__(self, item: int | slice, value: NDArrayType) -> None: ...
