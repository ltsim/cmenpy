import abc

from cmenpy.tensor.operations.stream.base import StreamOperator
from cmenpy.hints import NDArrayType


class AllOperator(StreamOperator):
    @abc.abstractmethod
    def __getitem__(self, item) -> None: ...

    @abc.abstractmethod
    def __setitem__(self, item: int | slice, value: NDArrayType) -> None: ...
