import abc

from cmenpy.tensor.operations.stream.base import StreamOperator
from cmenpy.hints import NDArrayType


class WriterOperator(StreamOperator):
    @abc.abstractmethod
    def __setitem__(self, item: int | slice, value: NDArrayType) -> None: ...
