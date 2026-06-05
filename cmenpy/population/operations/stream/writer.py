import abc

from cmenpy.population.operations.stream.base import StreamOperator
from cmenpy.types import NDArrayType


class WriterOperator(StreamOperator):
    @abc.abstractmethod
    def __setitem__(self, item: int | slice, value: NDArrayType) -> None: ...
