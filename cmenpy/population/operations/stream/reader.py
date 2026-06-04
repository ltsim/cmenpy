import abc

from cmenpy.population.operations.stream.base import StreamOperator
from cmenpy.types import NDArrayType


class ReaderOperator(StreamOperator):
    @abc.abstractmethod
    def __getitem__(self, item: int | slice) -> NDArrayType: ...
