import abc

from cmenpy.hints import NDArrayType
from cmenpy.population.operations.stream.base import StreamOperator


class WriterOperator(StreamOperator):
    """Write-only operator: item assignment (``op[i] = value``)."""
    @abc.abstractmethod
    def __setitem__(self, item: int | slice, value: NDArrayType) -> None: ...
