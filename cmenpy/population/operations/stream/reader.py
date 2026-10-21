"""Abstract read-only stream operator."""

import abc

from cmenpy.hints import NDArrayType
from cmenpy.population.operations.stream.base import StreamOperator


class ReaderOperator(StreamOperator):
    """Read-only operator: indexing, ``~`` (active rows) and ``>>``."""

    @abc.abstractmethod
    def __getitem__(self, item: int | slice) -> NDArrayType:
        """Return the value(s) at ``item``."""

    @abc.abstractmethod
    def __invert__(self) -> NDArrayType:
        """Return the active rows of the operator's view."""

    @abc.abstractmethod
    def __rshift__(self, other: NDArrayType) -> NDArrayType:
        """Pass ``other`` through (identity, for pipeline symmetry)."""
