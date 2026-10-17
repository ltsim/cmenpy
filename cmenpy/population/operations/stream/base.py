"""Abstract base for stream operators bound to a fixed buffer shape."""

import abc


class StreamOperator(abc.ABC):
    """Base for operators bound to a fixed buffer shape.

    Args:
        shape: The ``(rows, columns)`` shape of the backing buffer.

    Attributes:
        shape: The buffer shape.
        rows: Number of rows.
        columns: Number of columns.
    """

    def __init__(self, shape: tuple[int, ...]):
        """Store the buffer ``shape`` the operator is bound to."""
        self.__shape = shape

    @property
    def shape(self) -> tuple[int, ...]:
        """The buffer shape."""
        return self.__shape

    @property
    def rows(self) -> int:
        """Number of rows."""
        return self.__shape[0]

    @property
    def columns(self) -> int:
        """Number of columns."""
        return self.__shape[1]

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Return a compact representation of the operator."""
