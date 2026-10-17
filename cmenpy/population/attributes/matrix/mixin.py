"""Access and write-back mixins for the buffer matrix views."""

import abc
import typing

from cmenpy.hints import NDArrayType


class BufferMatrixAccess(abc.ABC):
    """Read access via ``~`` (the active rows of a matrix view)."""

    @abc.abstractmethod
    def __invert__(self) -> NDArrayType:
        """Return the active rows of the view."""


class BufferMatrixCompute(abc.ABC):
    """Write-back via ``<<`` (assign and score a block of solutions)."""

    @abc.abstractmethod
    def __lshift__(self, other: NDArrayType) -> typing.Self:
        """Assign and score ``other``, returning the view for chaining."""
