"""Whole-row reader operator (``pop.access``)."""

from cmenpy.hints import NDArrayType
from cmenpy.kernel import KernelBuffer
from cmenpy.population.operations.stream import ReaderOperator


class AccessOperator(ReaderOperator):
    """Whole-row reader: ``pop.access[i]``, ``~pop.access``.

    Args:
        buffer: The backing buffer.
    """

    def __init__(
        self,
        buffer: KernelBuffer,
    ):
        """Initialize the reader over ``buffer``."""
        super().__init__(buffer.shape)
        self.__buffer = buffer

    def __getitem__(self, item: int | slice) -> NDArrayType:
        """Return the full row(s) (fitness + solution) at ``item``."""
        return self.__buffer.raw[item, :]

    def __invert__(self) -> NDArrayType:
        """Return the full rows of the active individuals only."""
        return self.__buffer.raw[self.__buffer.mask.founds, :]

    def __rshift__(self, other: NDArrayType) -> NDArrayType:
        """Pass ``other`` through (identity, for pipeline symmetry)."""
        return other

    def __repr__(self) -> str:
        """Return a compact representation of the operator."""
        return "Access<ReaderOperator>()"
