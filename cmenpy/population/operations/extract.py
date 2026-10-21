"""Solution-column reader operator (``pop.extract``)."""

from cmenpy.hints import NDArrayType
from cmenpy.kernel import KernelBuffer
from cmenpy.population.operations.stream import ReaderOperator


class ExtractOperation(ReaderOperator):
    """Solution-column reader: ``pop.extract[i]``, ``~pop.extract``.

    Args:
        buffer: The backing buffer.
    """

    def __init__(self, buffer: KernelBuffer):
        """Initialize the reader over ``buffer``."""
        super().__init__(buffer.shape)
        self.__buffer = buffer

    def __getitem__(self, item: int | slice) -> NDArrayType:
        """Return the solution row(s) (columns ``1:``) at ``item``."""
        return self.__buffer.raw[item, 1:]

    def __invert__(self) -> NDArrayType:
        """Return the solution rows of the active individuals only."""
        return self.__buffer.raw[self.__buffer.mask.founds, 1:]

    def __rshift__(self, other: NDArrayType) -> NDArrayType:
        """Pass ``other`` through (identity, for pipeline symmetry)."""
        return other

    def __repr__(self) -> str:
        """Return a compact representation of the operator."""
        return "Extract<ReaderOperator>()"
