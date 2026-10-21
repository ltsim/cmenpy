"""Matrix views over the buffer: fitness column, solution block, operations."""

from cmenpy.hints import NDArrayType
from cmenpy.kernel import KernelBuffer
from cmenpy.kernel.index import BufferIndex
from cmenpy.population.attributes.matrix import (
    BufferMatrixAbstract,
    BufferMatrixAccess,
    BufferMatrixCompute,
)


class BufferMatrixFitness(BufferMatrixAbstract, BufferMatrixAccess):
    """Read view over the buffer's fitness column (column 0).

    Supports ``pop.F[i]``, ``pop.F.to_numpy()`` (or ``cm.cast(pop.F)``) for the
    full column and ``~pop.F`` for the active rows only.

    Args:
        buffer: The backing buffer.
    """

    def __init__(self, buffer: KernelBuffer):
        """Initialize the view over ``buffer``."""
        self.__buffer = buffer

    def __getitem__(self, item: int | slice) -> NDArrayType:
        """Return the fitness value(s) at row ``item``."""
        return self.__buffer.raw[:, 0].reshape(-1)[item]

    def to_numpy(self) -> NDArrayType:
        """Return the full fitness column as an ndarray."""
        return self.__buffer.raw[:, 0].reshape(-1)

    def __repr__(self) -> str:
        """Return a compact ``F [f0, f1, ...]`` representation."""
        _F = ", ".join(map(str, self.__buffer.raw[:, 0].reshape(-1)))

        return f"F [{_F}]"

    __str__ = __repr__

    def __invert__(self) -> NDArrayType:
        """Return the fitness values of the active rows only."""
        return self.__buffer.raw[:, 0].reshape(-1)[self.__buffer.mask.founds]


class BufferMatrixSolutions(
    BufferMatrixAbstract, BufferMatrixAccess, BufferMatrixCompute
):
    """Read/write view over the solution columns (columns ``1:``).

    Supports ``pop.X[i]``, ``pop.X.to_numpy()`` (or ``cm.cast(pop.X)``),
    ``~pop.X`` for the active rows, and the write-back ``pop.X << values``
    (which scores the block).

    Args:
        buffer: The backing buffer.
    """

    def __init__(self, buffer: KernelBuffer):
        """Initialize the view over ``buffer``."""
        self.__buffer = buffer

    def __getitem__(self, item: int | slice) -> NDArrayType:
        """Return the solution row(s) at ``item``."""
        return self.__buffer.raw[:, 1:][item]

    def to_numpy(self) -> NDArrayType:
        """Return the full solution block as an ndarray."""
        return self.__buffer.raw[:, 1:]

    def __repr__(self) -> str:
        """Return a compact ``X [row0, row1, ...]`` representation."""
        _X = ", ".join(map(str, self.__buffer.raw[:, 1:]))

        return f"X [{_X}]"

    __str__ = __repr__

    def __invert__(self) -> NDArrayType:
        """Return the solution rows of the active individuals only."""
        return self.__buffer.raw[:, 1:][self.__buffer.mask.founds]

    def __lshift__(self, other: NDArrayType) -> "BufferMatrixSolutions":
        """Score and insert ``other`` into the buffer (``pop.X << values``)."""
        self.__buffer.apply(other)
        return self


class BufferMatrixOperations:
    """Exposes the ``F``/``X`` matrix views, ``idx`` ranking and ``size``.

    Args:
        buffer: The backing buffer.

    Attributes:
        F: Fitness-column view.
        X: Solution-block view.
        idx: Best/worst/sort lookups.
        size: Number of active individuals.
    """

    def __init__(self, buffer: KernelBuffer):
        """Initialize the operations over ``buffer``."""
        self.__buffer = buffer

    @property
    def F(self) -> BufferMatrixFitness:
        """Fitness-column view."""
        return BufferMatrixFitness(self.__buffer)

    @property
    def X(self) -> BufferMatrixSolutions:
        """Solution-block view."""
        return BufferMatrixSolutions(self.__buffer)

    @property
    def idx(self) -> BufferIndex:
        """Best/worst/sort lookups."""
        return self.__buffer.idx

    @property
    def size(self) -> int:
        """Number of active individuals."""
        return self.__buffer.mask.active
