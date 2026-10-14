import numpy.typing as npt

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

    Supports ``pop.F[i]``, ``np.asarray(pop.F)`` for the full column and
    ``~pop.F`` for the active rows only.

    Parameters
    ----------
    buffer : KernelBuffer
        The backing buffer.
    """

    def __init__(self, buffer: KernelBuffer):
        self.__buffer = buffer

    def __getitem__(self, item: int | slice) -> NDArrayType:
        return self.__buffer.raw[:, 0].reshape(-1)[item]

    def __array__(
        self, dtype: "npt.DTypeLike | None" = None, copy: bool | None = None
    ) -> NDArrayType:
        return self.__buffer.raw[:, 0].reshape(-1)

    def __repr__(self) -> str:
        _F = ", ".join(map(str, self.__buffer.raw[:, 0].reshape(-1)))

        return f"F [{_F}]"

    __str__ = __repr__

    def __invert__(self) -> NDArrayType:
        return self.__buffer.raw[:, 0].reshape(-1)[self.__buffer.mask.founds]


class BufferMatrixSolutions(BufferMatrixAbstract, BufferMatrixAccess, BufferMatrixCompute):
    """Read/write view over the solution columns (columns ``1:``).

    Supports ``pop.X[i]``, ``np.asarray(pop.X)``, ``~pop.X`` for the active
    rows, and the write-back ``pop.X << values`` (which scores the block).

    Parameters
    ----------
    buffer : KernelBuffer
        The backing buffer.
    """

    def __init__(self, buffer: KernelBuffer):
        self.__buffer = buffer

    def __getitem__(self, item: int | slice) -> NDArrayType:
        return self.__buffer.raw[:, 1:][item]

    def __array__(
        self, dtype: "npt.DTypeLike | None" = None, copy: bool | None = None
    ) -> NDArrayType:
        return self.__buffer.raw[:, 1:]

    def __repr__(self) -> str:
        _X = ", ".join(map(str, self.__buffer.raw[:, 1:]))

        return f"X [{_X}]"

    __str__ = __repr__

    def __invert__(self) -> NDArrayType:
        return self.__buffer.raw[:, 1:][self.__buffer.mask.founds]

    def __lshift__(self, other: NDArrayType) -> "BufferMatrixSolutions":
        self.__buffer.apply(other)
        return self


class BufferMatrixOperations:
    """Exposes the ``F``/``X`` matrix views, ``idx`` ranking and ``size``.

    Parameters
    ----------
    buffer : KernelBuffer
        The backing buffer.

    Attributes
    ----------
    F : BufferMatrixFitness
        Fitness-column view.
    X : BufferMatrixSolutions
        Solution-block view.
    idx : BufferIndex
        Best/worst/sort lookups.
    size : int
        Number of active individuals.
    """

    def __init__(self, buffer: KernelBuffer):
        self.__buffer = buffer

    @property
    def F(self) -> BufferMatrixFitness:
        return BufferMatrixFitness(self.__buffer)

    @property
    def X(self) -> BufferMatrixSolutions:
        return BufferMatrixSolutions(self.__buffer)

    @property
    def idx(self) -> BufferIndex:
        return self.__buffer.idx

    @property
    def size(self) -> int:
        return self.__buffer.mask.active
