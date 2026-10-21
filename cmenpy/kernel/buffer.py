"""Flat ``(pop_size, ndim + 1)`` store backing a population."""

import numpy as np

from cmenpy.bounds import Bounds
from cmenpy.casting import unwrap_array
from cmenpy.hints import NDArrayType
from cmenpy.hints.option import SenseType
from cmenpy.kernel.index import BufferIndex
from cmenpy.kernel.low import init_buffer
from cmenpy.kernel.mask import KernelMask
from cmenpy.kernel.size import KernelSize
from cmenpy.target import TargetFunction


class KernelBuffer:
    """Flat ``(pop_size, ndim + 1)`` store backing a population.

    Column 0 holds fitness; columns ``1:`` hold the solution vector. New
    solutions are written with `apply`, which scores them against the
    objective (whole-block when the target is vectorized, row-by-row
    otherwise).

    Args:
        size: Population size bounds.
        bounds: Search space (defines ``ndim``).
        target: The wrapped objective used to score solutions.
        sense: ``"min"`` or ``"max"``.

    Attributes:
        shape: Shape of the raw buffer.
        mask: The active-row mask.
        raw: The underlying buffer array.
        idx: Best/worst/sort lookups over the fitness column.
        size: The population size bounds.
        snapshot: A fresh empty buffer with the same configuration.
    """

    def __init__(
        self,
        size: KernelSize,
        bounds: Bounds,
        target: TargetFunction,
        sense: SenseType,
    ):
        """Initialize the buffer, mask and configuration. See class docstring."""
        self.__size = size
        self.__bounds = bounds
        self.__target = target
        self.__sense = sense

        self.__buffer: NDArrayType = init_buffer(self.__size.max, bounds.ndim)
        self.__mask = KernelMask(self.__size)

    def apply(
        self,
        x: NDArrayType,
        idx: int | slice | list[int] | None = None,
    ) -> None:
        """Vectorizes the target function to evaluate and update the solutions matrix.

        The internal buffer stores "fitness" in column 0 and "solutions" or "x"
        from column 1 onwards. Supports updating a single row, a slice, a
        list/array of indices, or the entire active matrix if `idx` is None.

        Targets wrapped with `cmenpy.target.vectorized_target` score the whole
        block with a single call; the others fall back to a row-by-row
        evaluation.

        Args:
            x: An array containing the new solution data to be inserted into
                the buffer.
            idx: The row index, slice, or index mask to update. If None,
                defaults to the internal mask.
        """
        rows = self.__mask.founds if idx is None else idx
        x = unwrap_array(x)
        self.__buffer[rows, 1:] = x

        if isinstance(idx, int):
            self.__buffer[rows, 0] = self.__target.evaluate(x)
        elif self.__target.vectorized:
            self.__buffer[rows, 0] = self.__target.evaluate_matrix(
                self.__buffer[rows, 1:]
            )
        else:
            self.__buffer[rows, 0] = np.apply_along_axis(
                self.__target, 1, self.__buffer[rows, 1:]
            )

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the raw buffer."""
        return self.__buffer.shape

    @property
    def mask(self) -> KernelMask:
        """The active-row mask."""
        return self.__mask

    @property
    def raw(self) -> NDArrayType:
        """The underlying buffer array."""
        return self.__buffer

    @property
    def idx(self) -> BufferIndex:
        """Best/worst/sort lookups over the fitness column."""
        return BufferIndex(self.__buffer, self.__mask, self.__sense)

    @property
    def size(self) -> KernelSize:
        """The population size bounds."""
        return self.__size

    @property
    def snapshot(self) -> "KernelBuffer":
        """A fresh empty buffer with the same size, bounds, target and sense."""
        return KernelBuffer(self.__size, self.__bounds, self.__target, self.__sense)
