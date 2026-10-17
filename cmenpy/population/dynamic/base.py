"""Roster management over the buffer: insert, remove and free space."""

import numpy as np

from cmenpy.hints import NDArrayType
from cmenpy.kernel import KernelBuffer


class DynamicBase:
    """Roster management over the buffer: insert, remove, free space.

    Args:
        buffer: The buffer whose active rows are managed.

    Attributes:
        free_space: ``True`` while at least one row is inactive (insertable).
    """

    def __init__(self, buffer: KernelBuffer):
        """Initialize the roster over ``buffer``."""
        self.__buffer = buffer

    @property
    def free_space(self) -> bool:
        """``True`` while at least one row is inactive (insertable)."""
        return sum(self.__buffer.mask) < self.__buffer.size.max

    def remove(self, idx: int) -> None:
        """Deactivate row ``idx`` and clear it to ``NaN``."""
        self.__buffer.mask[idx] = False
        self.__buffer.raw[idx, :] = np.nan

    def insert(self, x: NDArrayType) -> int:
        """Insert a solution into the first free row and return its index.

        Args:
            x: The solution to score and insert.

        Returns:
            The index of the row the solution was inserted into.

        Raises:
            ValueError: If the buffer has no free rows.
        """
        founds = [i for i, x in enumerate(self.__buffer.mask.founds) if not x]

        if not len(founds) > 0:
            raise ValueError("Max agents in memory.")

        idx = founds[0]

        self.__buffer.mask[idx] = True
        self.__buffer.apply(x, idx)

        return idx
