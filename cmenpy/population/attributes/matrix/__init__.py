"""Abstract matrix-view interface shared by the ``F``/``X`` buffer views."""

import abc
import warnings

import numpy as np
import numpy.typing as npt

from cmenpy.hints import NDArrayType
from cmenpy.population.attributes.matrix.mixin import (
    BufferMatrixAccess,
    BufferMatrixCompute,
)


class BufferMatrixAbstract(abc.ABC):
    """Abstract read interface shared by the ``F``/``X`` matrix views.

    Subclasses must implement indexing, the explicit ``to_numpy`` conversion,
    and ``__repr__``/``__str__`` display. Implicit coercion through NumPy's
    ``__array__`` hook is deprecated; use ``cm.cast(...)`` or ``to_numpy()``.
    """

    @abc.abstractmethod
    def to_numpy(self) -> NDArrayType:
        """Return the view as an ndarray."""

    @abc.abstractmethod
    def __getitem__(self, item: int) -> NDArrayType:
        """Return the value(s) at ``item``."""

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Return a debug representation of the view."""

    @abc.abstractmethod
    def __str__(self) -> str:
        """Return a human-readable rendering of the view."""

    def __array__(
        self,
        dtype: "npt.DTypeLike | None" = None,
        copy: "bool | None" = None,
    ) -> NDArrayType:
        """Return the view as an ndarray through the deprecated NumPy hook."""
        warnings.warn(
            f"implicit casting of {type(self).__name__} via ``np.asarray`` is "
            "deprecated; use ``cm.cast(...)`` or ``.to_numpy()`` instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        array = self.to_numpy()

        if dtype is None and copy is None:
            return array

        return np.array(array, dtype=dtype, copy=copy)


__all__ = ["BufferMatrixAbstract", "BufferMatrixAccess", "BufferMatrixCompute"]
