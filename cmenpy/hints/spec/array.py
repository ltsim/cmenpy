"""Array cast-spec marker (`_ArraySpec`) and the `Array` singleton."""

import typing

import numba as nb
import numpy as np

from cmenpy.hints.primitives import CasterType


class _ArraySpec:
    """Marker describing how an array parameter is cast and typed.

    ``ndim`` can be an integer (the rank) or a string of comma-separated
    dimension labels (e.g. ``"m,n"``) used by generalized ufuncs.
    """

    __slots__ = ("dtype", "layout", "ndim")

    def __init__(
        self,
        dtype: typing.Any = None,
        ndim: int | str | None = None,
        layout: str = "C",
    ) -> None:
        self.dtype = dtype
        self.ndim = ndim
        self.layout = layout

    def __getitem__(self, params: typing.Any) -> "_ArraySpec":
        items = params if isinstance(params, tuple) else (params,)
        dtype = items[0]
        ndim = items[1] if len(items) > 1 else None
        layout = items[2] if len(items) > 2 else "C"

        return _ArraySpec(dtype, ndim, layout)

    @property
    def dims(self) -> tuple[str, ...] | None:
        """Dimension labels when ``ndim`` is a string, else ``None``."""
        if isinstance(self.ndim, str):
            return tuple(part.strip() for part in self.ndim.split(",") if part.strip())

        return None

    @property
    def rank(self) -> int | None:
        """Number of dimensions, whether ``ndim`` is a rank or labels."""
        if isinstance(self.ndim, str):
            dims = self.dims

            return len(dims) if dims else None

        return self.ndim

    @property
    def typed(self) -> bool:
        return self.dtype is not None and self.rank is not None

    def numba_type(self) -> typing.Any:
        return nb.types.Array(  # type: ignore[no-untyped-call]
            nb.from_dtype(np.dtype(self.dtype)), self.rank, self.layout
        )

    def caster(self) -> CasterType:
        if self.dtype is None:
            return np.asarray

        _dtype = np.dtype(self.dtype)

        if self.layout == "F":
            return lambda value: np.asfortranarray(value, dtype=_dtype)

        if self.layout == "A":
            return lambda value: np.asarray(value, dtype=_dtype)

        return lambda value: np.ascontiguousarray(value, dtype=_dtype)

    def __repr__(self) -> str:
        return f"cm.Array[{self.dtype!r}, {self.ndim!r}, {self.layout!r}]"


#: Marker: receive this parameter as an ndarray (optionally typed).
Array = _ArraySpec()
