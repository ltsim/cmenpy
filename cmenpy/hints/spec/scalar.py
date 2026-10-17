"""Scalar cast-spec marker (`_ScalarSpec`) and the `Scalar` singleton."""

import typing

import numba as nb
import numpy as np

from cmenpy.hints.primitives import CasterType
from cmenpy.hints.spec.signature import _Signature


class _ScalarSpec:
    """Marker describing how a scalar parameter is cast and typed."""

    __slots__ = ("dtype",)

    def __init__(self, dtype: typing.Any = None) -> None:
        self.dtype = dtype

    def __getitem__(self, params: typing.Any) -> "_ScalarSpec | _Signature":
        items = params if isinstance(params, tuple) else (params,)

        if len(items) == 1:
            return _ScalarSpec(items[0])

        result = _ScalarSpec(items[0])
        args = tuple(items[1])

        return _Signature(result, args)

    @property
    def typed(self) -> bool:
        return self.dtype is not None

    def numba_type(self) -> typing.Any:
        return nb.from_dtype(np.dtype(self.dtype))

    def caster(self) -> CasterType:
        if self.dtype is None:
            return float

        _dtype = np.dtype(self.dtype)

        return lambda value: _dtype.type(value)

    def __repr__(self) -> str:
        return f"cm.Scalar[{self.dtype!r}]"


#: Marker: receive this parameter as a scalar (optionally typed).
Scalar = _ScalarSpec()
