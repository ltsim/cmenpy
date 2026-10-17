"""Eager-signature marker (`_Signature`) for ``@cm.vectorize`` kernels."""

import typing

import numpy as np

if typing.TYPE_CHECKING:
    from cmenpy.hints.spec.array import _ArraySpec
    from cmenpy.hints.spec.scalar import _ScalarSpec


def _dtype_name(dtype: typing.Any) -> str:
    return str(np.dtype(dtype).name)


class _Signature:
    """Full eager signature for ``@cm.vectorize`` (result + argument specs)."""

    __slots__ = ("args", "result")

    def __init__(
        self,
        result: "_ArraySpec | _ScalarSpec",
        args: "tuple[_ArraySpec | _ScalarSpec, ...]",
    ) -> None:
        self.result = result
        self.args = args

    def numba_signature(self) -> str:
        result = _dtype_name(self.result.dtype)
        args = ", ".join(_dtype_name(arg.dtype) for arg in self.args)

        return f"{result}({args})"

    def __repr__(self) -> str:
        return f"cm.Scalar[{self.result.dtype!r}, {list(self.args)!r}]"
