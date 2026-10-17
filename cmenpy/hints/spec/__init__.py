"""Cast-spec markers describing how numba kernel parameters are typed and cast."""

import typing

from cmenpy.hints.spec.array import Array, _ArraySpec
from cmenpy.hints.spec.scalar import Scalar, _ScalarSpec
from cmenpy.hints.spec.signature import _Signature

#: A cast spec marker: an _ArraySpec, _ScalarSpec or _Signature.
SpecType = typing.Any

__all__ = [
    "Array",
    "Scalar",
    "SpecType",
    "_ArraySpec",
    "_ScalarSpec",
    "_Signature",
]
