"""Typing aliases and cast-spec markers shared across cmenpy."""

from cmenpy.hints.array import ArrayFloatType, ArrayIntegerType, NDArrayType
from cmenpy.hints.primitives import AB, A, B, CasterType, ScalarType
from cmenpy.hints.protocol import Target
from cmenpy.hints.spec import (
    Array,
    Scalar,
    _ArraySpec,
    _ScalarSpec,
    _Signature,
)

__all__ = [
    "AB",
    "A",
    "Array",
    "ArrayFloatType",
    "ArrayIntegerType",
    "B",
    "CasterType",
    "NDArrayType",
    "Scalar",
    "ScalarType",
    "Target",
    "_ArraySpec",
    "_ScalarSpec",
    "_Signature",
]
