"""Cast-spec markers describing how numba kernel parameters are typed and cast."""

from cmenpy.hints.spec.array import Array, _ArraySpec
from cmenpy.hints.spec.scalar import Scalar, _ScalarSpec
from cmenpy.hints.spec.signature import _Signature

__all__ = [
    "Array",
    "Scalar",
    "_ArraySpec",
    "_ScalarSpec",
    "_Signature",
]
