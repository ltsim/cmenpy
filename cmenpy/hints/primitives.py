"""Primitive typing aliases: type vars, callables and generic pairs."""

import typing

import numpy as np

#: Type variable bound to a NumPy scalar, used for scalar bounds/values.
ScalarType = typing.TypeVar("ScalarType", bound=np.number[typing.Any])

#: Callable casting a value (array or scalar) to a spec's dtype/layout.
CasterType = typing.Callable[[typing.Any], typing.Any]

#: First element of a ``(new, old)`` fitness pair compared by the comparators.
A = typing.TypeVar("A")
#: Second element of a ``(new, old)`` fitness pair compared by the comparators.
B = typing.TypeVar("B")

#: Generic ``(new, old)`` pair of fitness values.
AB = tuple[A, B]
