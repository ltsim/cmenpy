"""The ``_ContractSpec`` eager signature (core types + layout) for `cm.reduce`."""

import itertools
import typing

import numpy as np

from cmenpy.compile.backend import numba_module
from cmenpy.hints.spec import _ArraySpec, _ScalarSpec


def _contract_element(value: typing.Any) -> "_ArraySpec | _ScalarSpec":
    """Normalize one ``Contract`` entry into an array or scalar spec."""
    if value is Ellipsis:
        return _ScalarSpec(float)

    if isinstance(value, _ArraySpec):
        if value.dtype is None:
            raise TypeError("Contract array specs need a dtype, e.g. Array[float].")

        return value

    if isinstance(value, _ScalarSpec):
        if value.dtype is None:
            raise TypeError("Contract scalar specs need a dtype, e.g. Scalar[float].")

        return value

    try:
        np.dtype(value)
    except TypeError:
        raise TypeError(f"Unsupported Contract spec: {value!r}.") from None

    return _ArraySpec(value, "n")


def _contract_atom(dims: tuple[str, ...]) -> str:
    return "()" if not dims else "(" + ",".join(dims) + ")"


def _contract_numba_type(
    spec: "_ArraySpec | _ScalarSpec", dims: tuple[str, ...]
) -> typing.Any:
    nb = numba_module()

    if isinstance(spec, _ScalarSpec):
        return nb.from_dtype(np.dtype(spec.dtype))

    return nb.types.Array(nb.from_dtype(np.dtype(spec.dtype)), len(dims), spec.layout)


class _ContractSpec:
    """Full eager signature for ``@cm.reduce`` (core types + layout).

    The layout and casters are computed eagerly (NumPy only); the numba core
    types are built lazily, on first compilation, so declaring a ``Contract``
    does not require numba to be installed.
    """

    __slots__ = ("_numba_types", "_resolved", "casters", "layout")

    def __init__(self, inputs: tuple[typing.Any, ...], output: typing.Any) -> None:
        elements = [_contract_element(item) for item in (*inputs, output)]

        if isinstance(elements[-1], _ScalarSpec):
            raise TypeError(
                "cm.Contract does not support a scalar output; use an Array[...]."
            )

        named = {
            dim
            for spec in elements
            if isinstance(spec, _ArraySpec)
            for dim in spec.dims or ()
        }
        counter = itertools.count()
        resolved: list[tuple[_ArraySpec | _ScalarSpec, tuple[str, ...]]] = []

        for spec in elements:
            if isinstance(spec, _ScalarSpec):
                resolved.append((spec, ()))
                continue

            if isinstance(spec.ndim, str):
                resolved.append((spec, spec.dims or ()))
                continue

            rank = 1 if spec.ndim is None else spec.ndim
            dims: list[str] = []

            for _ in range(rank):
                while True:
                    label = f"k{next(counter)}"

                    if label not in named:
                        break

                dims.append(label)
                named.add(label)

            resolved.append((spec, tuple(dims)))

        input_dims = {dim for _, dims in resolved[:-1] for dim in dims}
        missing = set(resolved[-1][1]) - input_dims

        if missing:
            raise TypeError(
                f"Contract output dimensions {sorted(missing)} do not appear "
                "in any input."
            )

        self._resolved = resolved
        atoms = [_contract_atom(dims) for _, dims in resolved]
        self.layout = ",".join(atoms[:-1]) + "->" + atoms[-1]
        self.casters = [spec.caster() for spec, _ in resolved[:-1]]
        self._numba_types: tuple[typing.Any, ...] | None = None

    @property
    def numba_types(self) -> tuple[typing.Any, ...]:
        """The numba core types, built on first use (needs numba)."""
        if self._numba_types is None:
            self._numba_types = tuple(
                _contract_numba_type(spec, dims) for spec, dims in self._resolved
            )

        return self._numba_types

    def __repr__(self) -> str:
        return f"cm.Contract[{list(self.numba_types[:-1])!r}, {self.numba_types[-1]!r}]"
