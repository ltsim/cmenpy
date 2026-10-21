"""Argument casters and the autocast wrapper applied to compiled kernels."""

import functools
import inspect
import typing

import numpy as np

from cmenpy.casting import unwrap_array
from cmenpy.hints.primitives import CasterType


def _as_array(value: typing.Any) -> typing.Any:
    """Convert cmenpy views and array-protocol objects to plain ndarrays.

    Buffer views (``pop.X``, ``pop.F``, ...) unwrap through their explicit
    ``to_numpy`` method, so kernels receive the underlying ndarray without an
    implicit ``__array__`` cast; other array-protocol objects still convert
    through ``np.asarray``. NumPy scalars stay scalars and lists, tuples or
    arbitrary objects are not touched.
    """
    value = unwrap_array(value)

    if isinstance(value, (np.ndarray, np.generic)) or not hasattr(value, "__array__"):
        return value

    return np.asarray(value)


def _autocast(
    compiled: typing.Any,
    casters: typing.Sequence[CasterType | None],
    func: typing.Callable[..., typing.Any],
) -> typing.Any:
    """Wrap ``compiled`` so its arguments are cast before being passed.

    A parameter with a declared caster is cast by it (dtype/layout preserved);
    a parameter *without* one still accepts cmenpy views and array-protocol
    objects through `_as_array`, so buffer views reach the kernel as ndarrays.
    Declared casters also receive the unwrapped value, which keeps the
    deprecated ``__array__`` hook out of the kernel path entirely. NumPy scalar
    and tuple results are preserved; array-protocol results become ndarrays.
    """
    names = list(inspect.signature(func).parameters)
    mapping = {
        name: caster for name, caster in zip(names, casters) if caster is not None
    }

    @functools.wraps(func)
    def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        unwrapped = tuple(_as_array(value) for value in args)
        casted = tuple(
            (
                mapping[names[i]](value)
                if i < len(names) and names[i] in mapping
                else value
            )
            for i, value in enumerate(unwrapped)
        )
        unwrapped_kwargs = {name: _as_array(value) for name, value in kwargs.items()}
        casted_kwargs = {
            name: mapping[name](value) if name in mapping else value
            for name, value in unwrapped_kwargs.items()
        }

        return _as_array(compiled(*casted, **casted_kwargs))

    wrapper.__numba__ = compiled  # type: ignore[attr-defined]

    return wrapper
