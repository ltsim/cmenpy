"""Argument casters and the autocast wrapper applied to compiled kernels."""

import functools
import inspect
import typing

import numpy as np

from cmenpy.hints.primitives import CasterType


def _elementwise_caster(dtype: typing.Any) -> CasterType:
    """Cast arrays to ``dtype`` and scalars to a ``dtype`` scalar."""
    if dtype is None:
        return np.asarray

    _dtype = np.dtype(dtype)

    def cast(value: typing.Any) -> typing.Any:
        if np.ndim(value) == 0:
            return _dtype.type(value)

        return np.asarray(value, dtype=_dtype)

    return cast


def _as_array(value: typing.Any) -> typing.Any:
    """Convert array-protocol objects to plain ndarrays, passing anything else through.

    Buffer views (``pop.X``, ``pop.F``, ...) expose ``__array__``; kernels
    receive the underlying ndarray instead of the view. NumPy scalars stay
    scalars and lists, tuples or arbitrary objects are not touched.
    """
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
    a parameter *without* one still accepts array-protocol objects through
    `_as_array`, so buffer views reach the kernel as ndarrays. NumPy scalar
    and tuple results are preserved; array-protocol results become ndarrays.
    """
    names = list(inspect.signature(func).parameters)
    mapping = {
        name: caster for name, caster in zip(names, casters) if caster is not None
    }

    @functools.wraps(func)
    def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        casted = tuple(
            (
                mapping[names[i]](value)
                if i < len(names) and names[i] in mapping
                else _as_array(value)
            )
            for i, value in enumerate(args)
        )
        casted_kwargs = {
            name: mapping[name](value) if name in mapping else _as_array(value)
            for name, value in kwargs.items()
        }

        return _as_array(compiled(*casted, **casted_kwargs))

    wrapper.__numba__ = compiled  # type: ignore[attr-defined]

    return wrapper
