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


def _autocast(
    compiled: typing.Any,
    casters: typing.Sequence[CasterType | None],
    func: typing.Callable[..., typing.Any],
) -> typing.Any:
    """Wrap ``compiled`` so its arguments are cast before being passed."""
    if not any(casters):
        return compiled

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
                else value
            )
            for i, value in enumerate(args)
        )
        casted_kwargs = {
            name: mapping[name](value) if name in mapping else value
            for name, value in kwargs.items()
        }

        return compiled(*casted, **casted_kwargs)

    wrapper.__numba__ = compiled  # type: ignore[attr-defined]

    return wrapper
