"""Explicit casting of population views and array-likes to NumPy arrays."""

import typing

import numpy as np
import numpy.typing as npt


def unwrap_array(value: typing.Any) -> typing.Any:
    """Return the ndarray behind a cmenpy view, leaving other values unchanged.

    cmenpy buffer views (``pop.X``/``pop.F``) expose an explicit ``to_numpy``
    conversion instead of relying on NumPy's ``__array__`` hook, so compiled
    kernels and comparators can unwrap them without an implicit cast.

    Args:
        value: A cmenpy buffer view, an array, a list, a scalar, ...

    Returns:
        The underlying ``ndarray`` for cmenpy views, otherwise ``value``.
    """
    to_numpy = getattr(value, "to_numpy", None)

    return to_numpy() if callable(to_numpy) else value


def cast(
    value: typing.Any,
    dtype: npt.DTypeLike | None = None,
    *,
    layout: typing.Literal["A", "C", "F"] = "A",
) -> typing.Any:
    """Cast a cmenpy view or array-like to an ndarray with a dtype and layout.

    Buffer views (``pop.X``/``pop.F``) are unwrapped through their explicit
    ``to_numpy`` conversion, so ``cm.cast`` never relies on NumPy's implicit
    ``__array__`` cast.

    Args:
        value: A cmenpy buffer view, an array, a list, or a scalar.
        dtype: Target dtype; ``None`` keeps the original one.
        layout: Storage order for arrays. ``"A"`` (default) leaves the array as
            it is; ``"C"``/``"F"`` return a contiguous array, which is what a
            typed ``cm.Array`` kernel signature expects.

    Returns:
        An ``ndarray`` (or a NumPy scalar when ``value`` is a scalar and a
        ``dtype`` is requested).

    Examples:
        ::

            x = cm.cast(pop.X[:n], self.dtype)
            block = cm.cast(pop.X, float, layout="C")
    """
    value = unwrap_array(value)

    if dtype is None and layout == "A":
        return np.asarray(value)

    _dtype = None if dtype is None else np.dtype(dtype)

    if _dtype is not None and np.ndim(value) == 0:
        return _dtype.type(value)

    if layout == "C":
        return np.ascontiguousarray(value, dtype=_dtype)

    if layout == "F":
        return np.asfortranarray(value, dtype=_dtype)

    return np.asarray(value, dtype=_dtype)


__all__ = ["cast", "unwrap_array"]
