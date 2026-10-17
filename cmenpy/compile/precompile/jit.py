"""Deprecated ``cm.jit`` alias of `cm.precompile`."""

import typing
import warnings

from cmenpy.compile.precompile.precompile import precompile


def jit(
    *args: typing.Any,
    fastmath: bool = False,
    parallel: bool = False,
    **kwargs: typing.Any,
) -> typing.Any:
    """Deprecated alias of `precompile`."""
    warnings.warn(
        "cm.jit is deprecated; use cm.precompile instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return precompile(*args, fastmath=fastmath, parallel=parallel, **kwargs)
