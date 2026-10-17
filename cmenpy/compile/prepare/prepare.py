"""`cm.prepare`: force/report precompilation of a model's or class's kernels."""

import typing

from cmenpy.compile.prepare.kernels import kernels


def prepare(target: typing.Any) -> dict[str, list[str]]:
    """Force/report the precompilation of the kernels of a model or class."""
    runner = getattr(target, "prepare", None)

    if callable(runner):
        return typing.cast(dict[str, list[str]], runner())

    if isinstance(target, type):
        return kernels(target)

    raise TypeError(f"Cannot prepare '{target}'.")
