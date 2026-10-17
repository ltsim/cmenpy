"""Collecting the numba signatures of the kernels declared on a class."""

import typing

from cmenpy.compile.kernel import _Kernel


def kernels(
    cls: type[typing.Any],
    compile_options: dict[str, typing.Any] | None = None,
) -> dict[str, list[str]]:
    """Return the numba kernels declared on ``cls``."""
    found: dict[str, list[str]] = {}

    for name, attr in vars(cls).items():
        if isinstance(attr, _Kernel):
            found[name] = attr.signatures(compile_options)
            continue

        func = attr.__func__ if isinstance(attr, staticmethod) else attr
        numba_obj = getattr(func, "__numba__", func)

        # Numba marks plain Python functions with a ``"py_func"`` string.
        if isinstance(numba_obj, str):
            numba_obj = func

        signatures = getattr(numba_obj, "signatures", None)

        if signatures is None:
            signatures = getattr(numba_obj, "types", None)

        if signatures is None:
            continue

        found[name] = [str(signature) for signature in signatures]

    return found
