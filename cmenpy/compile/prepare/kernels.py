"""Collecting the numba signatures of the kernels declared on a class."""

import typing

from cmenpy.compile.kernel import _Kernel


def kernels(
    cls: type[typing.Any],
    compile_options: dict[str, typing.Any] | None = None,
) -> dict[str, list[str]]:
    """Return the numba kernels declared on ``cls`` and its bases.

    Names are resolved through the MRO, so an operator mixin inherited by a
    ``@cm.template`` model is reported too, and a subclass redefinition wins
    over the base entry.
    """
    found: dict[str, list[str]] = {}
    seen: set[str] = set()

    for base in cls.__mro__:
        for name, attr in vars(base).items():
            if name in seen:
                continue

            seen.add(name)

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
