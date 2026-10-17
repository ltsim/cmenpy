"""The ``cm.vectorize`` decorator and its eager-signature validation."""

import typing

import numba as nb

from cmenpy.compile.low.decorate import _SPECS, _template
from cmenpy.compile.vectorize.signature import _check_vectorize_specs


def vectorize(
    *args: typing.Any,
    fastmath: bool = False,
    parallel: bool = False,
    **kwargs: typing.Any,
) -> typing.Any:
    """Vectorize a static method, optionally precompiling it.

    Specs can be passed positionally (``cm.Array``, ``cm.Scalar[float, [...]]``)
    or declared as annotations in the decorated function::

        @cm.vectorize
        def affine(r: "Scalar[float]", a: "Scalar[float]") -> "Scalar[float]": ...

    Precompilation needs a full signature because numba requires the return
    type. ``fastmath`` and ``parallel`` can be forced on the kernel, and a model
    can also provide them for every kernel at instantiation. Any other option is
    forwarded as a keyword.
    """
    if args and all(isinstance(a, _SPECS) for a in args):
        _check_vectorize_specs(args)

    return _template(
        nb.vectorize,
        vectorize_mode=True,
        parallel_mode="target",
        parallel=parallel,
        fastmath=fastmath,
        **kwargs,
    )(*args)
