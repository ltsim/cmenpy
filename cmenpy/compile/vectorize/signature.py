"""Building a complete eager signature for an annotated ``cm.vectorize`` kernel."""

import typing

from cmenpy.compile.low.annotate import _return_spec
from cmenpy.hints.spec import _ArraySpec, _ScalarSpec, _Signature


def _check_vectorize_specs(specs: tuple[typing.Any, ...]) -> None:
    """Validate the eager signature of ``@cm.vectorize`` given positional specs."""
    if len(specs) == 1 and isinstance(specs[0], _Signature):
        return

    if any(getattr(spec, "typed", False) for spec in specs):
        raise TypeError(
            "Precompiled @cm.vectorize needs a full signature, e.g. "
            "cm.vectorize(cm.Scalar[float, [cm.Scalar[float]]])."
        )


def _vectorize_signature(
    func: typing.Callable[..., typing.Any],
    specs: "tuple[_ArraySpec | _ScalarSpec | None, ...]",
) -> tuple[typing.Any, ...]:
    """Build a complete signature for an annotated ``@cm.vectorize`` kernel."""
    if not any(spec is not None and spec.typed for spec in specs):
        return specs

    result = _return_spec(func)

    if (
        result is not None
        and result.typed
        and all(spec is not None and spec.typed for spec in specs)
    ):
        arguments = tuple(spec for spec in specs if spec is not None)

        return (_Signature(result, arguments),)

    raise TypeError(
        "Precompiled @cm.vectorize with annotations needs a typed return, e.g. "
        'def f(x: "Scalar[float]") -> "Scalar[float]": ...'
    )
