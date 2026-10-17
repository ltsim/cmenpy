"""Shared decorator factory machinery for the kernel decorators."""

import typing

from cmenpy.compile.kernel import _Kernel
from cmenpy.compile.low.annotate import _annotation_specs, _uses_annotations
from cmenpy.compile.low.cast import _autocast, _elementwise_caster
from cmenpy.hints.primitives import CasterType
from cmenpy.hints.spec import _ArraySpec, _ScalarSpec, _Signature

_SPECS = (_ArraySpec, _ScalarSpec, _Signature)


def _split(
    args: tuple[typing.Any, ...],
) -> tuple[tuple[typing.Any, ...], tuple[typing.Any, ...]]:
    """Separate leading cast markers from the numba options."""
    if args and all(isinstance(a, _SPECS) for a in args):
        return args, ()

    return (), args


def _resolve(
    specs: tuple[typing.Any, ...],
) -> tuple[tuple[typing.Any, ...], list[CasterType | None]]:
    """Turn cast markers into numba arguments and per-parameter casters."""
    if not specs:
        return (), []

    if len(specs) == 1 and isinstance(specs[0], _Signature):
        signature = specs[0]
        signature_casters: list[CasterType | None] = [
            _elementwise_caster(arg.dtype) for arg in signature.args
        ]

        return (signature.numba_signature(),), signature_casters

    parameter_casters: list[CasterType | None] = [
        spec.caster() if spec is not None else None for spec in specs
    ]

    if all(spec is not None and spec.typed for spec in specs):
        types = tuple(spec.numba_type() for spec in specs if spec is not None)

        return (types,), parameter_casters

    return (), parameter_casters


def _apply(
    func: typing.Callable[..., typing.Any],
    specs: tuple[typing.Any, ...],
    options: tuple[typing.Any, ...],
    options_map: dict[str, typing.Any],
    compiler: typing.Callable[..., typing.Any],
    *,
    vectorize_mode: bool = False,
    parallel_mode: str | None = None,
    parallel: bool = False,
) -> "_Kernel":
    """Build a deferred kernel, resolving the specs from arguments/annotations."""
    resolved: tuple[typing.Any, ...] = specs

    if not resolved and not options:
        annotated = _annotation_specs(func)

        if vectorize_mode and _uses_annotations(annotated):
            from cmenpy.compile.vectorize.signature import _vectorize_signature

            resolved = _vectorize_signature(func, annotated)
        else:
            resolved = annotated

    numba_args, casters = _resolve(resolved)

    def make(resolved_options: dict[str, typing.Any]) -> typing.Any:
        compiled = compiler(*numba_args, *options, **resolved_options)(func)

        return _autocast(compiled, casters, func)

    return _Kernel(func, make, options_map, parallel, parallel_mode)


def _template(
    compiler: typing.Callable[..., typing.Any],
    *,
    vectorize_mode: bool = False,
    parallel_mode: str | None = None,
    parallel: bool = False,
    **defaults: typing.Any,
) -> typing.Callable[..., typing.Any]:
    """Build a decorator for ``compiler`` supporting bare, option and spec calls."""

    def decorator(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        specs, options = _split(args)
        options_map = {**defaults, **kwargs}

        if len(options) == 1 and not kwargs and callable(options[0]):
            return _apply(
                options[0],
                specs,
                (),
                options_map,
                compiler,
                vectorize_mode=vectorize_mode,
                parallel_mode=parallel_mode,
                parallel=parallel,
            )

        def wrap(func: typing.Callable[..., typing.Any]) -> typing.Any:
            return _apply(
                func,
                specs,
                options,
                options_map,
                compiler,
                vectorize_mode=vectorize_mode,
                parallel_mode=parallel_mode,
                parallel=parallel,
            )

        return wrap

    return decorator
