"""The ``cm.reduce`` decorator: generalized-ufunc kernels from specs or annotations."""

import typing

import numba as nb

from cmenpy.compile.kernel import _Kernel
from cmenpy.compile.low.annotate import _annotation_specs, _uses_annotations
from cmenpy.compile.low.cast import _autocast
from cmenpy.compile.low.decorate import _SPECS
from cmenpy.compile.reduce.contract_spec import _ContractSpec
from cmenpy.hints.primitives import CasterType


def _parse_signatures(
    signatures: typing.Any,
) -> list[tuple[typing.Any, ...]]:
    """Normalize a flat list of specs into a list of core signatures."""
    if isinstance(signatures, _SPECS):
        signatures = [signatures]

    items = list(signatures)

    if not items:
        raise TypeError("cm.reduce requires at least one signature.")

    if all(isinstance(item, _SPECS) for item in items):
        return [tuple(items)]

    return [tuple(signature) for signature in items]


def _annotated_signature(
    func: typing.Callable[..., typing.Any],
) -> list[tuple[typing.Any, ...]]:
    """Derive a single core signature from the annotations of ``func``."""
    specs = _annotation_specs(func)

    if not _uses_annotations(specs):
        raise TypeError(
            "cm.reduce needs signatures or fully typed parameter annotations."
        )

    if any(spec is None or not spec.typed for spec in specs):
        raise TypeError(
            "cm.reduce requires fully typed parameter annotations, e.g. "
            'def clip(row: "Array[float, 1]", out: "Array[float, 1]"): ...'
        )

    return [tuple(spec for spec in specs if spec is not None)]


def _contract_from_annotations(
    func: typing.Callable[..., typing.Any],
) -> _ContractSpec:
    """Infer a `Contract` signature from the annotations of ``func``.

    Every parameter must be annotated; the last one is the output.
    """
    specs = _annotation_specs(func)

    if not _uses_annotations(specs):
        raise TypeError(
            "cm.reduce needs a Contract, explicit signatures, or annotations."
        )

    if any(spec is None for spec in specs):
        raise TypeError(
            "cm.reduce requires every parameter to be annotated, e.g. "
            "def matmul(a: \"Array[float, 'm,n']\", b: \"Array[float, 'n,k']\", "
            "out: \"Array[float, 'm,k']\"): ..."
        )

    resolved = [spec for spec in specs if spec is not None]

    if len(resolved) < 2:
        raise TypeError("cm.reduce needs at least one input and one output.")

    return _ContractSpec(tuple(resolved[:-1]), resolved[-1])


def _resolve_reduce(
    parsed: list[tuple[typing.Any, ...]],
    layout: str,
) -> tuple[list[tuple[typing.Any, ...]], list[CasterType]]:
    """Validate fully typed specs and build the numba types and casters."""
    for signature in parsed:
        for spec in signature:
            if not getattr(spec, "typed", False):
                raise TypeError(
                    "cm.reduce requires fully typed specs, e.g. "
                    "cm.Array[float, 1] or cm.Scalar[float]."
                )

    ftylist = [tuple(spec.numba_type() for spec in sig) for sig in parsed]
    n_inputs = _count_inputs(layout)
    first = parsed[0]

    if n_inputs > len(first):
        raise TypeError(
            f"Layout '{layout}' expects {n_inputs} inputs, "
            f"but {len(first)} specs were given."
        )

    casters = [first[index].caster() for index in range(n_inputs)]

    return ftylist, casters


def _count_inputs(layout: str) -> int:
    left = layout.split("->")[0].strip()

    if not left:
        return 0

    depth = 0
    count = 1

    for ch in left:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            count += 1

    return count


def reduce(
    signatures: typing.Any = None,
    layout: str | None = None,
    *,
    fastmath: bool = False,
    parallel: bool = False,
    **kwargs: typing.Any,
) -> typing.Any:
    """Build a generalized ufunc (``numba.guvectorize``) static method.

    Three additive forms are supported:

    * Legacy numba style: a list of core type specs plus a layout, e.g.
      ``[cm.Array[float, 1], cm.Scalar[float], cm.Array[float, 1]]`` with
      ``"(n),()->(n)"``.
    * The `Contract` marker, which encodes the core types and layout::

          @cm.reduce(Contract[[Array[float, "m,n"],
                                     Array[float, "n,k"]],
                                    Array[float, "m,k"]])
          def matmul(a, b, out): ...

    * Fully typed annotations, with the layout inferred (the last parameter is
      the output)::

          @cm.reduce
          def matmul(a: "Array[float, 'm,n']",
                     b: "Array[float, 'n,k']",
                     out: "Array[float, 'm,k']"): ...

    ``fastmath`` maps to the numba option and ``parallel`` sets
    ``target="parallel"``; any other option is forwarded as a keyword. A model
    can also provide ``fastmath``/``parallel``/``cache`` at instantiation.
    """
    options = {**kwargs, "fastmath": fastmath}

    if layout is None and isinstance(signatures, str):
        signatures, layout = None, signatures

    func: typing.Callable[..., typing.Any] | None = None

    if callable(signatures) and not isinstance(signatures, _ContractSpec):
        func, signatures = signatures, None

    inferred = signatures is None and layout is None

    if not isinstance(signatures, _ContractSpec) and not inferred and layout is None:
        raise TypeError("cm.reduce requires a layout, e.g. '(n),()->(n)'.")

    def wrap(target: typing.Callable[..., typing.Any]) -> typing.Any:
        casters: list[CasterType]

        if isinstance(signatures, _ContractSpec):
            ftylist = [signatures.numba_types]
            compiled_layout = signatures.layout
            casters = signatures.casters
        elif inferred:
            contract = _contract_from_annotations(target)
            ftylist = [contract.numba_types]
            compiled_layout = contract.layout
            casters = contract.casters
        elif signatures is None:
            assert layout is not None
            ftylist, casters = _resolve_reduce(_annotated_signature(target), layout)
            compiled_layout = layout
        else:
            assert layout is not None
            ftylist, casters = _resolve_reduce(_parse_signatures(signatures), layout)
            compiled_layout = layout

        def make(resolved_options: dict[str, typing.Any]) -> typing.Any:
            compiled = nb.guvectorize(  # type: ignore[no-untyped-call]
                ftylist, compiled_layout, **resolved_options
            )(target)

            return _autocast(compiled, casters, target)

        return _Kernel(target, make, options, parallel, "target")

    if func is not None:
        return wrap(func)

    return wrap
