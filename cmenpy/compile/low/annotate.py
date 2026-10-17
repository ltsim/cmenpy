"""Reading cast specs from function annotations (markers, strings, Annotated)."""

import inspect
import typing

from cmenpy.hints.spec import Array, Scalar, _ArraySpec, _ScalarSpec


def _annotation_namespace(
    func: typing.Callable[..., typing.Any],
) -> dict[str, typing.Any]:
    """Namespace used to resolve string annotations such as ``"Array[float, 2]"``."""
    return {
        **getattr(func, "__globals__", {}),
        "Array": Array,
        "Scalar": Scalar,
    }


def _to_spec(
    value: typing.Any,
    namespace: dict[str, typing.Any],
    depth: int = 0,
) -> "_ArraySpec | _ScalarSpec | None":
    """Normalize a single annotation into a cast spec, or ``None`` if it is not one.

    Accepts an already-built marker, a string such as ``"Array[float, 2]"`` and a
    ``typing.Annotated[...]`` whose metadata carries the marker. Anything else
    (plain type hints, ``None``, ...) is ignored.
    """
    if isinstance(value, (_ArraySpec, _ScalarSpec)):
        return value

    if isinstance(value, str):
        if depth >= 4:
            return None

        try:
            resolved = eval(value, dict(namespace))
        # An arbitrary annotation expression may raise anything; ignore non-specs.
        except Exception:  # noqa: BLE001
            return None

        return _to_spec(resolved, namespace, depth + 1)

    if typing.get_origin(value) is typing.Annotated:
        for metadata in typing.get_args(value)[1:]:
            spec = _to_spec(metadata, namespace, depth + 1)

            if spec is not None:
                return spec

    return None


def _function_hints(
    func: typing.Callable[..., typing.Any],
) -> dict[str, typing.Any]:
    """Resolve the annotations of ``func``, keeping ``Annotated`` metadata."""
    try:
        return typing.get_type_hints(func, include_extras=True)
    # Unresolvable plain-string specs (e.g. ``"Array[float, 2]"``) fail here.
    except Exception:  # noqa: BLE001
        return dict(getattr(func, "__annotations__", {}))


def _annotation_specs(
    func: typing.Callable[..., typing.Any],
) -> "tuple[_ArraySpec | _ScalarSpec | None, ...]":
    """Read the per-parameter cast specs from the annotations of ``func``."""
    hints = _function_hints(func)
    namespace = _annotation_namespace(func)
    specs: list[_ArraySpec | _ScalarSpec | None] = []

    for parameter in inspect.signature(func).parameters.values():
        annotation = hints.get(parameter.name, parameter.annotation)
        specs.append(_to_spec(annotation, namespace))

    return tuple(specs)


def _return_spec(
    func: typing.Callable[..., typing.Any],
) -> "_ArraySpec | _ScalarSpec | None":
    """Read the cast spec from the return annotation of ``func``, if any."""
    annotation = _function_hints(func).get("return")

    if annotation is None:
        return None

    return _to_spec(annotation, _annotation_namespace(func))


def _uses_annotations(
    specs: "tuple[_ArraySpec | _ScalarSpec | None, ...]",
) -> bool:
    return any(spec is not None for spec in specs)
