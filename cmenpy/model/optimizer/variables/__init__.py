"""Field declaration markers: `Argument` (tunable), `Constant` (fixed) and `Variable`."""

import typing

ANNOTATED_KIND = typing.get_origin(typing.Annotated[int, int])


class _ConstantValue:
    """Read-only descriptor holding a declared constant value.

    A ``cm.Constant`` is materialized on the class as one of these so it is
    available through ``self.name`` and ``Cls.name`` while rejecting any attempt
    to reassign or delete it from an instance.
    """

    __slots__ = ("_value",)

    def __init__(self, value: typing.Any) -> None:
        """Store the fixed ``value``."""
        self._value = value

    def __get__(
        self, instance: typing.Any, owner: type[typing.Any] | None = None
    ) -> typing.Any:
        """Return the constant value through an instance or the class."""
        return self._value

    def __set__(self, instance: typing.Any, value: typing.Any) -> None:
        """Reject reassignment of a constant."""
        raise AttributeError("Cannot reassign a constant field.")

    def __delete__(self, instance: typing.Any) -> None:
        """Reject deletion of a constant."""
        raise AttributeError("Cannot delete a constant field.")


def _field_metadata(field_type: typing.Any) -> dict[str, typing.Any] | None:
    """Return the ``dict`` metadata carried by an ``Annotated`` type, if any."""
    if typing.get_origin(field_type) is ANNOTATED_KIND:
        type_args = typing.get_args(field_type)

        if len(type_args) >= 2 and isinstance(type_args[1], dict):
            return type_args[1]

    return None


def _check_constant_type(target_type: typing.Any, value: typing.Any) -> None:
    """Best-effort type check of a constant value at declaration time.

    Exotic typing forms (``Union``, ``Literal``, parametrized generics) are
    skipped so they never raise on a value that is valid for them.

    Raises:
        TypeError: If ``value`` clearly violates the declared ``target_type``.
    """
    origin = typing.get_origin(target_type)

    if origin in (typing.Union, typing.Literal):
        return

    check_type = origin if origin is not None else target_type

    if not isinstance(check_type, type):
        return

    if isinstance(value, check_type):
        return

    # Numeric widening mirrors the casters used by the field markers.
    if check_type is float and isinstance(value, (int, float)):
        return

    raise TypeError(f"Constant value {value!r} does not match type {target_type!r}.")


def is_constant(field_type: typing.Any) -> bool:
    """Return whether ``field_type`` was declared with ``cm.Constant``.

    Args:
        field_type: A resolved annotation (possibly ``Annotated``).

    Returns:
        ``True`` when the annotation carries ``cm.Constant`` metadata.
    """
    metadata = _field_metadata(field_type)

    return metadata is not None and "constant" in metadata


def install_constants(cls: type[typing.Any]) -> None:
    """Materialize every ``cm.Constant`` annotation on ``cls`` as a descriptor.

    Inherited annotations are collected with ``typing.get_type_hints`` (which
    walks the MRO), so a constant declared on a base class or operator mixin is
    installed on ``cls`` too, and a subclass may redeclare the name to override
    the value.

    Args:
        cls: The class whose ``cm.Constant`` annotations are installed.
    """
    hints = typing.get_type_hints(cls, include_extras=True)

    for name, field_type in hints.items():
        metadata = _field_metadata(field_type)

        if metadata is not None and "constant" in metadata:
            setattr(cls, name, _ConstantValue(metadata["constant"]))


class Argument:
    """Declare a tunable, range-checked model field.

    ``Argument`` is a subscript-only marker that builds a
    ``typing.Annotated`` type understood by ``@cm.declare`` /
    ``@cm.template``. It is never instantiated — only subscripted on a class
    annotation. Supported forms::

        Argument[type]                  # required, no range/default
        Argument[type, default]         # with a default value
        Argument[type, (min, max)]      # range-checked, required
        Argument[type, (min, max), default]

    The subscript accepts ``target_type`` (the field's type), optionally a
    ``(min_val, max_val)`` range, and optionally a ``default_val``. An
    unsupported form raises ``TypeError``.
    """

    def __class_getitem__(cls, params: typing.Any) -> typing.Any:
        """Build the ``Annotated`` field type from the subscript ``params``.

        Raises:
            TypeError: If the subscript form is not supported.
        """
        if not isinstance(params, tuple):
            params = (params,)

        match params:
            case (target_type, (min_val, max_val), default_val):
                return typing.Annotated[
                    target_type,
                    {"min": min_val, "max": max_val, "default": default_val},
                ]

            case (target_type, default_val) if (
                not isinstance(default_val, tuple) or len(default_val) != 2
            ):
                return typing.Annotated[target_type, {"default": default_val}]

            case (target_type, (min_val, max_val)):
                return typing.Annotated[target_type, {"min": min_val, "max": max_val}]

            case (target_type,):
                return typing.Annotated[target_type, {"delayed_init": True}]

            case _:
                raise TypeError(
                    "Invalid MyType syntax. Supported formats:\n"
                    "  Argument[type]\n"
                    "  Argument[type, default]\n"
                    "  Argument[type, (min, max)]\n"
                    "  Argument[type, (min, max), default]"
                )


class Constant:
    """Declare a fixed, read-only field for templates and operator mixins.

    ``Constant`` is a subscript-only marker that builds a ``typing.Annotated``
    type understood by ``@cm.template`` and ``@cm.module``. The declared value
    is installed as a class-level descriptor, so it is visible as
    ``self.name``/``Cls.name`` but is never a constructor parameter::

        Constant[type, value]    # a fixed value checked once at declaration

    Constants work the same way on a ``@cm.template`` model and on a
    ``@cm.module`` mixin, where every implementing model inherits them.
    """

    def __class_getitem__(cls, params: typing.Any) -> typing.Any:
        """Build the ``Annotated`` constant type from the subscript ``params``.

        Raises:
            TypeError: If the subscript is not ``[type, value]``, or if the
                value does not match the declared type.
        """
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError(
                "Invalid Constant syntax. Supported format:\n  Constant[type, value]"
            )

        target_type, value = params
        _check_constant_type(target_type, value)

        return typing.Annotated[target_type, {"constant": value}]


class Variable:
    """Declare a free internal-state field (no range check, no default).

    ``Variable`` is a subscript-only marker for model state that is not a
    tunable parameter::

        w: cm.Variable[np.ndarray]

    It wraps ``target_type`` (the field's declared type) in
    ``typing.Annotated`` metadata marking it as delayed-init state.
    """

    def __class_getitem__(cls, target_type: typing.Any) -> typing.Any:
        """Build the ``Annotated`` field type wrapping ``target_type``."""
        return typing.Annotated[target_type, {"delayed_init": True}]
