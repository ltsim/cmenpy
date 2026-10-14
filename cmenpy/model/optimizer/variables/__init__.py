import typing


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


class Variable:
    """Declare a free internal-state field (no range check, no default).

    ``Variable`` is a subscript-only marker for model state that is not a
    tunable parameter::

        w: cm.Variable[np.ndarray]

    It wraps ``target_type`` (the field's declared type) in
    ``typing.Annotated`` metadata marking it as delayed-init state.
    """
    def __class_getitem__(cls, target_type: typing.Any) -> typing.Any:
        return typing.Annotated[target_type, {"delayed_init": True}]
