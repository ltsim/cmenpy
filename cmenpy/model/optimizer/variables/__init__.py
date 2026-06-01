import typing


class Argument:
    def __class_getitem__(cls, params):
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
    def __class_getitem__(cls, target_type):
        return typing.Annotated[target_type, {"delayed_init": True}]
