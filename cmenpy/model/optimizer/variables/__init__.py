import typing


class Argument:
    def __class_getitem__(cls, params):
        if isinstance(params, tuple) and len(params) == 3:
            target_type, (min_value, max_value), default_val = params

            return typing.Annotated[
                target_type,
                {"min": min_value, "max": max_value, "default": default_val},
            ]
        elif isinstance(params, tuple) and len(params) == 2:
            target_type, (min_value, max_value) = params

            return typing.Annotated[target_type, {"min": min_value, "max": max_value}]

        raise TypeError(
            "The second argument of Validator must be a tuple of [(min, max)] or ([min, max], default)."
        )


class Variable:
    def __class_getitem__(cls, target_type):
        return typing.Annotated[target_type, {"delayed_init": True}]
