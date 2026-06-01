import typing

ANNOTATED_KIND = typing.get_origin(typing.Annotated[int, int])


def _validate_range_and_type(value, field_type, field_name):
    if typing.get_origin(field_type) is ANNOTATED_KIND:
        type_args = typing.get_args(field_type)
        base_type, metadata = type_args[0], type_args[1]
    else:
        base_type, metadata = field_type, {}

    actual_origin = typing.get_origin(base_type) or base_type
    if not isinstance(value, actual_origin):
        raise TypeError(
            f"Invalid type for '{field_name}'. "
            f"Expected {actual_origin.__name__}, got {type(value).__name__}"
        )

    min_val = metadata.get("min")
    max_val = metadata.get("max")

    if min_val is not None and max_val is not None:
        try:
            import numpy as np

            has_numpy = True
        except ImportError:
            has_numpy = False

        if has_numpy and isinstance(value, np.ndarray):
            if not np.all((value >= min_val) & (value <= max_val)):
                raise ValueError(
                    f"Array elements in '{field_name}' out of range [{min_val}, {max_val}]."
                )
            return

        if isinstance(value, (list, tuple, set)):
            for item in value:
                if not (min_val <= item <= max_val):
                    raise ValueError(
                        f"Element {item} in collection '{field_name}' out of range [{min_val}, {max_val}]."
                    )
            return

        if not (min_val <= value <= max_val):
            raise ValueError(
                f"Value {value} for '{field_name}' out of range [{min_val}, {max_val}]."
            )


class FunctionParamsArguments:
    def __init__(self, type_dict: dict[str, typing.Type[typing.Any]]) -> None:
        self.__annotations__ = {
            attr_name: attr_type for attr_name, attr_type in type_dict.items()
        }

    def load(self, values_dict: dict[str, typing.Any]) -> None:
        for name, value in values_dict.items():
            if name not in self.__annotations__:
                raise AttributeError(
                    f"Cannot initialize '{name}': It was not declared in MyArguments."
                )
            setattr(self, name, value)

    def __setattr__(self, name: str, value: typing.Any) -> None:
        if (
            name != "__annotations__"
            and hasattr(self, "__annotations__")
            and name in self.__annotations__
        ):
            field_type = self.__annotations__[name]
            _validate_range_and_type(value, field_type, name)

        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> typing.Any:
        if name in self.__dict__.get("__annotations__", {}):
            raise AttributeError(
                f"Attribute '{name}' has been declared but not initialized yet."
            )
        raise AttributeError(f"'FunctionParams' object has no attribute '{name}'")
