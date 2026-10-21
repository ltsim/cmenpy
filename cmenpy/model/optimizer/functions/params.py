"""Argument declaration exceptions and the bound-values holder."""

import typing

import numpy as np

ANNOTATED_KIND = typing.get_origin(typing.Annotated[int, int])


class ExceptionValueParams(Exception):
    """Raised when an argument value falls outside its declared range."""


class ExceptionTypeParams(Exception):
    """Raised when an argument value does not match its declared type."""


def _validate_range_and_type(
    value: typing.Any, field_type: typing.Any, field_name: str
) -> None:
    """Check ``value`` against the declared type and range of ``field_name``.

    Args:
        value: The value being assigned.
        field_type: The declared type; ``Annotated`` metadata may carry
            ``min``/``max``.
        field_name: The field name, used in error messages.

    Raises:
        ExceptionTypeParams: If the type does not match the declaration.
        ExceptionValueParams: If the value falls outside the declared range.
    """
    if typing.get_origin(field_type) is ANNOTATED_KIND:
        type_args = typing.get_args(field_type)
        base_type, metadata = type_args[0], type_args[1]
    else:
        base_type, metadata = field_type, {}

    actual_origin = typing.get_origin(base_type) or base_type
    if not isinstance(value, actual_origin):
        raise ExceptionTypeParams(
            f"Invalid type for '{field_name}'. "
            f"Expected {actual_origin.__name__}, got {type(value).__name__}"
        )

    min_val = metadata.get("min")
    max_val = metadata.get("max")

    if min_val is not None and max_val is not None:
        if isinstance(value, np.ndarray):
            if not np.all((value >= min_val) & (value <= max_val)):
                raise ExceptionValueParams(
                    f"Array elements in '{field_name}' out of range [{min_val}, {max_val}]."
                )
            return

        if isinstance(value, (list, tuple, set)):
            for item in value:
                if not (min_val <= item <= max_val):
                    raise ExceptionValueParams(
                        f"Element {item} in collection '{field_name}' out of range [{min_val}, {max_val}]."
                    )
            return

        if not (min_val <= value <= max_val):
            raise ExceptionValueParams(
                f"Value {value} for '{field_name}' out of range [{min_val}, {max_val}]."
            )


class FunctionParamsArguments:
    """Holds the declared ``cm.Argument`` fields and their bound values.

    Args:
        type_dict: Mapping of field name to declared type (``Annotated``
            metadata carries the range/default).

    Raises:
        AttributeError: On loading a name that was never declared, or
            reading an uninitialized declared field.
        ExceptionValueParams: If a value falls outside its declared range.
        ExceptionTypeParams: If a value does not match its declared type.
    """

    def __init__(self, type_dict: dict[str, type[typing.Any]]) -> None:
        """Store the declared field names and types."""
        self._fields = dict(type_dict)

    def load(self, values_dict: dict[str, typing.Any]) -> None:
        """Bind ``values_dict`` into the declared fields.

        Args:
            values_dict: Mapping of field name to value.

        Raises:
            AttributeError: If a name was never declared.
        """
        for name, value in values_dict.items():
            if name not in self._fields:
                raise AttributeError(
                    f"Cannot initialize '{name}': It was not declared in MyArguments."
                )
            setattr(self, name, value)

    def __setattr__(self, name: str, value: typing.Any) -> None:
        """Validate and store ``value`` for the declared field ``name``."""
        field_type = getattr(self, "_fields", {}).get(name)

        if field_type is not None:
            _validate_range_and_type(value, field_type, name)

        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> typing.Any:
        """Read a declared field, or raise for unknown/uninitialized names."""
        if name in self.__dict__.get("_fields", {}):
            raise AttributeError(
                f"Attribute '{name}' has been declared but not initialized yet."
            )
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
