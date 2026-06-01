import typing


class Attribute:
    def __class_getitem__(cls, target_type: typing.Any):
        return typing.Annotated[target_type, {"delayed_init": True}]


class Scalar:
    def __class_getitem__(cls, target_type: typing.Any):
        return typing.Annotated[target_type, {"delayed_init": True}]


class Vector:
    def __class_getitem__(cls, target_type: typing.Any):
        return typing.Annotated[target_type, {"delayed_init": True}]


class DefineAgent:
    def __class_getitem__(cls, target_type: typing.Any):
        return typing.Annotated[target_type, {"delayed_init": True}]


def extract_fields(schema: type) -> dict[str, object]:
    return dict(getattr(schema, "__annotations__", {}))
