import inspect
import typing

from cmenpy.model.optimizer import (
    ModelProtocol,
    TemplateOptimizerModel,
    FunctionOptimizerModel,
)
from cmenpy.model.optimizer.base import BaseOptimizer
from cmenpy.model.optimizer.functions import CallableFunction, AlgorithmFunction


def check_type_protocol_from_class(
    cls: typing.Type[ModelProtocol], strict: bool = False
) -> typing.Type[ModelProtocol]:
    methods_to_check = ["initialize", "evolve"]

    protocol_hints = {
        m: typing.get_type_hints(getattr(ModelProtocol, m)) for m in methods_to_check
    }
    protocol_sigs = {
        m: inspect.signature(getattr(ModelProtocol, m)) for m in methods_to_check
    }

    for method_name in methods_to_check:
        if not hasattr(cls, method_name) or not inspect.isfunction(
            getattr(cls, method_name)
        ):
            raise TypeError(
                f"Class '{cls.__name__}' must implement the method '{method_name}'"
            )

        cls_method = getattr(cls, method_name)
        cls_method_hints = typing.get_type_hints(cls_method)
        cls_sig = inspect.signature(cls_method)

        proto_sig = protocol_sigs[method_name]
        proto_hints = protocol_hints[method_name]

        proto_params = [p for p in proto_sig.parameters if p != "self"]
        cls_params = [p for p in cls_sig.parameters if p != "self"]

        if strict:
            if proto_params != cls_params:
                raise TypeError(
                    f"Method '{method_name}' in '{cls.__name__}' has incorrect parameter order or names. "
                    f"Expected sequence: {proto_params}, but got: {cls_params}"
                )

            for param_name in proto_params:
                if param_name not in cls_method_hints:
                    raise TypeError(
                        f"Method '{method_name}' parameter '{param_name}' is missing a type hint"
                    )
                if cls_method_hints[param_name] != proto_hints[param_name]:
                    raise TypeError(
                        f"Method '{method_name}' parameter '{param_name}' must be of type "
                        f"{proto_hints[param_name].__name__}, not {cls_method_hints[param_name].__name__}"
                    )
        else:
            for param_name in proto_params:
                if param_name not in cls_params:
                    raise TypeError(
                        f"Method '{method_name}' in '{cls.__name__}' is missing "
                        f"the required parameter '{param_name}'"
                    )

    return cls


def declare_from_func(func: AlgorithmFunction, **kwargs):
    model = FunctionOptimizerModel(**kwargs)

    return model.define(func)


def declare_from_class(cls: typing.Type[ModelProtocol]):
    methods_to_check = ["initialize", "evolve"]
    original_init = cls.__init__

    def dynamic_init(self, *args, **kwargs):
        hints = typing.get_type_hints(cls)
        field_names = [k for k in hints.keys() if k not in methods_to_check]

        pos_arguments = dict(zip(field_names, args))
        all_arguments = {**pos_arguments, **kwargs}

        for field_name, field_type in hints.items():
            if field_name in methods_to_check:
                continue

            value = all_arguments.get(field_name)

            if typing.get_origin(field_type) is typing.Annotated:
                base_type, metadata = (
                    typing.get_args(field_type)[0],
                    typing.get_args(field_type)[1],
                )

                if (
                    isinstance(metadata, dict)
                    and "min" in metadata
                    and "max" in metadata
                ):
                    if value is None and "default" in metadata:
                        value = metadata["default"]

                    if value is None:
                        raise TypeError(f"Missing required argument: '{field_name}'")

                    if not isinstance(value, base_type):
                        raise TypeError(
                            f"Field '{field_name}' must be of type {base_type.__name__}"
                        )

                    if not (metadata["min"] <= value <= metadata["max"]):
                        raise ValueError(
                            f"Value {value} out of range for '{field_name}'"
                        )

            setattr(self, field_name, value)

        if original_init is not object.__init__:
            original_init(self, *args, **kwargs)

    setattr(cls, "__init__", dynamic_init)

    return TemplateOptimizerModel(
        cls_model=cls,
    )


def template_check(
    strict: bool,
) -> typing.Union[
    BaseOptimizer, typing.Callable[[typing.Type[ModelProtocol]], BaseOptimizer]
]:
    def wrapper(target: typing.Type[ModelProtocol]):
        if not isinstance(target, type):
            raise TypeError(f"Model '{target}' is not a function.")

        return declare_from_class(check_type_protocol_from_class(target, strict))

    return wrapper


def declare(
    **kwargs: typing.Any,
) -> typing.Callable[[AlgorithmFunction], BaseOptimizer]:
    def wrapper(target: AlgorithmFunction):
        if not inspect.isfunction(target):
            raise TypeError(f"Model '{target}' is not a function.")

        return declare_from_func(target, **kwargs)

    return wrapper


def template(model: typing.Type[ModelProtocol]) -> BaseOptimizer:
    if not isinstance(model, type):
        raise TypeError(f"Model '{model}' is not a class.")

    return declare_from_class(model)
