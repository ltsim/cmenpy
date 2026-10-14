import inspect
import typing

from cmenpy.model.optimizer import (
    FunctionOptimizerModel,
    ModelProtocol,
    TemplateOptimizerModel,
)
from cmenpy.model.optimizer.functions import AlgorithmFunction

#: Compile options a model can set at instantiation; forwarded to its kernels.
_COMPILE_OPTION_KEYS = ("parallel", "fastmath", "cache")


def declare_from_func(
    func: AlgorithmFunction, **kwargs: typing.Any
) -> FunctionOptimizerModel:
    model = FunctionOptimizerModel(**kwargs)
    model.define(func)

    return model


def declare_from_class(cls: type[ModelProtocol]) -> TemplateOptimizerModel:
    methods_to_check = ["initialize", "evolve"]
    original_init = cls.__init__

    def dynamic_init(self: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> None:
        compile_options = {
            key: bool(kwargs.pop(key, False)) for key in _COMPILE_OPTION_KEYS
        }
        self._compile_options = compile_options

        hints = typing.get_type_hints(cls, include_extras=True)
        field_names = [k for k in hints if k not in methods_to_check]

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

    cls.__init__ = dynamic_init  # type: ignore[method-assign]

    return TemplateOptimizerModel(
        cls_model=cls,
    )


def declare(
    *args: typing.Any,
    **kwargs: typing.Any,
) -> typing.Callable[[AlgorithmFunction], FunctionOptimizerModel]:
    """Turn a function into a model (the function-style API).

    The decorated function receives ``(args, epoch, ctx)`` and runs the
    whole optimization inside its body. Parameters are declared with
    ``cm.Argument`` and bound by calling the returned model.

    Parameters
    ----------
    **kwargs
        ``name=cm.Argument[...]`` field declarations.

    Returns
    -------
    callable
        A decorator wrapping the function in a `FunctionOptimizerModel`.

    Examples
    --------
    ::

        @cm.declare(x=cm.Argument[int, (0, 3), 2])
        def my_algorithm(args, epoch, ctx): ...

        model = my_algorithm(x=1)
    """
    def wrapper(target: AlgorithmFunction) -> FunctionOptimizerModel:
        if not inspect.isfunction(target):
            raise TypeError(f"Model '{target}' is not a function.")

        return declare_from_func(target, **kwargs)

    return wrapper


def template(model: type[ModelProtocol]) -> TemplateOptimizerModel:
    """Turn a class into a model (the class-style API).

    The class must define ``initialize(self, ctx)`` and
    ``evolve(self, e, ctx)``; fields are declared with ``cm.Argument`` /
    ``cm.Variable``. Kernels declared on the class receive the model's
    compile options at instantiation.

    Parameters
    ----------
    model : type
        The class to wrap.

    Returns
    -------
    TemplateOptimizerModel
        A model wrapping the class.

    Raises
    ------
    TypeError
        If ``model`` is not a class.

    Examples
    --------
    ::

        @cm.template
        class MyAlgorithm:
            a: cm.Argument[int, (1, 100)]

            def initialize(self, ctx) -> None: ...
            def evolve(self, e, ctx) -> None: ...
    """
    if not isinstance(model, type):
        raise TypeError(f"Model '{model}' is not a class.")

    return declare_from_class(model)
