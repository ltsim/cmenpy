"""Model declaration API: the function-style `declare` and class-style `template`."""

import inspect
import typing

from cmenpy.model.optimizer import (
    FunctionOptimizerModel,
    ModelProtocol,
    TemplateOptimizerModel,
)
from cmenpy.model.optimizer.functions import AlgorithmFunction
from cmenpy.model.optimizer.variables import (
    install_constants,
    is_constant,
)

#: Compile options a model can set at instantiation; forwarded to its kernels.
_COMPILE_OPTION_KEYS = ("parallel", "fastmath", "cache")


def declare_from_func(
    func: AlgorithmFunction, **kwargs: typing.Any
) -> FunctionOptimizerModel:
    """Bind ``func`` to a fresh `FunctionOptimizerModel` built from ``kwargs``.

    Args:
        func: The ``(args, epoch, ctx) -> None`` algorithm function.
        kwargs: ``name=cm.Argument[...]`` field declarations.

    Returns:
        The model wrapping ``func``.

    Raises:
        TypeError: If any field is a ``cm.Constant``. Constants are a
            class-style feature (``@cm.template`` / ``@cm.module``) and are
            never constructor-bound, so they have no meaning in the function
            style.
    """
    for field_name, field_type in kwargs.items():
        if is_constant(field_type):
            raise TypeError(
                f"Constant field '{field_name}' is not supported by "
                "cm.declare; use a cm.Argument or move the algorithm to a "
                "cm.template class."
            )

    model = FunctionOptimizerModel(**kwargs)
    model.define(func)

    return model


def declare_from_class(cls: type[ModelProtocol]) -> TemplateOptimizerModel:
    """Wrap ``cls`` in a `TemplateOptimizerModel` with a dynamic ``__init__``.

    The generated ``__init__`` extracts the compile options
    (``parallel``/``fastmath``/``cache``), reads the class type hints
    (``Annotated`` metadata carries ranges and defaults), validates the
    arguments and stores them as instance attributes before delegating to
    the original ``__init__``.

    ``cm.Constant`` fields are materialized as read-only class descriptors by
    ``install_constants`` and skipped by the generated ``__init__``: they are
    never constructor-bound, so passing one raises ``TypeError``.

    Args:
        cls: The class to wrap; must define ``initialize`` and ``evolve``.

    Returns:
        The model wrapping ``cls``.

    Raises:
        TypeError: If a ``cm.Constant`` field is passed to the constructor.
    """
    methods_to_check = ["initialize", "evolve"]
    original_init = cls.__init__

    install_constants(cls)

    def dynamic_init(self: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> None:
        compile_options = {
            key: bool(kwargs.pop(key, False)) for key in _COMPILE_OPTION_KEYS
        }
        self._compile_options = compile_options

        hints = typing.get_type_hints(cls, include_extras=True)
        field_names = [
            name
            for name, field_type in hints.items()
            if name not in methods_to_check and not is_constant(field_type)
        ]

        pos_arguments = dict(zip(field_names, args))
        all_arguments = {**pos_arguments, **kwargs}

        for field_name, field_type in hints.items():
            if field_name in methods_to_check:
                continue

            if is_constant(field_type):
                if field_name in all_arguments:
                    raise TypeError(
                        f"Constant field '{field_name}' cannot be passed to the "
                        "constructor; it is fixed at declaration."
                    )

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

    Args:
        *args: Positional field values (bound like ``**kwargs``).
        **kwargs: ``name=cm.Argument[...]`` field declarations.

    Returns:
        A decorator wrapping the function in a `FunctionOptimizerModel`.

    Examples:
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
    ``cm.Variable`` / ``cm.Constant``. ``cm.Constant`` fields are read-only and
    never bound by the constructor. Kernels declared on the class receive the
    model's compile options at instantiation.

    Args:
        model: The class to wrap.

    Returns:
        A model wrapping the class.

    Raises:
        TypeError: If ``model`` is not a class.

    Examples:
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
