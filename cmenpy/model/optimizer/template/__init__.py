import typing

from cmenpy.bounds import Bounds, SequenceStructure
from cmenpy.compile import kernels
from cmenpy.context import Context, MainContextManager
from cmenpy.hints import NDArrayType, ScalarType
from cmenpy.hints.option import SenseType
from cmenpy.model.optimizer.base import BaseOptimizer, build_context_manager
from cmenpy.model.optimizer.functions import CallableFunction
from cmenpy.model.optimizer.template.protocols import ModelProtocol
from cmenpy.target import Target
from cmenpy.tracker import EpochHistory


class TemplateOptimizerModel(BaseOptimizer):
    def __init__(self, cls_model: type[ModelProtocol]) -> None:
        self.__alias = str(cls_model.__name__)
        self.__cls_model: type[ModelProtocol] = cls_model
        self.__inner: CallableFunction | None = None
        self.__model: ModelProtocol | None = None
        self.__resource: MainContextManager | None = None
        self.__seed: int | None = None
        self.__debug = False
        self.__sense: SenseType = "min"

    @property
    def alias(self) -> str:
        return self.__alias

    @property
    def model_class(self) -> type[ModelProtocol]:
        """The class wrapped by :func:`cmenpy.template`."""
        return self.__cls_model

    @property
    def tracker(self) -> list[EpochHistory]:
        if self.__resource is None:
            return []

        return self.__resource.tracker.history

    @property
    def compile_options(self) -> dict[str, bool]:
        """Compile options (``parallel``/``fastmath``/``cache``) of the model."""
        return dict(getattr(self.__model, "_compile_options", {}))

    def prepare(self) -> dict[str, list[str]]:
        """Return the numba kernels declared on the wrapped class.

        Kernels compile on first use; this hook forces compilation with the
        model's compile options and reports their signatures.
        """
        return kernels(self.__cls_model, self.compile_options)

    def solve(
        self,
        f: Target,
        bounds: Bounds | SequenceStructure[ScalarType],
        epochs: int,
        pop_size: int,
        pop_range: tuple[int, int] | None = None,
        debug: bool = False,
        sense: SenseType = "min",
        prepare: bool = False,
    ) -> tuple[float, NDArrayType]:
        if self.__model is None:
            raise NotImplementedError()

        if prepare:
            self.prepare()

        self.__debug = debug
        self.__sense = sense
        self.__resource = build_context_manager(
            f, bounds, epochs, pop_size, pop_range, self.__seed, debug, sense
        )

        if hasattr(self.__model, "initialize"):
            self.__model.initialize(
                ctx=Context(
                    f=self.__resource.target,
                    bounds=self.__resource.bounds,
                    buffer=self.__resource.buffer,
                    generator=self.__resource.default_generator,
                    sense=self.__sense,
                    epoch=self.__resource.epoch_it.current,
                    epochs=self.__resource.epoch_it.max,
                ),
            )

        for e in self.__resource.epoch_it:
            self.__model.evolve(
                e=e,
                ctx=Context(
                    f=self.__resource.target,
                    bounds=self.__resource.bounds,
                    buffer=self.__resource.buffer,
                    generator=self.__resource.default_generator,
                    sense=self.__sense,
                    epoch=self.__resource.epoch_it.current,
                    epochs=self.__resource.epoch_it.max,
                ),
            )

        X = self.__resource.buffer.raw[self.__resource.buffer.idx.best]

        return float(X[0]), X[1:]

    def __call__(
        self, seed: int | None = None, *args: typing.Any, **kwargs: typing.Any
    ) -> "BaseOptimizer":
        self.__seed = seed
        self.__model = self.__cls_model(*args, **kwargs)

        return self
