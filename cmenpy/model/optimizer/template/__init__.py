"""The class-style model (`TemplateOptimizerModel`) built by ``@cm.template``."""

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
    """Model built by ``@cm.template`` from a class.

    The class supplies ``initialize``/``evolve``; compile options
    (``numba``/``parallel``/``fastmath``/``cache``) are set at instantiation
    and propagate to the class's numba kernels. ``prepare=True`` compiles
    them eagerly at construction instead of on first use.

    Args:
        cls_model: The wrapped class implementing ``initialize``/``evolve``.

    Attributes:
        alias: The wrapped class name.
        model_class: The wrapped class.
        compile_options: Resolved compile options of the model.
        tracker: Per-epoch history (empty before the first ``solve``).
    """

    def __init__(self, cls_model: type[ModelProtocol]) -> None:
        """Initialize the model state around the wrapped class."""
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
        """The wrapped class name."""
        return self.__alias

    @property
    def model_class(self) -> type[ModelProtocol]:
        """The class wrapped by `cmenpy.template`."""
        return self.__cls_model

    @property
    def tracker(self) -> list[EpochHistory]:
        """Per-epoch history (empty before the first ``solve``)."""
        if self.__resource is None:
            return []

        return self.__resource.tracker.history

    @property
    def compile_options(self) -> dict[str, bool]:
        """Compile options (``numba``/``parallel``/``fastmath``/``cache``)."""
        return dict(getattr(self.__model, "_compile_options", {}))

    def prepare(self) -> dict[str, list[str]]:
        """Return the numba kernels declared on the wrapped class.

        Kernels compile on first use; this hook forces compilation with the
        model's compile options and reports their signatures. Kernels running
        uncompiled (``numba=False`` or numba not installed) report the
        ``"<python>"`` marker. Pass ``prepare=True`` at construction to run
        this automatically during instantiation.
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
        """Run the optimization and return the best ``(fitness, solution)``.

        Calls ``initialize`` once, then ``evolve`` per epoch.

        Args:
            f: The objective callable.
            bounds: The search space (or bounds sequence).
            epochs: Total number of epochs.
            pop_size: Population size.
            pop_range: ``(min, max)`` allowed population sizes.
            debug: Record per-epoch history.
            sense: ``"min"`` (default) or ``"max"``.
            prepare: Force compilation of the class's kernels before running.

        Returns:
            The best fitness and the best solution vector found.

        Raises:
            NotImplementedError: If the model was not instantiated first.
        """
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
        """Instantiate the wrapped class (binding parameters and seed).

        Args:
            seed: Random seed forwarded to the run.
            *args: Positional field values for the wrapped class.
            **kwargs: Named field values (plus compile options and
                ``prepare``) for the wrapped class.
        """
        self.__seed = seed
        self.__model = self.__cls_model(*args, **kwargs)

        return self
