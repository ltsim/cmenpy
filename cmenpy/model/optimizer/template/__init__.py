import typing

from cmenpy.bounds import Bounds, SequenceStructure
from cmenpy.context import MainContextManager, Context
from cmenpy.kernel import KernelSize
from cmenpy.model.optimizer.base import BaseOptimizer
from cmenpy.model.optimizer.functions import CallableFunction
from cmenpy.model.optimizer.template.protocols import ModelProtocol
from cmenpy.target import Target
from cmenpy.tracker import EpochHistory
from cmenpy.hints import DType, NDArrayType
from cmenpy.hints.option import SenseType


class TemplateOptimizerModel(BaseOptimizer):
    def __init__(self, cls_model: typing.Type[ModelProtocol]) -> None:
        self.__alias = str(cls_model.__name__)
        self.__cls_model: typing.Type[ModelProtocol] = cls_model
        self.__inner: typing.Optional[CallableFunction] = None
        self.__resource: typing.Optional[MainContextManager] = None
        self.__seed: typing.Optional[int] = None
        self.__debug = False
        self.__sense: SenseType = "min"

    @property
    def alias(self) -> str:
        return self.__alias

    @property
    def tracker(self) -> list[EpochHistory]:
        if self.__resource is None:
            return []

        return self.__resource.tracker.history

    def solve(
        self,
        f: Target,
        bounds: Bounds | SequenceStructure[DType],
        epochs: int,
        pop_size: int,
        pop_range: typing.Optional[tuple[int, int]] = None,
        debug: bool = False,
        sense: SenseType = "min",
    ) -> tuple[float, NDArrayType]:
        if self.__model is None:
            raise NotImplementedError()

        self.__debug = debug

        if pop_range is None:
            pop_range = pop_size, pop_size

        min_pop, max_pop = pop_range

        if min_pop > max_pop or pop_size < min_pop:
            raise IndexError("Population size is too small.")
        elif pop_size > max_pop:
            raise IndexError("Population size is too large.")

        self.__sense = sense
        self.__size = KernelSize(pop_size, min_size=min_pop, max_size=max_pop)
        self.__bounds = Bounds[bounds]
        self.__resource = MainContextManager(
            f,
            self.__bounds,
            epochs,
            self.__size,
            self.__seed,
            self.__debug,
            self.__sense,
        )

        if self.__resource is None:
            raise NotImplementedError("Function not implemented.")

        if hasattr(self.__model, "initialize"):
            self.__model.initialize(
                ctx=Context(
                    f=self.__resource.target,
                    bounds=self.__resource.bounds,
                    buffer=self.__resource.buffer,
                    generator=self.__resource.default_generator,
                    sense=self.__sense,
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
                ),
            )

        X = self.__resource.buffer.raw_data[self.__resource.buffer.idx.best]

        return float(X[0]), X[1:]

    def __call__(
        self, seed: typing.Optional[int] = None, *args: typing.Any, **kwargs: typing.Any
    ) -> "BaseOptimizer":
        self.__seed = seed
        self.__model = self.__cls_model(*args, **kwargs)

        return self
