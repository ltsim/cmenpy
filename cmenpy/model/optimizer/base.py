import abc
import typing

from cmenpy.bounds import Bounds, SequenceStructure, create_bounds
from cmenpy.context import MainContextManager
from cmenpy.hints import NDArrayType, ScalarType
from cmenpy.hints.option import SenseType
from cmenpy.kernel import KernelSize
from cmenpy.target import Target
from cmenpy.tracker import EpochHistory


class BaseOptimizer(abc.ABC):
    """Abstract base every optimizer model satisfies.

    Defines the `solve` contract, the ``__call__`` configuration
    hook, the `alias` name, and the `tracker` history.
    """

    @abc.abstractmethod
    def solve(
        self,
        f: Target,
        bounds: Bounds | SequenceStructure[ScalarType],
        epochs: int,
        pop_size: int,
        pop_range: tuple[int, int] | None = None,
        debug: bool = False,
        sense: SenseType = "min",
    ) -> tuple[float, NDArrayType]: ...

    @abc.abstractmethod
    def __call__(self, *args: typing.Any, **kwargs: typing.Any) -> "BaseOptimizer": ...

    @property
    @abc.abstractmethod
    def alias(self) -> str: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alias={self.alias})"

    @property
    @abc.abstractmethod
    def tracker(self) -> list[EpochHistory]: ...


def build_context_manager(
    f: Target,
    bounds: Bounds | SequenceStructure[ScalarType],
    epochs: int,
    pop_size: int,
    pop_range: tuple[int, int] | None,
    seed: int | None,
    debug: bool,
    sense: SenseType,
) -> MainContextManager:
    """Validate the population size and build a `MainContextManager`.

    Parameters
    ----------
    f : Target
        The objective callable.
    bounds : Bounds | SequenceStructure[ScalarType]
        The search space (or bounds sequence).
    epochs : int
        Total number of epochs.
    pop_size : int
        Initial population size.
    pop_range : tuple[int, int], optional
        ``(min, max)`` allowed population sizes.
    seed : int, optional
        Random seed.
    debug : bool, optional
        Record per-epoch history.
    sense : SenseType, optional
        ``"min"`` (default) or ``"max"``.

    Returns
    -------
    MainContextManager
        The wired per-run resource holder.

    Raises
    ------
    IndexError
        If ``pop_size`` falls outside ``pop_range``.
    """
    if pop_range is None:
        pop_range = pop_size, pop_size

    min_pop, max_pop = pop_range

    if min_pop > max_pop or pop_size < min_pop:
        raise IndexError("Population size is too small.")
    elif pop_size > max_pop:
        raise IndexError("Population size is too large.")

    size = KernelSize(pop_size, min_size=min_pop, max_size=max_pop)

    return MainContextManager(
        f, create_bounds(bounds), epochs, size, seed, debug, sense
    )
