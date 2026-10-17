"""Per-epoch run context (`Context`) and its owner (`MainContextManager`)."""

import numpy as np

from cmenpy.bounds import Bounds
from cmenpy.epoch import EpochIteration
from cmenpy.generator import DefaultGenerator
from cmenpy.hints.option import SenseType
from cmenpy.kernel import KernelBuffer, KernelSize
from cmenpy.target import Target, TargetFunction
from cmenpy.tracker import Tracker


class Context:
    """Read-only view of the run handed to ``initialize``/``evolve``.

    A fresh ``Context`` is built for every epoch. It exposes the search
    space, the population buffer, the random generator, the optimization
    sense, and the current/total epoch counters.

    Attributes:
        target: The wrapped objective (counts NFE, enforces any budget).
        bounds: The search space (``low``, ``up``, ``ndim``).
        buff: The buffer backing the population.
        rng: Seeded random generator.
        sense: ``"min"`` or ``"max"``.
        epoch: Current epoch index.
        epochs: Total number of epochs.
    """

    def __init__(
        self,
        f: Target,
        bounds: Bounds,
        buffer: KernelBuffer,
        generator: DefaultGenerator,
        sense: SenseType,
        epoch: int = 0,
        epochs: int = 0,
    ):
        """Wrap the run resources into a per-epoch view."""
        self.__target = TargetFunction(f, bounds)
        self.__bounds = bounds
        self.__buff = buffer
        self.__generator = generator
        self.__sense = sense
        self.__epoch = epoch
        self.__epochs = epochs

    @property
    def target(self) -> TargetFunction:
        """The wrapped objective (counts NFE, enforces any budget)."""
        return self.__target

    @property
    def bounds(self) -> Bounds:
        """The search space (``low``, ``up``, ``ndim``)."""
        return self.__bounds

    @property
    def buff(self) -> KernelBuffer:
        """The buffer backing the population."""
        return self.__buff

    @property
    def rng(self) -> np.random.Generator:
        """Seeded random generator."""
        return self.__generator.rng

    @property
    def sense(self) -> SenseType:
        """``"min"`` or ``"max"``."""
        return self.__sense

    @property
    def epoch(self) -> int:
        """Current epoch index."""
        return self.__epoch

    @property
    def epochs(self) -> int:
        """Total number of epochs."""
        return self.__epochs


class MainContextManager:
    """Owns every per-run resource and builds the per-epoch ``Context``.

    This is the internal orchestrator created by ``solve``. It wires the
    objective, bounds, buffer, generator, tracker and epoch iterator
    together and hands out `Context` views.

    Args:
        f: The objective callable.
        bounds: The search space.
        epochs: Total number of epochs.
        size: Population size (current, min, max).
        seed: Random seed forwarded to the generator.
        debug: Record per-epoch history. Default ``False``.
        sense: ``"min"`` (default) or ``"max"``.

    Attributes:
        sense: The optimization sense.
        epoch_it: The epoch iterator driving the run.
        buffer: The population buffer.
        target: The wrapped objective.
        bounds: The search space.
        default_generator: The seeded random generator.
        tracker: The per-epoch history recorder.
    """

    def __init__(
        self,
        f: Target,
        bounds: Bounds,
        epochs: int,
        size: KernelSize,
        seed: int | None = None,
        debug: bool = False,
        sense: SenseType = "min",
    ):
        """Wire the objective, buffer, generator, tracker and epoch iterator."""
        self.__size = size
        self.__target = TargetFunction(f, bounds)
        self.__bounds = bounds
        self.__sense = sense
        self.__k_buff = KernelBuffer(size, self.__bounds, self.__target, self.__sense)
        self.__generator = DefaultGenerator(seed=seed)
        self.__tracker = Tracker(self.__k_buff)
        self.__epoch_it = EpochIteration(epochs, self.__tracker, self.__target, debug)

    @property
    def sense(self) -> SenseType:
        """The optimization sense."""
        return self.__sense

    @property
    def epoch_it(self) -> EpochIteration:
        """The epoch iterator driving the run."""
        return self.__epoch_it

    @property
    def buffer(self) -> KernelBuffer:
        """The population buffer."""
        return self.__k_buff

    @property
    def target(self) -> TargetFunction:
        """The wrapped objective."""
        return self.__target

    @property
    def bounds(self) -> Bounds:
        """The search space."""
        return self.__bounds

    @property
    def default_generator(self) -> DefaultGenerator:
        """The seeded random generator."""
        return self.__generator

    @property
    def tracker(self) -> Tracker:
        """The per-epoch history recorder."""
        return self.__tracker
