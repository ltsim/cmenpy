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
        epoch_it: The epoch iterator driving the run.
        buffer: The population buffer.
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
        self.__target = TargetFunction(f, bounds)
        self.__k_buff = KernelBuffer(size, bounds, self.__target, sense)
        self.__generator = DefaultGenerator(seed=seed)
        self.__tracker = Tracker(self.__k_buff)
        self.__epoch_it = EpochIteration(epochs, self.__tracker, self.__target, debug)
        self.__bounds = bounds
        self.__sense = sense

    def context(self) -> Context:
        """Build the per-epoch `Context` view over the run's resources."""
        return Context(
            f=self.__target,
            bounds=self.__bounds,
            buffer=self.__k_buff,
            generator=self.__generator,
            sense=self.__sense,
            epoch=self.__epoch_it.current,
            epochs=self.__epoch_it.max,
        )

    @property
    def epoch_it(self) -> EpochIteration:
        """The epoch iterator driving the run."""
        return self.__epoch_it

    @property
    def buffer(self) -> KernelBuffer:
        """The population buffer."""
        return self.__k_buff

    @property
    def tracker(self) -> Tracker:
        """The per-epoch history recorder."""
        return self.__tracker
