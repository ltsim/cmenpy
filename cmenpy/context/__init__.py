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

    Attributes
    ----------
    target : TargetFunction
        The wrapped objective (counts NFE, enforces any budget).
    bounds : Bounds
        The search space (``low``, ``up``, ``ndim``).
    buff : KernelBuffer
        The buffer backing the population.
    rng : numpy.random.Generator
        Seeded random generator.
    sense : SenseType
        ``"min"`` or ``"max"``.
    epoch : int
        Current epoch index.
    epochs : int
        Total number of epochs.
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
        self.__target = TargetFunction(f, bounds)
        self.__bounds = bounds
        self.__buff = buffer
        self.__generator = generator
        self.__sense = sense
        self.__epoch = epoch
        self.__epochs = epochs

    @property
    def target(self) -> TargetFunction:
        return self.__target

    @property
    def bounds(self) -> Bounds:
        return self.__bounds

    @property
    def buff(self) -> KernelBuffer:
        return self.__buff

    @property
    def rng(self) -> np.random.Generator:
        return self.__generator.rng

    @property
    def sense(self) -> SenseType:
        return self.__sense

    @property
    def epoch(self) -> int:
        return self.__epoch

    @property
    def epochs(self) -> int:
        return self.__epochs


class MainContextManager:
    """Owns every per-run resource and builds the per-epoch ``Context``.

    This is the internal orchestrator created by ``solve``. It wires the
    objective, bounds, buffer, generator, tracker and epoch iterator
    together and hands out `Context` views.

    Parameters
    ----------
    f : Target
        The objective callable.
    bounds : Bounds
        The search space.
    epochs : int
        Total number of epochs.
    size : KernelSize
        Population size (current, min, max).
    seed : int, optional
        Random seed forwarded to the generator.
    debug : bool, optional
        Record per-epoch history. Default ``False``.
    sense : SenseType, optional
        ``"min"`` (default) or ``"max"``.

    Attributes
    ----------
    sense : SenseType
        The optimization sense.
    epoch_it : EpochIteration
        The epoch iterator driving the run.
    buffer : KernelBuffer
        The population buffer.
    target : TargetFunction
        The wrapped objective.
    bounds : Bounds
        The search space.
    default_generator : DefaultGenerator
        The seeded random generator.
    tracker : Tracker
        The per-epoch history recorder.
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
        return self.__sense

    @property
    def epoch_it(self) -> EpochIteration:
        return self.__epoch_it

    @property
    def buffer(self) -> KernelBuffer:
        return self.__k_buff

    @property
    def target(self) -> TargetFunction:
        return self.__target

    @property
    def bounds(self) -> Bounds:
        return self.__bounds

    @property
    def default_generator(self) -> DefaultGenerator:
        return self.__generator

    @property
    def tracker(self) -> Tracker:
        return self.__tracker
