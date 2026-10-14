import numpy as np


class DefaultGenerator:
    """Seeded random generator handed to models as ``ctx.rng``.

    Parameters
    ----------
    seed : int, optional
        Seed for the underlying ``numpy.random.Generator``. ``None`` seeds
        from OS entropy (non-deterministic).

    Attributes
    ----------
    seed : int or None
        The seed this generator was constructed with.
    rng : numpy.random.Generator
        The live random generator used for sampling.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.__seed = seed
        self.__rng = np.random.default_rng(self.__seed)

    @property
    def seed(self) -> int | None:
        return self.__seed

    @property
    def rng(self) -> np.random.Generator:
        return self.__rng
