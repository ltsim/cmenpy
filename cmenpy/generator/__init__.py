"""Seeded random generator handed to models as ``ctx.rng``."""

import numpy as np


class DefaultGenerator:
    """Seeded random generator handed to models as ``ctx.rng``.

    Args:
        seed: Seed for the underlying ``numpy.random.Generator``. ``None``
            seeds from OS entropy (non-deterministic).

    Attributes:
        seed: The seed this generator was constructed with.
        rng: The live random generator used for sampling.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Create the underlying ``numpy.random.Generator`` from ``seed``."""
        self.__seed = seed
        self.__rng = np.random.default_rng(self.__seed)

    @property
    def seed(self) -> int | None:
        """The seed this generator was constructed with."""
        return self.__seed

    @property
    def rng(self) -> np.random.Generator:
        """The live random generator used for sampling."""
        return self.__rng
