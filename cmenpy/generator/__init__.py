"""Seeded random generator handed to models as ``ctx.rng``."""

import numpy as np


class DefaultGenerator:
    """Seeded random generator handed to models as ``ctx.rng``.

    Args:
        seed: Seed for the underlying ``numpy.random.Generator``. ``None``
            seeds from OS entropy (non-deterministic).

    Attributes:
        rng: The live random generator used for sampling.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Create the underlying ``numpy.random.Generator`` from ``seed``."""
        self.__rng = np.random.default_rng(seed)

    @property
    def rng(self) -> np.random.Generator:
        """The live random generator used for sampling."""
        return self.__rng
