import numpy as np


class DefaultGenerator:
    def __init__(self, seed: int | None = None) -> None:
        self.__seed = seed
        self.__rng = np.random.default_rng(self.__seed)

    @property
    def seed(self) -> int | None:
        return self.__seed

    @property
    def rng(self) -> np.random.Generator:
        return self.__rng
