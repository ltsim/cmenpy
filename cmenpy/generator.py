import typing

import numpy as np


class DefaultGenerator:
    def __init__(self, seed: typing.Optional[int] = None) -> None:
        self.__seed = seed
        self.__rng = np.random.default_rng(self.__seed)

    @property
    def seed(self) -> typing.Optional[int]:
        return self.__seed

    @property
    def rng(self) -> np.random.Generator:
        return self.__rng
