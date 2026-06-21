import typing

import numpy as np

from cmenpy.decoder.base import BaseDecoder
from cmenpy.hints import ArrayFloatType, ArrayIntegerType


class BinaryDecoder(BaseDecoder[ArrayIntegerType]):
    def __init__(
        self,
        n_vars: int = 1,
        name: str = "binary",
        generator: typing.Optional[np.random.Generator] = None,
        seed: typing.Optional[int] = None,
    ) -> None:
        super().__init__(name, generator=generator, seed=seed)

        if not isinstance(n_vars, int) or n_vars < 1:
            raise ValueError("n_vars must be a positive integer.")

        self.n_vars = n_vars
        self.low = np.zeros(n_vars, dtype=np.float64)
        self.up = np.full(n_vars, 2.0 - self.epsilon, dtype=np.float64)

    def generate(self) -> ArrayIntegerType:
        return self.generator.integers(0, 2, self.n_vars)

    def encode(
        self, values: typing.Union[typing.Sequence[int], ArrayIntegerType]
    ) -> ArrayFloatType:
        return np.array(values, dtype=np.float64)

    def correct(self, x: ArrayFloatType) -> ArrayIntegerType:
        return np.array(np.clip(x, self.low, self.up), dtype=np.int_)

    def decode(self, x: ArrayFloatType) -> ArrayIntegerType:
        return self.correct(x)
