import typing

import numpy as np

from cmenpy.decoder.base import BaseDecoder
from cmenpy.hints import ArrayFloatType
from cmenpy.hints.array import ArrayBoolType


class BoolDecoder(BaseDecoder[ArrayBoolType]):
    def __init__(self, n_vars: int = 1, name: str = "bool") -> None:
        super().__init__(name)
        if not isinstance(n_vars, int) or n_vars < 1:
            raise ValueError("n_vars must be a positive integer.")
        self.n_vars = n_vars
        self.lb = np.zeros(n_vars, dtype=np.float64)
        self.ub = np.full(n_vars, 2.0 - self.epsilon, dtype=np.float64)

    def generate(self) -> ArrayBoolType:
        return self.generator.integers(0, 2, self.n_vars).astype(np.bool_)

    def encode(
        self, values: typing.Union[typing.Sequence[bool], ArrayBoolType]
    ) -> ArrayFloatType:
        return np.array(values, dtype=np.float64)

    def correct(self, x: ArrayFloatType) -> ArrayBoolType:
        clipped: ArrayFloatType = np.clip(x, self.lb, self.ub)
        return np.array(clipped, dtype=np.int_).astype(np.bool_)

    def decode(self, x: ArrayFloatType) -> ArrayBoolType:
        return self.correct(x)
