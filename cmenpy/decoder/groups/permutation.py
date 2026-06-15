import typing

import numpy as np

from cmenpy.decoder.base import BaseDecoder, as_label_set
from cmenpy.hints import ArrayFloatType, ArrayIntegerType
from cmenpy.hints.array import ArrayObjectType


class PermutationDecoder(BaseDecoder[ArrayObjectType]):
    labels: ArrayObjectType
    index: typing.Dict[typing.Any, int]

    def __init__(
        self, valid_set: typing.Sequence[typing.Any], name: str = "permutation"
    ) -> None:
        super().__init__(name)
        if len(valid_set) < 2:
            raise ValueError("Permutation needs at least two elements.")
        self.labels, self.index = as_label_set(valid_set)
        self.n_vars = len(self.labels)
        self.lb = np.zeros(self.n_vars, dtype=np.float64)
        self.ub = np.full(self.n_vars, self.n_vars - self.epsilon, dtype=np.float64)

    def generate(self) -> ArrayObjectType:
        return self.generator.permutation(self.labels)

    def encode(self, values: typing.Sequence[typing.Any]) -> ArrayFloatType:
        return np.array([self.index[val] for val in values], dtype=np.float64)

    def correct(self, x: ArrayFloatType) -> ArrayIntegerType:
        return np.argsort(x).astype(np.int_)

    def decode(self, x: ArrayFloatType) -> ArrayObjectType:
        order: ArrayIntegerType = self.correct(x)
        return self.labels[order]
