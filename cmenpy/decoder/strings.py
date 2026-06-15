import typing

import numpy as np

from cmenpy.decoder.base import (
    BaseDecoder,
    as_label_sets,
)
from cmenpy.hints import ArrayFloatType, ArrayIntegerType
from cmenpy.hints.array import ArrayObjectType


class StringDecoder(BaseDecoder[typing.List[typing.Any]]):
    labels: typing.Tuple[ArrayObjectType, ...]
    index: typing.Tuple[typing.Dict[typing.Any, int], ...]

    def __init__(
        self, valid_sets: typing.Sequence[typing.Any], name: str = "string"
    ) -> None:
        super().__init__(name)
        self.labels, self.index = as_label_sets(valid_sets)
        self.n_vars = len(self.labels)
        self.lb = np.zeros(self.n_vars, dtype=np.float64)
        self.ub = np.array(
            [len(arr) - self.epsilon for arr in self.labels], dtype=np.float64
        )

    def generate(self) -> typing.List[typing.Any]:
        return [self.generator.choice(arr) for arr in self.labels]

    def encode(self, values: typing.Sequence[typing.Any]) -> ArrayFloatType:
        return np.array(
            [table[val] for table, val in zip(self.index, values)], dtype=np.float64
        )

    def correct(self, x: ArrayFloatType) -> ArrayIntegerType:
        return np.array(np.clip(x, self.lb, self.ub), dtype=np.int_)

    def decode(self, x: ArrayFloatType) -> typing.List[typing.Any]:
        idx: ArrayIntegerType = self.correct(x)
        return [arr[i] for arr, i in zip(self.labels, idx)]
