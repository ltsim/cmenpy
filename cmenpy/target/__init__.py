import typing

import numpy as np

from cmenpy.bounds import Bounds
from cmenpy.hints import ArrayType, ArrayIntegerType


class MaxNumberOfCalls(Exception): ...


class Target(typing.Protocol):
    def __call__(self, x: ArrayType) -> ArrayType | typing.Any: ...


class TargetFunction:
    def __init__(
        self, target: Target, bounds: Bounds, max_nfe: typing.Optional[int] = None
    ) -> None:
        def wrapper(x: ArrayType) -> ArrayType | typing.Any:
            if not self.__max_nfe > self.__nfe:
                raise MaxNumberOfCalls("Max number of function calls exceeded.")

            self.__nfe += 1

            return target(x)

        self.__target = wrapper
        self.__bounds = bounds
        self.__nfe = 0
        self.__max_nfe = np.inf if max_nfe is None else max_nfe

    def __call__(self, x: ArrayType) -> ArrayType | typing.Any:
        return self.__target(x)

    def evaluate(self, x: ArrayType) -> ArrayType | typing.Any:
        return self.__target(x)

    def apply_axis(self, x: ArrayType, mask: ArrayIntegerType) -> ArrayType:
        return np.apply_along_axis(self.__target, 1, x[mask, 1:])

    @property
    def nfe(self) -> int:
        return self.__nfe

    @property
    def max(self) -> float | int:
        return self.__max_nfe

    @property
    def bounds(self) -> Bounds:
        return self.__bounds

    @property
    def f(self) -> Target:
        return self.evaluate
