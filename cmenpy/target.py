import typing

import numpy as np
import numpy.typing as npt

from cmenpy.bounds import Bounds

ScalarType = typing.TypeVar("ScalarType", bound=np.number)


class MaxNumberOfCalls(Exception): ...


class Target(typing.Protocol):
    def __call__(self, x: npt.NDArray[ScalarType]) -> ScalarType | typing.Any: ...


class TargetFunction:
    def __init__(
        self, target: Target, bounds: Bounds, max_nfe: typing.Optional[int] = None
    ) -> None:
        self.__target = target
        self.__bounds = bounds
        self.__nfe = 0
        self.__max_nfe = np.inf if max_nfe is None else max_nfe

    def __call__(self, x: npt.NDArray[ScalarType]) -> ScalarType | typing.Any:
        if not self.__max_nfe > self.__nfe:
            raise MaxNumberOfCalls("Max number of function calls exceeded.")

        self.__nfe += 1

        return self.__target(x)

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
        return lambda x: self.__call__(x)
