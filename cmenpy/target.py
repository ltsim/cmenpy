import typing

import numpy as np
import numpy.typing as npt

from cmenpy.bounds import Bounds

ScalarType = typing.TypeVar("ScalarType", bound=np.number)
Target = typing.Callable[[npt.NDArray[ScalarType]], ScalarType | typing.Any]


class TargetFunction:
    def __init__(self, target: Target, bounds: Bounds):
        self.__target = target
        self.__bounds = bounds
        self.__nfe = 0

    def __call__(self, x):
        self.__nfe += 1

        return self.__target(x)

    @property
    def nfe(self):
        return self.__nfe

    @property
    def bounds(self):
        return self.__bounds
