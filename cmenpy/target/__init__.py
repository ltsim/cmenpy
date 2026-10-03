import typing

import numpy as np

from cmenpy.bounds import Bounds
from cmenpy.hints import NDArrayType, ArrayIntegerType


class MaxNumberOfCalls(Exception): ...


class Target(typing.Protocol):
    def __call__(self, x: NDArrayType) -> NDArrayType | typing.Any: ...


class VectorizedTarget:
    """A target that maps a whole ``(pop, ndim)`` matrix to a ``(pop,)`` vector.

    Marking a target as vectorized lets ``KernelBuffer.apply`` score a whole
    population with a single call instead of ``np.apply_along_axis``, which
    spends one Python call per individual.
    """

    vectorized = True

    def __init__(self, target: Target) -> None:
        self.__target = target

    def __call__(self, x: NDArrayType) -> NDArrayType | typing.Any:
        return self.__target(x)


def vectorized_target(target: Target) -> VectorizedTarget:
    """Declare ``target`` as able to evaluate a whole ``(pop, ndim)`` block.

    The wrapped callable receives a 2-D array and must return one value per
    row, so ``np.sum(x**2, axis=1)`` replaces a per-row comprehension. The
    result is identical to the row-by-row evaluation, only dispatched once.

    Examples
    --------
    ::

        def sphere(x):           # x: (pop, ndim) -> (pop,)
            return np.sum(x**2, axis=1)

        model.solve(cm.vectorized_target(sphere), bounds, 100, 40)
    """
    return VectorizedTarget(target)


class TargetFunction:
    def __init__(
        self, target: Target, bounds: Bounds, max_nfe: typing.Optional[int] = None
    ) -> None:
        def wrapper(x: NDArrayType) -> NDArrayType | typing.Any:
            self.__spend(1)

            return target(x)

        self.__target = wrapper
        self.__raw = target
        self.__vectorized = getattr(target, "vectorized", False) is True
        self.__bounds = bounds
        self.__nfe = 0
        self.__max_nfe = np.inf if max_nfe is None else max_nfe

    def __spend(self, calls: int) -> None:
        if self.__nfe + calls > self.__max_nfe:
            raise MaxNumberOfCalls("Max number of function calls exceeded.")

        self.__nfe += calls

    def __call__(self, x: NDArrayType) -> NDArrayType | typing.Any:
        return self.__target(x)

    def evaluate(self, x: NDArrayType) -> NDArrayType | typing.Any:
        return self.__target(x)

    def evaluate_matrix(self, x: NDArrayType) -> NDArrayType:
        """Score a whole ``(pop, ndim)`` block and return a ``(pop,)`` vector."""
        if not self.__vectorized:
            raise TypeError(
                "Target is not vectorized; wrap it with `vectorized_target`."
            )

        self.__spend(len(x))

        return self.__raw(x)

    def apply_axis(self, x: NDArrayType, mask: ArrayIntegerType) -> NDArrayType:
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

    @property
    def vectorized(self) -> bool:
        return self.__vectorized
