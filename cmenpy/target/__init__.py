"""Objective wrappers: vectorized targets and the NFE-counting `TargetFunction`."""

import typing

import numpy as np

from cmenpy.bounds import Bounds
from cmenpy.casting import unwrap_array
from cmenpy.hints import NDArrayType
from cmenpy.hints.protocol import Target as Target


class MaxNumberOfCalls(Exception):
    """Raised when a target's evaluation budget (``max_nfe``) is exceeded."""


class VectorizedTarget:
    """A target that maps a whole ``(pop, ndim)`` matrix to a ``(pop,)`` vector.

    Marking a target as vectorized lets ``KernelBuffer.apply`` score a whole
    population with a single call instead of ``np.apply_along_axis``, which
    spends one Python call per individual.
    """

    vectorized = True

    def __init__(self, target: Target) -> None:
        """Wrap ``target`` so it is dispatched as a whole-block evaluator."""
        self.__target = target

    def __call__(self, x: NDArrayType) -> NDArrayType | typing.Any:
        """Score the whole block ``x`` with one call to the wrapped target."""
        return self.__target(x)


def vectorized_target(target: Target) -> VectorizedTarget:
    """Declare ``target`` as able to evaluate a whole ``(pop, ndim)`` block.

    The wrapped callable receives a 2-D array and must return one value per
    row, so ``np.sum(x**2, axis=1)`` replaces a per-row comprehension. The
    result is identical to the row-by-row evaluation, only dispatched once.

    Examples:
        ::

            def sphere(x):           # x: (pop, ndim) -> (pop,)
                return np.sum(x**2, axis=1)

            model.solve(cm.vectorized_target(sphere), bounds, 100, 40)
    """
    return VectorizedTarget(target)


class TargetFunction:
    """Wraps an objective, counting NFE and enforcing an optional budget.

    The wrapper scores solutions one individual at a time (``evaluate``) or a
    whole block at once (`evaluate_matrix`), increments the function-evaluation
    counter on every individual scored, and raises `MaxNumberOfCalls` once
    ``max_nfe`` is spent. A single individual always costs exactly one
    evaluation, whatever the shape of the objective underneath: a row written
    to a vectorized target is dispatched as a one-row block.

    Args:
        target: The raw objective callable.
        bounds: The search space.
        max_nfe: Evaluation budget; ``None`` means unlimited.

    Attributes:
        nfe: Evaluations spent so far.
        max: The budget (``inf`` when unlimited).
        bounds: The search space.
        vectorized: Whether the target scores whole blocks.
    """

    def __init__(
        self, target: Target, bounds: Bounds, max_nfe: int | None = None
    ) -> None:
        """Wrap ``target``, counting evaluations against the ``max_nfe`` budget."""
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
        """Score one individual, spending a single evaluation."""
        return self.evaluate(x)

    def evaluate(self, x: NDArrayType) -> NDArrayType | typing.Any:
        """Score a single individual, spending exactly one evaluation.

        A plain target receives the solution vector as it is; a vectorized
        target receives it as a ``(1, ndim)`` block and the single score is
        unwrapped again, so one individual never costs ``ndim``.

        Args:
            x: The solution vector, or a one-row ``(1, ndim)`` block.

        Returns:
            The fitness of that individual.
        """
        solution = unwrap_array(x)

        if self.__vectorized:
            rows = np.atleast_2d(solution)

            self.__spend(len(rows))

            scores = typing.cast(NDArrayType, self.__raw(rows))

            return scores[0] if len(rows) == 1 else scores

        self.__spend(1)

        return self.__raw(solution)

    def evaluate_matrix(self, x: NDArrayType) -> NDArrayType:
        """Score a whole ``(pop, ndim)`` block and return a ``(pop,)`` vector.

        Spending is one evaluation per row of the block, so masked and
        index-selected blocks cost only what they actually score.
        """
        if not self.__vectorized:
            raise TypeError(
                "Target is not vectorized; wrap it with `vectorized_target`."
            )

        block = unwrap_array(x)

        if np.ndim(block) != 2:
            raise TypeError(
                "`evaluate_matrix` expects a (pop, ndim) block; score a single "
                "solution with `evaluate`."
            )

        self.__spend(len(block))

        return typing.cast(NDArrayType, self.__raw(block))

    @property
    def nfe(self) -> int:
        """Evaluations spent so far."""
        return self.__nfe

    @property
    def max(self) -> float | int:
        """The budget (``inf`` when unlimited)."""
        return self.__max_nfe

    @property
    def bounds(self) -> Bounds:
        """The search space."""
        return self.__bounds

    @property
    def vectorized(self) -> bool:
        """Whether the target scores whole blocks."""
        return self.__vectorized
