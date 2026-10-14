import typing

import numpy as np
import numpy.typing as npt

from cmenpy.hints.option import SenseType

A = typing.TypeVar("A")
B = typing.TypeVar("B")
AB = tuple[A, B]


def _compare(ab: AB[typing.Any, typing.Any], sense: SenseType, better: bool) -> npt.NDArray[np.intp]:
    a, b = np.array(ab[0]), np.array(ab[1])

    if (sense == "min") == better:
        mask = a < b
    else:
        mask = a > b

    return np.flatnonzero(mask)


def is_best(ab: AB[typing.Any, typing.Any], sense: SenseType = "min") -> npt.NDArray[np.intp]:
    """Indices where the first element of the pair beats the second.

    Parameters
    ----------
    ab : tuple
        A ``(new, old)`` pair of fitness arrays (or scalars) to compare.
    sense : SenseType, optional
        ``"min"`` (default) or ``"max"``; selects the direction of "better".

    Returns
    -------
    numpy.ndarray
        Integer indices where ``new`` is better than ``old``.
    """
    return _compare(ab, sense, better=True)


def is_worst(ab: AB[typing.Any, typing.Any], sense: SenseType = "min") -> npt.NDArray[np.intp]:
    """Indices where the first element of the pair is worse than the second.

    Parameters
    ----------
    ab : tuple
        A ``(new, old)`` pair of fitness arrays (or scalars) to compare.
    sense : SenseType, optional
        ``"min"`` (default) or ``"max"``; selects the direction of "worse".

    Returns
    -------
    numpy.ndarray
        Integer indices where ``new`` is worse than ``old``.
    """
    return _compare(ab, sense, better=False)


def best_index(values: npt.ArrayLike, sense: SenseType = "min") -> int:
    """Index of the best value in ``values``.

    Parameters
    ----------
    values : array-like
        Fitness values to scan.
    sense : SenseType, optional
        ``"min"`` (default) or ``"max"``.

    Returns
    -------
    int
        Index of the minimum (``"min"``) or maximum (``"max"``) value.
    """
    _values = np.asarray(values)

    return int(np.argmin(_values)) if sense == "min" else int(np.argmax(_values))


def worst_index(values: npt.ArrayLike, sense: SenseType = "min") -> int:
    """Index of the worst value in ``values``.

    Parameters
    ----------
    values : array-like
        Fitness values to scan.
    sense : SenseType, optional
        ``"min"`` (default) or ``"max"``.

    Returns
    -------
    int
        Index of the maximum (``"min"``) or minimum (``"max"``) value.
    """
    _values = np.asarray(values)

    return int(np.argmax(_values)) if sense == "min" else int(np.argmin(_values))


__all__ = ["best_index", "is_best", "is_worst", "worst_index"]
