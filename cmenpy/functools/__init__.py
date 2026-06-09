import typing

from cmenpy.types.option import SenseType

A = typing.TypeVar("A")
B = typing.TypeVar("B")
AB = tuple[A, B]


def is_best(ab: AB, sense: SenseType = "min") -> bool:
    a, b = ab

    if sense == "min":
        return a < b

    return a > b


def is_worst(ab: AB, sense: SenseType = "min") -> bool:
    a, b = ab

    if sense == "min":
        return a > b

    return a < b


def best_of(a: typing.Collection[A], sense: SenseType = "min") -> A:
    if sense == "min":
        return min(a)

    return max(a)


def worst_of(a: typing.Collection[A], sense: SenseType = "min") -> A:
    if sense == "max":
        return min(a)

    return max(a)


__all__ = ["is_best", "is_worst", "best_of", "worst_of"]
