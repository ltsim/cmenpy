from cmenpy.types.dynamic import DualSource, SourceIterable
from cmenpy.types.option import SenseType


def is_best(ab: DualSource, sense: SenseType = "min") -> bool:
    a, b = ab

    if sense == "min":
        return a[0] < b[0]

    return a[0] > b[0]


def is_worst(ab: DualSource, sense: SenseType = "min") -> bool:
    a, b = ab

    if sense == "min":
        return a[0] > b[0]

    return a[0] < b[0]


def best_of(ab: DualSource, sense: SenseType = "min") -> SourceIterable:
    a, b = ab

    if is_best(ab, sense):
        return a

    return b


def worst_of(ab: DualSource, sense: SenseType = "min") -> SourceIterable:
    a, b = ab

    if is_best(ab, sense):
        return a

    return b
