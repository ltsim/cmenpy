from collections.abc import Iterable

from cmenpy.population.agent.base import BaseAgent
from cmenpy.hints import ArrayType

Source = ArrayType | Iterable[float | int]
SourceIterable = BaseAgent | ArrayType
DualSource = tuple[SourceIterable, SourceIterable]
