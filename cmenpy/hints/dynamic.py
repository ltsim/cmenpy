from collections.abc import Iterable

from cmenpy.population.agent.base import BaseAgent
from cmenpy.hints import NDArrayType

Source = NDArrayType | Iterable[float | int]
SourceIterable = BaseAgent | NDArrayType
DualSource = tuple[SourceIterable, SourceIterable]
