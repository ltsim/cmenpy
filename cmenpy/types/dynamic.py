from collections.abc import Iterable

from cmenpy.population.agent import Agent
from cmenpy.types import NDArrayType

Source = NDArrayType | Iterable[float | int]
SourceIterable = Agent | NDArrayType
DualSource = tuple[SourceIterable, SourceIterable]
