from collections.abc import Iterable
from cmenpy.agent import Agent
from cmenpy.types import NDArrayType

SourceIterable = Agent | NDArrayType | Iterable[float | int]
DualSource = tuple[SourceIterable, SourceIterable]
