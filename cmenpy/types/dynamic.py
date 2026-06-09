from collections.abc import Iterable

from cmenpy.agent.base import BaseAgent
from cmenpy.types import NDArrayType

Source = NDArrayType | Iterable[float | int]
SourceIterable = BaseAgent | NDArrayType
DualSource = tuple[SourceIterable, SourceIterable]
