"""Agent types: the abstract `BaseAgent` and its buffer-bound variants."""

from cmenpy.population.agent.base import BaseAgent
from cmenpy.population.agent.inmutable import ImmutableAgent
from cmenpy.population.agent.mutable import MutableAgent

__all__ = [
    "BaseAgent",
    "ImmutableAgent",
    "MutableAgent",
]
