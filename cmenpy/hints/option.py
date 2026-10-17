"""Literal option types (e.g. the optimization sense)."""

import typing

#: Optimization direction: ``"min"`` or ``"max"``.
SenseType = typing.Literal["min", "max"]
