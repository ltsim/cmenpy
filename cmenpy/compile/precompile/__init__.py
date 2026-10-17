"""``numba.njit`` kernel decorators (`precompile` and the deprecated `jit`)."""

from cmenpy.compile.precompile.jit import jit
from cmenpy.compile.precompile.precompile import precompile

__all__ = ["jit", "precompile"]
