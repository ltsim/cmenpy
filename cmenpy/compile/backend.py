"""Optional Numba backend: availability probe and lazy compiler accessors.

Numba is an *optional* dependency. Nothing in ``cmenpy`` imports it at module
load time; the package is located with ``importlib.util.find_spec`` and only
imported when a kernel is actually asked to compile. Kernels that are not
compiled run as pure Python/NumPy callables (see ``cmenpy.compile.low.fallback``).
"""

import importlib
import importlib.util
import typing

__all__ = ["HAS_NUMBA", "lazy", "numba_module", "prange"]

try:  # pragma: no cover - trivial availability probe
    HAS_NUMBA = importlib.util.find_spec("numba") is not None
except (ImportError, ValueError):  # pragma: no cover - broken/hidden install
    HAS_NUMBA = False

#: Message raised when compilation is requested but numba cannot be imported.
_INSTALL_HINT = (
    "cmenpy needs numba to compile kernels. Install it with "
    '`pip install "cmenpy[numba]"` (or `uv add numba`), or run without '
    "compilation by instantiating the model with `numba=False`."
)


def numba_module() -> typing.Any:
    """Return the `numba` module, importing it on first use.

    Returns:
        The imported ``numba`` module.

    Raises:
        ImportError: If numba is not installed (or not importable).
    """
    if not HAS_NUMBA:
        raise ImportError(_INSTALL_HINT)

    return importlib.import_module("numba")


def prange() -> typing.Any:
    """Return ``numba.prange``, falling back to the builtin ``range``.

    Kernels written with ``cm.prange`` stay importable (and runnable) when
    numba is absent: the pure-Python fallback executes the body as an ordinary
    loop, and when numba *is* present the compiled code gets real ``prange``
    parallelism.
    """
    try:
        return numba_module().prange
    except ImportError:
        return range


def lazy(name: str) -> typing.Any:
    """Resolve a backend name exposed lazily by ``cmenpy`` and ``cmenpy.compile``.

    ``prange`` needs a call (it imports numba when present) and ``HAS_NUMBA``
    must stay live for tests that patch it, so both are resolved on access.

    Raises:
        KeyError: If ``name`` is not a lazily exposed backend name.
    """
    if name == "prange":
        return prange()

    if name == "HAS_NUMBA":
        return HAS_NUMBA

    raise KeyError(name)
