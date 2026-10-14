import functools
import typing

from cmenpy.hints import ScalarType

InnerSequence = tuple[ScalarType, ...] | list[ScalarType]
SequenceStructure = list[InnerSequence[ScalarType]]


class Bounds:
    """Search-space bounds: one ``(low, up)`` pair per dimension.

    Parameters
    ----------
    bounds : SequenceStructure[ScalarType]
        A sequence of ``(low, up)`` pairs, one per dimension. The number of
        pairs defines `ndim`.

    Attributes
    ----------
    ndim : int
        Number of dimensions (length of the bounds sequence).
    low : scalar
        Lower bound of the first dimension, convenient for vectorized
        sampling over a box-shaped space.
    up : scalar
        Upper bound of the first dimension, convenient for vectorized
        sampling over a box-shaped space.

    Examples
    --------
    ::

        bounds = cm.Bounds([(-1, 1), (-2, 2), (0, 5)])
        bounds.ndim      # 3
        bounds.low, bounds.up   # -1, 1
    """

    def __init__(self, bounds: SequenceStructure[ScalarType]):
        self.__bounds: SequenceStructure[typing.Any] = bounds

    @functools.cached_property
    def ndim(self) -> int:
        return len(self.__bounds)

    @property
    def low(self) -> typing.Any:
        return self.__bounds[0][0]

    @property
    def up(self) -> typing.Any:
        return self.__bounds[0][1]

    def __iter__(self) -> typing.Iterator[InnerSequence[typing.Any]]:
        return iter(self.__bounds)

    def __len__(self) -> int:
        return len(self.__bounds)

    def __class_getitem__(cls, params: typing.Any) -> "Bounds":
        return create_bounds(params)


def create_bounds(bounds: SequenceStructure[ScalarType] | Bounds) -> Bounds:
    """Coerce ``bounds`` into a `Bounds`, passing instances through.

    Parameters
    ----------
    bounds : SequenceStructure[ScalarType] | Bounds
        Either an already-built `Bounds` or a sequence of
        ``(low, up)`` pairs.

    Returns
    -------
    Bounds
        A `Bounds` instance wrapping the input.
    """
    if isinstance(bounds, Bounds):
        return bounds

    return Bounds(bounds)
