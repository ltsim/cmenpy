import functools
import typing

from cmenpy.hints import ScalarType

InnerSequence = tuple[ScalarType, ...] | list[ScalarType]
SequenceStructure = list[InnerSequence[ScalarType]]


class Bounds:
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
    if isinstance(bounds, Bounds):
        return bounds

    return Bounds(bounds)
