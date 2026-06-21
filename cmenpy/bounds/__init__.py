import functools
import typing

from cmenpy.decoder import BaseDecoder
from cmenpy.hints import ScalarType, ArrayFloatType

InnerSequence = typing.Union[typing.Tuple[ScalarType, ...], typing.List[ScalarType]]


class Bounds:
    def __init__(self, bounds: ArrayFloatType):
        self.__bounds = bounds

    @functools.cached_property
    def ndim(self):
        return len(self.__bounds)

    @property
    def low(self):
        return self.__bounds[0][0]

    @property
    def up(self):
        return self.__bounds[0][1]

    def __iter__(self):
        return (b for b in self.__bounds)

    def __len__(self):
        return len(self.__bounds)

    def __class_getitem__(cls, params):
        return create_bounds(params)


def create_bounds(
    spaces: ArrayFloatType | Bounds | list[BaseDecoder] | tuple[BaseDecoder, ...],
) -> Bounds:
    def create_bound_from_space(space: list[BaseDecoder]) -> Bounds:
        pass

    if isinstance(spaces, Bounds):
        return spaces
    elif isinstance(spaces, tuple) or isinstance(spaces, list):
        return create_bound_from_space(list(spaces))

    return Bounds(spaces)
