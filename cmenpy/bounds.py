import functools
import typing

from cmenpy.types import DType

InnerSequence = typing.Union[typing.Tuple[DType, ...], typing.List[DType]]
SequenceStructure = typing.List[InnerSequence[DType]]


class Bounds:
    def __init__(self, bounds: SequenceStructure[DType]):
        self.__bounds = bounds

    @functools.cached_property
    def ndim(self):
        return len(self.__bounds)

    @property
    def lb(self) -> DType:
        return self.__bounds[0][0]

    @property
    def ub(self) -> DType:
        return self.__bounds[0][1]

    def __iter__(self):
        return (b for b in self.__bounds)

    def __len__(self):
        return len(self.__bounds)

    def __class_getitem__(cls, params):
        return create_bounds(params)


def create_bounds(bounds: SequenceStructure[DType] | Bounds) -> Bounds:
    if isinstance(bounds, Bounds):
        return bounds

    return Bounds(bounds)
