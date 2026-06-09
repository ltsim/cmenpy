import typing

from cmenpy.hints import NDArrayType

K = typing.TypeVar("K")


class Iterable:
    def __init__(self, idx: list[int], x_obj: NDArrayType):
        if len(idx) != len(x_obj):
            raise TypeError(f"Expected {len(idx)} elements, got {len(x_obj)}")

        self.__idx = idx
        self.__x_obj = x_obj

    def __iter__(self):
        return iter(zip(self.__idx, self.__x_obj[self.__idx]))
