import functools


class Bounds:
    def __init__(self, bounds):
        self.__bounds = bounds

    @functools.cached_property
    def ndim(self):
        return len(self.__bounds)

    def __iter__(self):
        return (b for b in self.__bounds)
