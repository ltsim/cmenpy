from cmenpy.bounds import Bounds


class TargetFunction:
    def __init__(self, f, bounds: Bounds):
        self.__f = f
        self.__n_call = 0
        self.__bounds = bounds

    def __call__(self, x, *args, **kwargs):
        self.__n_call += 1

        return self.__f(x)

    def wrapper(self):
        def inner(x):
            self.__n_call += 1

            return self.__f(x)

        return inner

    @property
    def n_call(self):
        return self.__n_call

    @property
    def bounds(self):
        return self.__bounds
