import typing

import numba as nb

P = typing.ParamSpec("P")
R = typing.TypeVar("R")


def vectorize(*args: typing.Any, **kwargs: typing.Any) -> typing.Callable[[typing.Callable[P, R]], staticmethod]:
    def wrapper(f: typing.Callable[P, R]) -> staticmethod:
        f = nb.vectorize(['float64(float64, float64)'])(f)
        return staticmethod(f)

    return wrapper


def parallel(*args, **kwargs) -> typing.Callable[[typing.Callable[P, R]], staticmethod]:
    def inner(f: typing.Callable[P, R]) -> staticmethod:
        f = nb.vectorize(['float64(float64, float64)'])(f)
        return staticmethod(f)

    return inner


def jit(*args: typing.Any, **kwargs: typing.Any) -> typing.Callable[[typing.Callable[P, R]], staticmethod]:
    def inner(f: typing.Callable[P, R]) -> typing.Any:
        f = nb.njit(f)
        return staticmethod(f)

    return inner
