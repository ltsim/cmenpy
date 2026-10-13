import abc


class StreamOperator(abc.ABC):
    def __init__(self, shape: tuple[int, ...]):
        self.__shape = shape

    @property
    def shape(self) -> tuple[int, ...]:
        return self.__shape

    @property
    def rows(self) -> int:
        return self.__shape[0]

    @property
    def columns(self) -> int:
        return self.__shape[1]

    @abc.abstractmethod
    def __repr__(self) -> str: ...
