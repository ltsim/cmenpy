import abc


class StreamOperator(abc.ABC):
    def __init__(self, shape: tuple[int, int]):
        self.__shape = shape

    @property
    def shape(self) -> tuple[int, int]:
        return self.__shape

    @property
    def rows(self):
        return self.__shape[0]

    @property
    def columns(self):
        return self.__shape[1]

    @abc.abstractmethod
    def __repr__(self) -> str: ...
