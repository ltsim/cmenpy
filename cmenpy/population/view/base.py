import abc


class ViewBase(abc.ABC):
    @abc.abstractmethod
    def __repr__(self): ...

    @property
    @abc.abstractmethod
    def size(self) -> int: ...
