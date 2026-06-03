import abc


class StreamOperator(abc.ABC): ...


class ReaderOperator(StreamOperator):
    @abc.abstractmethod
    def __getitem__(self, item): ...


class WriterOperator(StreamOperator):
    @abc.abstractmethod
    def __setitem__(self, item, value): ...


class AllOperator(StreamOperator):
    @abc.abstractmethod
    def __getitem__(self, item): ...

    @abc.abstractmethod
    def __setitem__(self, item, value): ...
