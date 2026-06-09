import abc
import typing

from cmenpy.kernel import KernelBuffer
from cmenpy.population.agent import Agent, MutableAgent
from cmenpy.population.people.collection.mutable import PeopleMutableCollection
from cmenpy.population.people.collection.sequence import PeopleSequenceCollection
from cmenpy.population.people.properties import GlobalPopulationProperty
from cmenpy.types import NDArrayType
from cmenpy.types.option import SenseType


class PeopleSequence(PeopleSequenceCollection, GlobalPopulationProperty):
    def __init__(
        self,
        buffer: KernelBuffer,
        agents: typing.List[Agent],
        sense: SenseType = "min",
    ):
        super().__init__(buffer, sense)
        self.__buffer = buffer
        self.__agents = agents

    def __getitem__(self, index) -> Agent | typing.List[Agent]:
        return self.__agents[index]

    def __iter__(self) -> typing.Iterator[Agent]:
        return iter(self.__agents)

    def __len__(self) -> int:
        return len(self.__agents)


class PeopleMutable(PeopleMutableCollection, GlobalPopulationProperty):
    def __init__(
        self,
        buffer: KernelBuffer,
        sense: SenseType = "min",
        d_class: typing.Optional[typing.Type[Agent]] = None,
    ):
        if d_class is None:
            d_class = MutableAgent

        PeopleMutableCollection.__init__(self, buffer)
        GlobalPopulationProperty.__init__(self, buffer)

        self.__buffer = buffer
        self.__sense = sense
        self.__d_class = d_class

    def __getitem__(self, index: int) -> MutableAgent: ...

    def __setitem__(self, index: int, value: NDArrayType) -> None: ...

    def __delitem__(self, index: int) -> None: ...

    def __len__(self): ...

    def __iter__(self): ...


class PeopleGenerator(abc.ABC):
    @property
    @abc.abstractmethod
    def agents(self) -> PeopleMutableCollection: ...
