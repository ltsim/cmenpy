"""The `PopulationSwarm` handle composing all population operations."""

from cmenpy.kernel import KernelBuffer
from cmenpy.population.attributes import BufferMatrixOperations
from cmenpy.population.dynamic.base import DynamicBase
from cmenpy.population.iterator.base import BaseIterator
from cmenpy.population.operations import StreamOperations


class PopulationSwarm(
    StreamOperations, BufferMatrixOperations, BaseIterator, DynamicBase
):
    """Main handle over a `KernelBuffer`.

    A swarm layers a readable/writable matrix view (``X``/``F``), stream
    operators (``compute``/``assign``/``access``/``extract``/``swap``),
    ranking (``best``/``worst``/``sort``) and roster management
    (``insert``/``remove``/``free_space``) on top of the flat buffer.

    Args:
        buffer: The buffer backing this population.

    Examples:
        ::

            pop = cm.PopulationSwarm(ctx.buff)
            pop.compute << ctx.rng.uniform(bounds.low, bounds.up,
                                           (pop.size, bounds.ndim))
            best_rows = pop.X[pop.sort][:3]
    """

    def __init__(self, buffer: KernelBuffer) -> None:
        """Compose the matrix views, operators, iterator and roster over ``buffer``."""
        StreamOperations.__init__(self, buffer)
        BufferMatrixOperations.__init__(self, buffer)
        BaseIterator.__init__(self, buffer)
        DynamicBase.__init__(self, buffer)

        self.__buffer = buffer
