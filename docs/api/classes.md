# Classes

Every class the framework exposes, grouped by the layer it belongs to.
Decorators and free functions live in [Functions](functions.md); the packages
themselves are described in [Modules](modules.md).

## Optimizers

::: cmenpy.model.optimizer.base.BaseOptimizer
    options:
      heading_level: 3

::: cmenpy.model.optimizer.FunctionOptimizerModel
    options:
      heading_level: 3

::: cmenpy.model.optimizer.TemplateOptimizerModel
    options:
      heading_level: 3

## Field markers

::: cmenpy.Argument
    options:
      heading_level: 3

::: cmenpy.Constant
    options:
      heading_level: 3

::: cmenpy.Variable
    options:
      heading_level: 3

## Population

The swarm handle and everything layered on top of the kernel buffer: stream
operators, matrix views and the roster/iteration bases.

::: cmenpy.PopulationSwarm
    options:
      heading_level: 3

::: cmenpy.population.operations.StreamOperations
    options:
      heading_level: 3

::: cmenpy.population.operations.ComputeOperator
    options:
      heading_level: 3

::: cmenpy.population.operations.AssignOperator
    options:
      heading_level: 3

::: cmenpy.population.operations.AccessOperator
    options:
      heading_level: 3

::: cmenpy.population.operations.ExtractOperation
    options:
      heading_level: 3

::: cmenpy.population.operations.SwapOperation
    options:
      heading_level: 3

::: cmenpy.population.operations.compute.ComputePipeline
    options:
      heading_level: 3

::: cmenpy.population.operations.assign.AssignPipeline
    options:
      heading_level: 3

::: cmenpy.population.operations.stream.base.StreamOperator
    options:
      heading_level: 3

::: cmenpy.population.operations.stream.reader.ReaderOperator
    options:
      heading_level: 3

::: cmenpy.population.operations.stream.writer.WriterOperator
    options:
      heading_level: 3

::: cmenpy.population.operations.stream.vectorizable.VectorizableOperator
    options:
      heading_level: 3

::: cmenpy.population.operations.stream.vectorizable.PipelineOperator
    options:
      heading_level: 3

::: cmenpy.population.attributes.base.BufferMatrixOperations
    options:
      heading_level: 3

::: cmenpy.population.attributes.base.BufferMatrixFitness
    options:
      heading_level: 3

::: cmenpy.population.attributes.base.BufferMatrixSolutions
    options:
      heading_level: 3

::: cmenpy.population.attributes.matrix.BufferMatrixAbstract
    options:
      heading_level: 3

::: cmenpy.population.attributes.matrix.mixin.BufferMatrixAccess
    options:
      heading_level: 3

::: cmenpy.population.attributes.matrix.mixin.BufferMatrixCompute
    options:
      heading_level: 3

::: cmenpy.population.dynamic.base.DynamicBase
    options:
      heading_level: 3

::: cmenpy.population.iterator.base.BaseIterator
    options:
      heading_level: 3

## Buffer

The flat buffer that backs every population, plus the mask, size and ranking
support types.

::: cmenpy.KernelBuffer
    options:
      heading_level: 3

::: cmenpy.KernelSize
    options:
      heading_level: 3

::: cmenpy.kernel.KernelMask
    options:
      heading_level: 3

::: cmenpy.kernel.index.BufferIndex
    options:
      heading_level: 3

## Objectives

::: cmenpy.target.Target
    options:
      heading_level: 3

::: cmenpy.target.TargetFunction
    options:
      heading_level: 3

::: cmenpy.VectorizedTarget
    options:
      heading_level: 3

::: cmenpy.target.MaxNumberOfCalls
    options:
      heading_level: 3

## Search space

::: cmenpy.Bounds
    options:
      heading_level: 3

## Runtime

The per-run machinery: the context handed to models, the epoch iterator, the
history tracker and the seeded generator.

::: cmenpy.context.Context
    options:
      heading_level: 3

::: cmenpy.context.MainContextManager
    options:
      heading_level: 3

::: cmenpy.EpochIteration
    options:
      heading_level: 3

::: cmenpy.tracker.Tracker
    options:
      heading_level: 3

::: cmenpy.tracker.EpochHistory
    options:
      heading_level: 3

::: cmenpy.generator.DefaultGenerator
    options:
      heading_level: 3

## Protocols

The structural types models and kernels are checked against.

::: cmenpy.model.optimizer.ModelProtocol
    options:
      heading_level: 3

::: cmenpy.model.optimizer.functions.AlgorithmFunction
    options:
      heading_level: 3

::: cmenpy.model.optimizer.functions.CallableFunction
    options:
      heading_level: 3

## Support types and aliases

Exception/argument holders raised or built by the declaration machinery, and
the shared typing aliases.

::: cmenpy.model.optimizer.functions.FunctionParamsArguments
    options:
      heading_level: 3

::: cmenpy.model.optimizer.functions.ExceptionTypeParams
    options:
      heading_level: 3

::: cmenpy.model.optimizer.functions.ExceptionValueParams
    options:
      heading_level: 3

::: cmenpy.hints.array.NDArrayType
    options:
      heading_level: 3

::: cmenpy.hints.array.ArrayFloatType
    options:
      heading_level: 3

::: cmenpy.hints.array.ArrayIntegerType
    options:
      heading_level: 3

::: cmenpy.hints.primitives.ScalarType
    options:
      heading_level: 3

::: cmenpy.hints.option.SenseType
    options:
      heading_level: 3
