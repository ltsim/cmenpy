# API Reference

Complete autodoc for the public surface of `cmenpy`. Every class, function and
type alias the framework exposes is documented here, grouped by subpackage. Each
symbol links to the page that documents it.

!!! note "How to read this reference"
    Documentation is generated from the source via
    [mkdocstrings](https://mkdocstrings.github.io/). Signatures are shown with
    their annotations, and class initializers are merged into the class entry.

## Symbol index

| Symbol | Kind | Page |
| ------ | ---- | ---- |
| `Array`, `Scalar` | compile markers | [Compile](compile.md) |
| `Contract` | compile marker | [Compile](compile.md) |
| `precompile`, `vectorize`, `reduce` | decorators | [Compile](compile.md) |
| `jit` | decorator (deprecated) | [Compile](compile.md) |
| `kernels`, `prepare` | introspection | [Compile](compile.md) |
| `declare`, `template` | model decorators | [Models](models.md) |
| `module` | Operator Module decorator | [Models](models.md) |
| `Argument`, `Constant`, `Variable` | field markers | [Models](models.md) |
| `BaseOptimizer` | abstract base | [Models](models.md) |
| `FunctionOptimizerModel`, `TemplateOptimizerModel` | models | [Models](models.md) |
| `ModelProtocol`, `AlgorithmFunction`, `CallableFunction` | protocols | [Models](models.md) |
| `FunctionParamsArguments` | arguments holder | [Models](models.md) |
| `ExceptionTypeParams`, `ExceptionValueParams` | exceptions | [Models](models.md) |
| `build_context_manager`, `declare_from_func`, `declare_from_class` | builders | [Models](models.md) |
| `PopulationSwarm` | population handle | [Population](population.md) |
| `StreamOperations` | operator set | [Population](population.md) |
| `ComputeOperator`, `AssignOperator`, `AccessOperator`, `ExtractOperation`, `SwapOperation` | operators | [Population](population.md) |
| `ComputePipeline`, `AssignPipeline` | per-row targets | [Population](population.md) |
| `StreamOperator`, `ReaderOperator`, `WriterOperator`, `VectorizableOperator`, `PipelineOperator` | operator ABCs | [Population](population.md) |
| `BufferMatrixOperations`, `BufferMatrixFitness`, `BufferMatrixSolutions` | matrix views | [Population](population.md) |
| `BufferMatrixAbstract`, `BufferMatrixAccess`, `BufferMatrixCompute` | matrix ABCs | [Population](population.md) |
| `DynamicBase`, `BaseIterator` | population bases | [Population](population.md) |
| `BaseAgent`, `ImmutableAgent`, `MutableAgent`, `AgentTemplate` | agents | [Population](population.md) |
| `KernelBuffer` | buffer | [Kernel](kernel.md) |
| `KernelMask`, `KernelSize` | buffer support | [Kernel](kernel.md) |
| `BufferIndex` | ranking | [Kernel](kernel.md) |
| `init_buffer` | allocator | [Kernel](kernel.md) |
| `Target`, `TargetFunction` | objective | [Target](target.md) |
| `VectorizedTarget`, `vectorized_target` | vectorized objective | [Target](target.md) |
| `MaxNumberOfCalls` | exception | [Target](target.md) |
| `Bounds`, `create_bounds` | search space | [Bounds](bounds.md) |
| `InnerSequence`, `SequenceStructure` | bounds aliases | [Bounds](bounds.md) |
| `Context`, `MainContextManager` | run context | [Runtime](runtime.md) |
| `EpochIteration` | epoch iterator | [Runtime](runtime.md) |
| `Tracker`, `EpochHistory` | history | [Runtime](runtime.md) |
| `DefaultGenerator` | RNG | [Runtime](runtime.md) |
| `is_best`, `is_worst` | comparators | [Utilities](utilities.md) |
| `best_index`, `worst_index` | ranking helpers | [Utilities](utilities.md) |
| `NDArrayType`, `ArrayFloatType`, `ArrayIntegerType` | array aliases | [Utilities](utilities.md) |
| `ScalarType`, `SenseType` | scalar/option aliases | [Utilities](utilities.md) |

## Browse by category

- **[Compile](compile.md)** — numba kernels, markers and introspection.
- **[Models](models.md)** — declaration decorators, field markers, optimizers.
- **[Population](population.md)** — the swarm handle, operators, matrix views, agents.
- **[Kernel](kernel.md)** — the flat buffer, masks, sizes and ranking.
- **[Target](target.md)** — objectives and vectorized targets.
- **[Bounds](bounds.md)** — the search-space description.
- **[Runtime](runtime.md)** — context, epoch iteration, history, RNG.
- **[Utilities](utilities.md)** — comparators and type aliases.
