"""Benchmark element-wise op `C = exp(tan(A + B))` across backends.

Compares:
    - NumPy (vectorized, multiple memory passes)
    - Numba (parallel JIT with prange + fastmath)
    - PyOpenCL on CPU device (single fused kernel)
    - PyOpenCL on GPU device (single fused kernel)

For OpenCL backends, kernel-only timing is measured separately from
kernel + device-to-host copy timing to isolate compute cost from
transfer cost.
"""

import time

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array

from numba import njit, prange


MATRIX_SIDE: int = 20000
RNG_SEED: int = 42
INPUT_SCALE: float = 0.5
N_RUNS: int = 5


KERNEL_SRC = """
__kernel void elementwise(
    __global const float *A,
    __global const float *B,
    __global float *C,
    const int total)
{
    int gid = get_global_id(0);
    if (gid >= total) return;
    C[gid] = exp(tan(A[gid] + B[gid]));
}
"""


@dataclass(slots=True)
class BenchResult:
    """Holds the average and minimum runtime for a benchmarked function."""

    name: str
    avg_seconds: float
    min_seconds: float

    @property
    def avg_ms(self) -> float:
        return self.avg_seconds * 1000

    @property
    def min_ms(self) -> float:
        return self.min_seconds * 1000


def make_inputs(side: int, seed: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Build two `side x side` float32 matrices in `[-scale/2, scale/2]`.

    The bounded range keeps `exp(tan(x))` finite for all entries.
    """
    rng = np.random.default_rng(seed)
    a = (rng.random((side, side), dtype=np.float32) - 0.5) * scale
    b = (rng.random((side, side), dtype=np.float32) - 0.5) * scale
    return a, b


def benchmark(
    func: Callable[..., object],
    name: str,
    *args: object,
    n_runs: int = N_RUNS,
    warmup: bool = True,
) -> tuple[object, BenchResult]:
    """Run `func(*args)` repeatedly and return its last result with timings.

    A single warmup call is issued before timing to amortize JIT
    compilation, kernel build and cache effects.
    """
    if warmup:
        func(*args)
    times: list[float] = []
    result: object = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    res = BenchResult(name=name, avg_seconds=sum(times) / len(times), min_seconds=min(times))
    print(f"{res.name:40s} -> {res.avg_ms:8.2f} ms  (min: {res.min_ms:.2f} ms)")
    return result, res


def elementwise_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Reference implementation using NumPy ufuncs."""
    return np.exp(np.tan(a + b))


@njit(parallel=True, fastmath=True, cache=True)
def elementwise_numba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Numba parallel implementation with row-wise prange.

    Each row is independent so prange is applied only to the outer loop,
    avoiding any write contention on the output buffer.
    """
    n, m = a.shape
    c = np.empty((n, m), dtype=a.dtype)
    for i in prange(n):
        for j in range(m):
            c[i, j] = np.exp(np.tan(a[i, j] + b[i, j]))
    return c


def list_opencl_devices() -> None:
    """Print every OpenCL platform and device visible to the ICD loader."""
    print("\nAvailable OpenCL devices:")
    print("-" * 70)
    for platform in cl.get_platforms():
        print(f"  Platform: {platform.name.strip()}")
        for dev in platform.get_devices():
            dev_type = cl.device_type.to_string(dev.type)
            mem_mb = dev.global_mem_size // (1024**2)
            print(
                f"    [{dev_type}] {dev.name.strip()}"
                f"  ({dev.max_compute_units} CU, {mem_mb} MB)"
            )


def find_device(device_type: int) -> cl.Device | None:
    """Return the first device of the requested type across all platforms."""
    for platform in cl.get_platforms():
        try:
            devices = platform.get_devices(device_type=device_type)
        except cl.RuntimeError:
            continue
        if devices:
            return devices[0]
    return None


@dataclass(slots=True)
class OpenCLContext:
    """Bundle of OpenCL objects bound to one device, ready to dispatch."""

    device: cl.Device
    context: cl.Context
    queue: cl.CommandQueue
    program: cl.Program
    a_dev: cl_array.Array
    b_dev: cl_array.Array
    c_dev: cl_array.Array

    @property
    def total(self) -> np.int32:
        return np.int32(self.a_dev.size)


def setup_opencl(device: cl.Device, a: np.ndarray, b: np.ndarray) -> OpenCLContext:
    """Build a per-device OpenCL context and upload inputs once."""
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    program = cl.Program(context, KERNEL_SRC).build()
    a_dev = cl_array.to_device(queue, a)
    b_dev = cl_array.to_device(queue, b)
    c_dev = cl_array.empty_like(a_dev)
    return OpenCLContext(device, context, queue, program, a_dev, b_dev, c_dev)


def make_runners(ocl: OpenCLContext) -> tuple[Callable[[], None], Callable[[], np.ndarray]]:
    """Return `(kernel_only, kernel_plus_copy)` callables for benchmarking.

    `kernel_only` measures pure device compute, while `kernel_plus_copy`
    also includes the device-to-host transfer of the output.
    """

    def kernel_only() -> None:
        ocl.program.elementwise(
            ocl.queue,
            (ocl.a_dev.size,),
            None,
            ocl.a_dev.data,
            ocl.b_dev.data,
            ocl.c_dev.data,
            ocl.total,
        )
        ocl.queue.finish()

    def kernel_plus_copy() -> np.ndarray:
        kernel_only()
        return ocl.c_dev.get()

    return kernel_only, kernel_plus_copy


def run_opencl_backend(
    label: str,
    device_type: int,
    a: np.ndarray,
    b: np.ndarray,
    results: dict[str, BenchResult],
) -> np.ndarray | None:
    """Run both OpenCL timings for the first device of the given type.

    Stores results under keys `{label}-kernel` and `{label}-copy`.
    Returns the output array for correctness verification, or `None`
    when no matching device is available.
    """
    device = find_device(device_type)
    if device is None:
        type_name = cl.device_type.to_string(device_type)
        print(f"\n>> No OpenCL device of type {type_name} found")
        return None

    print(f"\n>> OpenCL {label}: {device.name.strip()}")
    ocl = setup_opencl(device, a, b)
    kernel_only, kernel_plus_copy = make_runners(ocl)

    _, results[f"{label}-kernel"] = benchmark(
        kernel_only, f"PyOpenCL {label} (kernel only)"
    )
    output, results[f"{label}-copy"] = benchmark(
        kernel_plus_copy, f"PyOpenCL {label} (kernel + D->H copy)"
    )
    return output


def verify(reference: np.ndarray, candidate: np.ndarray | None, name: str) -> None:
    """Print the max absolute deviation from the NumPy reference."""
    if candidate is None:
        return
    diff = float(np.abs(reference - candidate).max())
    print(f"  NumPy vs {name:14s} -> max diff: {diff:.2e}")


def print_summary(results: dict[str, BenchResult]) -> None:
    """Pretty-print a comparison table sorted by configured labels."""
    print("\n" + "=" * 70)
    print(f"{'Implementation':40s} {'Time (ms)':>12s} {'vs NumPy':>10s}")
    print("-" * 70)

    base = results["NumPy"].avg_seconds
    labels: dict[str, str] = {
        "NumPy": "NumPy (3 memory passes)",
        "Numba": "Numba (prange + fastmath, fused)",
        "CPU-kernel": "PyOpenCL CPU - kernel only",
        "CPU-copy": "PyOpenCL CPU - kernel + D->H copy",
        "GPU-kernel": "PyOpenCL GPU - kernel only",
        "GPU-copy": "PyOpenCL GPU - kernel + D->H copy",
    }
    for key, label in labels.items():
        if key not in results:
            continue
        r = results[key]
        print(f"{label:40s} {r.avg_ms:12.2f} {base / r.avg_seconds:9.2f}x")

    if "CPU-kernel" in results and "GPU-kernel" in results:
        gpu_vs_cpu_kernel = results["CPU-kernel"].avg_seconds / results["GPU-kernel"].avg_seconds
        gpu_vs_cpu_total = results["CPU-copy"].avg_seconds / results["GPU-copy"].avg_seconds
        print("\nWithin OpenCL:")
        print(f"  GPU vs CPU (kernel only): {gpu_vs_cpu_kernel:.2f}x")
        print(f"  GPU vs CPU (with copy):   {gpu_vs_cpu_total:.2f}x")


def main() -> None:
    """Entry point: build inputs, run all backends, print comparisons."""
    a, b = make_inputs(MATRIX_SIDE, RNG_SEED, INPUT_SCALE)

    list_opencl_devices()
    print(
        f"\nElement-wise: C = exp(tan(A + B))   shape={MATRIX_SIDE}x{MATRIX_SIDE}  "
        f"({a.size:,} elements, float32)"
    )
    print("=" * 70)

    results: dict[str, BenchResult] = {}
    c_numpy, results["NumPy"] = benchmark(elementwise_numpy, "NumPy", a, b)
    c_numba, results["Numba"] = benchmark(
        elementwise_numba, "Numba (prange + fastmath)", a, b
    )
    c_cpu = run_opencl_backend("CPU", cl.device_type.CPU, a, b, results)
    c_gpu = run_opencl_backend("GPU", cl.device_type.GPU, a, b, results)

    print("\nCorrectness check:")
    verify(c_numpy, c_numba, "Numba")
    verify(c_numpy, c_cpu, "OpenCL-CPU")
    verify(c_numpy, c_gpu, "OpenCL-GPU")

    print_summary(results)


if __name__ == "__main__":
    main()
