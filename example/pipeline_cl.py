import numpy as np
import pyopencl as cl

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

N = 1024 * 1024
mf = cl.mem_flags

host_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.ALLOC_HOST_PTR, size=N * np.float32().nbytes)

mapped_array, event = cl.enqueue_map_buffer(
    queue,
    host_buf,
    cl.map_flags.WRITE,
    0,
    (N,),
    np.float32,
)
event.wait()

with mapped_array.base:
    mapped_array[:] = np.arange(N, dtype=np.float32)

kernel_src = """
__kernel void scale(__global float *data, const float factor) {
    int gid = get_global_id(0);
    data[gid] = data[gid] * factor;
}
"""

program = cl.Program(ctx, kernel_src).build()
program.scale(queue, (N,), None, host_buf, np.float32(2.5))

result_array, event = cl.enqueue_map_buffer(
    queue,
    host_buf,
    cl.map_flags.READ,
    0,
    (N,),
    np.float32,
)
event.wait()

with result_array.base:
    print("First 5 values:", result_array[:5])
    print("Last 5 values:", result_array[-5:])
    print("Mean:", result_array.mean())