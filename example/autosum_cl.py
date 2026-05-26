import numpy as np
import pyopencl as cl

platforms = cl.get_platforms()

platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

matrix_shape = (1024, 1024)
h_matrix = np.random.rand(*matrix_shape).astype(np.float32)

h_result = np.zeros(1, dtype=np.float32)

print(True, f"Theoretical sum (Numpy): {h_matrix.sum()}")

mf = cl.mem_flags
d_matrix = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_matrix)
d_result = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_result)

kernel_code = """
__kernel void global_sum_reduction(__global const float *matrix, 
                                   __global float *result, 
                                   __local float *local_cache, 
                                   const int total_elements) {

    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_size = get_local_size(0);

    if (global_id < total_elements) {
        local_cache[local_id] = matrix[global_id];
    } else {
        local_cache[local_id] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            local_cache[local_id] += local_cache[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        union { unsigned int intVal; float floatVal; } next, expected, current;
        current.floatVal = *result;
        do {
            expected.intVal = current.intVal;
            next.floatVal = expected.floatVal + local_cache[0];
            current.intVal = atom_cmpxchg((__global unsigned int*)result, expected.intVal, next.intVal);
        } while (current.intVal != expected.intVal);
    }
}
"""

program = cl.Program(context, kernel_code).build()

total_elements = h_matrix.size
local_size = 256

global_size = int(np.ceil(total_elements / local_size) * local_size)

program.global_sum_reduction(
    queue,
    (global_size,),
    (local_size,),
    d_matrix,
    d_result,
    cl.LocalMemory(local_size * 4),
    np.int32(total_elements)
)

cl.enqueue_copy(queue, h_result, d_result)

print(f"Sum calculated on GPU: {h_result[0]}")

assert np.allclose(h_result[0], h_matrix.sum(), rtol=1e-3)

print("Success! The parallel autosum matches perfectly")