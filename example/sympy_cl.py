import sympy as sp
import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np

x = sp.Symbol('x')
expr = sp.sin(x) * sp.exp(-x / 2)

opencl_code = sp.ccode(expr)

print("C code: ", opencl_code)

kernel_source = """
__kernel void math_kernel(__global const float *a, __global float *out) {
    int gid = get_global_id(0);
    
    float x = a[gid];
    out[gid] = %s;
}
""" % opencl_code

print("Kernel code: ", kernel_source)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

h_a = np.random.rand(10000).astype(np.float32)
d_a = cl_array.to_device(queue, h_a)
d_out = cl_array.empty_like(d_a)

prg = cl.Program(ctx, kernel_source).build()
prg.math_kernel(queue, h_a.shape, None, d_a.data, d_out.data)

h_out = d_out.get()

print("First 5 computed values on GPU:", h_out[:5])
