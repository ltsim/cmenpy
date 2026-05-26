import numpy as np
import pyopencl as cl


class AckleyCL(object):
    KERNEL_SRC = """
    __kernel void ackley_partials(
        __global const float *x,
        __global float *sum_sq,
        __global float *sum_cos,
        const float c,
        const unsigned int n,
        __local float *loc_sq,
        __local float *loc_cos)
    {
        unsigned int gid = get_global_id(0);
        unsigned int lid = get_local_id(0);
        unsigned int lsize = get_local_size(0);
        unsigned int grp = get_group_id(0);

        float val_sq = 0.0f;
        float val_cos = 0.0f;
        if (gid < n) {
            float xv = x[gid];
            val_sq = xv * xv;
            val_cos = cos(c * xv);
        }

        loc_sq[lid] = val_sq;
        loc_cos[lid] = val_cos;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int s = lsize / 2; s > 0; s >>= 1) {
            if (lid < s) {
                loc_sq[lid] += loc_sq[lid + s];
                loc_cos[lid] += loc_cos[lid + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            sum_sq[grp] = loc_sq[0];
            sum_cos[grp] = loc_cos[0];
        }
    }
    """

    def __init__(self, device: str = "gpu"):
        device_type_map = {
            "gpu": cl.device_type.GPU,
            "cpu": cl.device_type.CPU,
        }

        target = device.lower()
        if target not in device_type_map:
            raise ValueError("device must be 'cpu' or 'gpu'")

        self.__selected_device = None

        for platform in cl.get_platforms():
            try:
                devices = platform.get_devices(device_type=device_type_map[target])
            except cl.RuntimeError:
                devices = []

            if devices:
                self.__selected_device = devices[0]
                break

        if self.__selected_device is None:
            raise RuntimeError(f"No OpenCL {target.upper()} device available")

        self.__ctx = cl.Context([self.__selected_device])
        self.__queue = cl.CommandQueue(self.__ctx)

        self.__program = cl.Program(self.__ctx, self.KERNEL_SRC).build()

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(self, x: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2.0 * np.pi) -> float:
        n = x.size

        if n == 0:
            raise ValueError("Input array must not be empty")

        if self.__selected_device is None:
            raise RuntimeError("No OpenCL device available")

        max_wg = self.__selected_device.max_work_group_size
        local_size = 1

        while local_size * 2 <= max_wg and local_size * 2 <= 256:
            local_size *= 2

        num_groups = (n + local_size - 1) // local_size
        global_size = num_groups * local_size

        mf = cl.mem_flags
        x_buf = cl.Buffer(self.__ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
        sum_sq_buf = cl.Buffer(self.__ctx, mf.WRITE_ONLY, size=num_groups * np.dtype(np.float32).itemsize)
        sum_cos_buf = cl.Buffer(self.__ctx, mf.WRITE_ONLY, size=num_groups * np.dtype(np.float32).itemsize)

        self.__program.ackley_partials(
            self.__queue,
            (global_size,),
            (local_size,),
            x_buf,
            sum_sq_buf,
            sum_cos_buf,
            np.float32(c),
            np.uint32(n),
            cl.LocalMemory(local_size * np.dtype(np.float32).itemsize),
            cl.LocalMemory(local_size * np.dtype(np.float32).itemsize),
        )

        partial_sq = np.empty(num_groups, dtype=np.float32)
        partial_cos = np.empty(num_groups, dtype=np.float32)

        cl.enqueue_copy(self.__queue, partial_sq, sum_sq_buf)
        cl.enqueue_copy(self.__queue, partial_cos, sum_cos_buf)

        self.__queue.finish()

        sum_sq = float(partial_sq.sum())
        sum_cos = float(partial_cos.sum())

        term1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
        term2 = -np.exp(sum_cos / n)
        result = term1 + term2 + a + np.e

        return float(result)
