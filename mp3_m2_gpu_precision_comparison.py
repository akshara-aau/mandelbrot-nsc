import numpy as np
import pyopencl as cl
import time
import matplotlib.pyplot as plt


# 1. FLOAT32 KERNEL

KERNEL_F32 = """
__kernel void mandelbrot_f32(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int width, const int height,
    const int max_iter
) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (col < width && row < height) {
        float x = x_min + col * (x_max - x_min) / (width - 1);
        float y = y_min + row * (y_max - y_min) / (height - 1);

        float zr = 0.0f;
        float zi = 0.0f;
        int n = 0;

        while (n < max_iter && (zr*zr + zi*zi) <= 4.0f) {
            float zr_new = zr*zr - zi*zi + x;
            zi = 2.0f * zr * zi + y;
            zr = zr_new;
            n++;
        }
        result[row * width + col] = n;
    }
}
"""


# 2. FLOAT64 KERNEL (Emulated on Mac)

KERNEL_F64 = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void mandelbrot_f64(
    __global int *result,
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const int width, const int height,
    const int max_iter
) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (col < width && row < height) {
        double x = x_min + col * (x_max - x_min) / (double)(width - 1);
        double y = y_min + row * (y_max - y_min) / (double)(height - 1);

        double zr = 0.0;
        double zi = 0.0;
        int n = 0;

        while (n < max_iter && (zr*zr + zi*zi) <= 4.0) {
            double zr_new = zr*zr - zi*zi + x;
            zi = 2.0 * zr * zi + y;
            zr = zr_new;
            n++;
        }
        result[row * width + col] = n;
    }
}
"""

def bench_precision(N=1024, max_iter=1000):
    # Setup
    platforms = cl.get_platforms()
    platform = [p for p in platforms if p.name == 'Apple'][0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    if 'cl_khr_fp64' not in device.extensions:
        print("No native fp64 support -- Apple Silicon: emulated, expect large slowdown")

    # Build programs
    prog_f32 = cl.Program(ctx, KERNEL_F32).build()
    
    try:
        prog_f64 = cl.Program(ctx, KERNEL_F64).build()
        f64_available = True
    except Exception as e:
        print(f"Error building f64 kernel: {e}")
        f64_available = False

    def run_kernel(knl, dtype_float, name):
        width = np.int32(N)
        height = np.int32(N)
        max_iter_val = np.int32(max_iter)
        
        xmin, xmax = dtype_float(-2.0), dtype_float(1.0)
        ymin, ymax = dtype_float(-1.5), dtype_float(1.5)

        h_res = np.zeros((N, N), dtype=np.int32)
        d_res = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, h_res.nbytes)

        knl(queue, (N, N), None, d_res, xmin, xmax, ymin, ymax, width, height, max_iter_val)
        queue.finish()

        t0 = time.perf_counter()
        knl(queue, (N, N), None, d_res, xmin, xmax, ymin, ymax, width, height, max_iter_val)
        queue.finish()
        t_total = time.perf_counter() - t0
        
        cl.enqueue_copy(queue, h_res, d_res)
        print(f"[{name}] N={N}: {t_total:.4f}s")
        return h_res, t_total

    print(f"--- Precision Benchmark (N={N}) ---")
    _, t32 = run_kernel(prog_f32.mandelbrot_f32, np.float32, "Float32")
    
    if f64_available:
        _, t64 = run_kernel(prog_f64.mandelbrot_f64, np.float64, "Float64")
        print(f"Speed Ratio (f32/f64): {t64/t32:.1f}x slower for f64")
    else:
        t64 = None

    return t32, t64

if __name__ == "__main__":
    for size in [1024, 2048]:
        bench_precision(N=size, max_iter=1000)
