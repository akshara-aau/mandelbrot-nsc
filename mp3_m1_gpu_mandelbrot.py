import numpy as np
import pyopencl as cl
import time
import matplotlib.pyplot as plt


# This is a direct translation of the Numba loop into C
KERNEL_SRC = """
__kernel void mandelbrot(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int width, const int height,
    const int max_iter
) {
    // Get 2D global indexes
    int col = get_global_id(0);
    int row = get_global_id(1);

    // Safety check for bounds
    if (col < width && row < height) {
        // Map pixel to complex coordinate (c = x + iy)
        float x = x_min + col * (x_max - x_min) / (width - 1);
        float y = y_min + row * (y_max - y_min) / (height - 1);

        float zr = 0.0f;
        float zi = 0.0f;
        int n = 0;

        // Escape time loop
        while (n < max_iter && (zr*zr + zi*zi) <= 4.0f) {
            float zr_new = zr*zr - zi*zi + x;
            zi = 2.0f * zr * zi + y;
            zr = zr_new;
            n++;
        }

        // Store result in 1D flattened array
        result[row * width + col] = n;
    }
}
"""

def run_gpu_mandelbrot(N=1024, max_iter=1000):
    # Setup OpenCL
    # Use apple platform for M4
    platforms = cl.get_platforms()
    platform = [p for p in platforms if p.name == 'Apple'][0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    width = np.int32(N)
    height = np.int32(N)
    max_iter_val = np.int32(max_iter)
    xmin, xmax = np.float32(-2.0), np.float32(1.0)
    ymin, ymax = np.float32(-1.5), np.float32(1.5)

    h_result = np.zeros((height, width), dtype=np.int32)
    d_result = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, h_result.nbytes)

    prog = cl.Program(ctx, KERNEL_SRC).build()
    knl = prog.mandelbrot

    knl(queue, (width, height), None, 
        d_result, xmin, xmax, ymin, ymax, width, height, max_iter_val)
    queue.finish()

    print(f"Running GPU Mandelbrot ({N}x{N}, max_iter={max_iter})")
    t0 = time.perf_counter()
    
    knl(queue, (width, height), None, 
        d_result, 
        xmin, xmax, 
        ymin, ymax, 
        width, height, 
        max_iter_val)
    
    queue.finish()
    t_gpu = time.perf_counter() - t0
    
    cl.enqueue_copy(queue, h_result, d_result)
    
    print(f"GPU Runtime: {t_gpu:.4f}s")
    return h_result, t_gpu

if __name__ == "__main__":
    res, t_gpu = run_gpu_mandelbrot(N=1024, max_iter=1000)
    
    # Visualize
    plt.figure(figsize=(10, 10))
    plt.imshow(res, cmap='magma', extent=[-2.0, 1.0, -1.5, 1.5])
    plt.title(f"OpenCL GPU Mandelbrot (M4) - {t_gpu:.4f}s")
    plt.colorbar(label='Iterations')
    plt.axis('off')
    plt.savefig("mandelbrot_gpu_result.png")
    print("Result saved to 'mandelbrot_gpu_result.png'")
