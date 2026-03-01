import numpy as np
import time
import statistics
from numba import njit

# Hybrid approach: inner function compiled
@njit
def mandelbrot_point_numba(c, max_iter=100):
    z = 0j
    for n in range(max_iter):
        if (z.real * z.real + z.imag * z.imag) > 4.0:
            return n
        z = z * z + c
    return max_iter


def mandelbrot_hybrid(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    x_vals = np.linspace(xmin, xmax, width)
    y_vals = np.linspace(ymin, ymax, height)

    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):          # Python loop
        for j in range(width):       # Python loop
            c = x_vals[j] + 1j * y_vals[i]
            result[i, j] = mandelbrot_point_numba(c, max_iter)

    return result


# Fully compiled approach (recommended)
@njit
def mandelbrot_naive_numba(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    x_vals = np.linspace(xmin, xmax, width)
    y_vals = np.linspace(ymin, ymax, height)

    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):          # compiled
        for j in range(width):       # compiled
            c = x_vals[j] + 1j * y_vals[i]
            z = 0j
            n = 0

            while n < max_iter and (z.real*z.real + z.imag*z.imag) <= 4.0:
                z = z * z + c
                n += 1

            result[i, j] = n

    return result


# Benchmark helper
def bench(fn, *args, runs=5):
    fn(*args)  # extra warm-up
    times = []

    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)

    return statistics.median(times)


# -------------------------------------------------
# Warm-up (JIT compilation — do NOT time)
# -------------------------------------------------
_ = mandelbrot_hybrid(-2, 1, -1.5, 1.5, 64, 64)
_ = mandelbrot_naive_numba(-2, 1, -1.5, 1.5, 64, 64)

# -------------------------------------------------
# Benchmark
# -------------------------------------------------
t_hybrid = bench(mandelbrot_hybrid, -2, 1, -1.5, 1.5, 1024, 1024)
t_full = bench(mandelbrot_naive_numba, -2, 1, -1.5, 1.5, 1024, 1024)

print(f"Hybrid: {t_hybrid:.3f} s")
print(f"Fully compiled: {t_full:.3f} s")
print(f"Ratio: {t_hybrid / t_full:.1f}x")