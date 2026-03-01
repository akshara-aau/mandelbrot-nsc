import numpy as np
import time
import statistics
from numba import njit

# --- 1. NAIVE VERSION (For Milestone 2 Profiling) ---
#  Keep the @profile decorator only when running line_profiler
def mandelbrot_naive(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    x_vals = np.linspace(xmin, xmax, width)
    y_vals = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width), dtype=np.int32)
    
    for i in range(height):
        for j in range(width):
            c = complex(x_vals[j], y_vals[i])
            z = 0j
            n = 0
            while n < max_iter and (z.real**2 + z.imag**2) <= 4.0:
                z = z*z + c
                n += 1
            result[i, j] = n
    return result

# --- 2. NUMPY VERSION (Vectorized) ---
def mandelbrot_numpy(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=np.int32)
    
    for n in range(max_iter):
        mask = (Z.real**2 + Z.imag**2) <= 4.0
        Z[mask] = Z[mask]**2 + C[mask]
        M[mask] += 1
    return M

# --- 3. NUMBA VERSION (Fully Compiled - Milestone 3) ---
@njit
def mandelbrot_numba(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    x_vals = np.linspace(xmin, xmax, width)
    y_vals = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            c = x_vals[j] + 1j * y_vals[i]
            z = 0j
            n = 0
            while n < max_iter and (z.real*z.real + z.imag*z.imag) <= 4.0:
                z = z * z + c
                n += 1
            result[i, j] = n
    return result

# --- BENCHMARK HELPER ---
def bench(fn, *args, runs=5):
    # Warm-up (especially important for Numba JIT)
    fn(*args)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)

# --- EXECUTION ---
if __name__ == "__main__":
    args = (-2, 1, -1.5, 1.5, 1024, 1024)
    
    print("Starting benchmarks...")
    t_naive = bench(mandelbrot_naive, *args)
    t_numpy = bench(mandelbrot_numpy, *args)
    t_numba = bench(mandelbrot_numba, *args)

    print(f"Naive: {t_naive:.3f}s")
    print(f"NumPy: {t_numpy:.3f}s ({t_naive / t_numpy:.1f}x speedup)")
    print(f"Numba: {t_numba:.3f}s ({t_naive / t_numba:.1f}x speedup)")