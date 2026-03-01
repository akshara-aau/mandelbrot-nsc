import numpy as np
import time
import statistics
import matplotlib.pyplot as plt
from numba import njit

# =========================================================
# 1. NAIVE VERSION (Milestone 1 & 2)
# =========================================================
# Use @profile here only when running: kernprof -l -v script.py
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

# =========================================================
# 2. NUMPY VERSION (Vectorized)
# =========================================================
def mandelbrot_numpy(xmin, xmax, ymin, ymax, width, height, max_iter=100, dtype=np.complex128):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = (X + 1j * Y).astype(dtype)
    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=np.int32)
    
    for n in range(max_iter):
        mask = (Z.real**2 + Z.imag**2) <= 4.0
        Z[mask] = Z[mask]**2 + C[mask]
        M[mask] += 1
    return M

# =========================================================
# 3. NUMBA VERSION (Milestone 3 & 4)
# =========================================================
@njit
def mandelbrot_numba(xmin, xmax, ymin, ymax, width, height, max_iter=100, dtype=np.float64):
    x_vals = np.linspace(xmin, xmax, width).astype(dtype)
    y_vals = np.linspace(ymin, ymax, height).astype(dtype)
    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            c = x_vals[j] + 1j * y_vals[i]
            z = 0j
            n = 0
            # Milestone 4: The precision of these math ops depends on 'dtype'
            while n < max_iter and (z.real*z.real + z.imag*z.imag) <= 4.0:
                z = z * z + c
                n += 1
            result[i, j] = n
    return result

# =========================================================
# BENCHMARKING UTILITIES
# =========================================================
def bench(fn, *args, **kwargs):
    # Warm-up call (essential for Numba)
    fn(*args, **kwargs)
    
    runs = 5
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)

# =========================================================
# MAIN EXECUTION (Milestone 3 & 4 Results)
# =========================================================
if __name__ == "__main__":
    args = (-2, 1, -1.5, 1.5, 1024, 1024)
    
    print("--- Milestone 3: Implementation Benchmarks ---")
    t_naive = bench(mandelbrot_naive, *args)
    t_numpy = bench(mandelbrot_numpy, *args)
    t_numba = bench(mandelbrot_numba, *args, dtype=np.float64)

    print(f"Naive: {t_naive:.4f}s")
    print(f"NumPy: {t_numpy:.4f}s ({t_naive/t_numpy:.1f}x speedup)")
    print(f"Numba: {t_numba:.4f}s ({t_naive/t_numba:.1f}x speedup)")

    print("\n--- Milestone 4: Data Type Optimization ---")
    # float16 is skipped in Numba due to lack of CPU hardware support
    for dt in [np.float32, np.float64]:
        t_dt = bench(mandelbrot_numba, *args, dtype=dt)
        print(f"Numba {dt.__name__}: {t_dt:.4f}s")

    # Final Accuracy/Visual Comparison
    r16 = mandelbrot_numpy(*args, dtype=np.complex64) # Using numpy for 16-bit-ish check
    r32 = mandelbrot_numba(*args, dtype=np.float32)
    r64 = mandelbrot_numba(*args, dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    data = [r16, r32, r64]
    labels = ["float16 (NumPy)", "float32 (Numba)", "float64 (Numba)"]
    
    for ax, res, title in zip(axes, data, labels):
        ax.imshow(res, cmap='magma', extent=[-2, 1, -1.5, 1.5])
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("mandelbrot_precision.png")
    print("\nPlot saved as 'mandelbrot_precision.png'")
    
    max_diff = np.abs(r32 - r64).max()
    print(f"Max difference between float32 and float64: {max_diff}")
    print (f" Max diff float16 vs float64 : {np.abs(r16 - r64 ). max ()}")