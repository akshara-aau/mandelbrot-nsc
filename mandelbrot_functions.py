import numpy as np
import time
import statistics
import matplotlib.pyplot as plt
from numba import njit
from typing import Any, Callable, Type



# naive
def mandelbrot_naive(
    xmin: float, 
    xmax: float, 
    ymin: float, 
    ymax: float, 
    width: int, 
    height: int, 
    max_iter: int = 100
) -> np.ndarray:
    
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


# numpy
def mandelbrot_numpy(
    xmin: float, 
    xmax: float, 
    ymin: float, 
    ymax: float, 
    width: int, 
    height: int, 
    max_iter: int = 100, 
    dtype: Type[np.complexfloating] = np.complex128
) -> np.ndarray:
   
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = (X + 1j * Y).astype(dtype)
    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=np.int32)
    
    for _ in range(max_iter):
        mask = (Z.real**2 + Z.imag**2) <= 4.0
        Z[mask] = Z[mask]**2 + C[mask]
        M[mask] += 1
    return M


#numba
@njit
def mandelbrot_numba(
    xmin: float, 
    xmax: float, 
    ymin: float, 
    ymax: float, 
    width: int, 
    height: int, 
    max_iter: int = 100, 
    dtype: Type[np.floating] = np.float64
) -> np.ndarray:
    
    x_vals = np.linspace(xmin, xmax, width).astype(dtype)
    y_vals = np.linspace(ymin, ymax, height).astype(dtype)
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


# bench mark
def bench(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> float:
  
    # Warm-up call (essential for Numba compilation or cache)
    fn(*args, **kwargs)
    
    runs = 5
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(statistics.median(times))


# main
if __name__ == "__main__":
    standard_args = (-2.0, 1.0, -1.5, 1.5, 1024, 1024)
    
    print("--- Milestone 3: Implementation Benchmarks ---")
    t_naive = bench(mandelbrot_naive, *standard_args)
    t_numpy = bench(mandelbrot_numpy, *standard_args)
    t_numba = bench(mandelbrot_numba, *standard_args, dtype=np.float64)

    print(f"Naive: {t_naive:.4f}s")
    print(f"NumPy: {t_numpy:.4f}s ({t_naive/t_numpy:.1f}x speedup)")
    print(f"Numba: {t_numba:.4f}s ({t_naive/t_numba:.1f}x speedup)")

    print("\n--- Milestone 4: Data Type Optimization ---")
    for dt in [np.float32, np.float64]:
        t_dt = bench(mandelbrot_numba, *standard_args, dtype=dt)
        print(f"Numba {dt.__name__}: {t_dt:.4f}s")

    # Visual Comparison
    res32 = mandelbrot_numba(*standard_args, dtype=np.float32)
    res64 = mandelbrot_numba(*standard_args, dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(res32, cmap='magma', extent=[-2, 1, -1.5, 1.5])
    axes[0].set_title("float32 (Numba)")
    axes[1].imshow(res64, cmap='magma', extent=[-2, 1, -1.5, 1.5])
    axes[1].set_title("float64 (Numba)")
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("mandelbrot_precision_comparison.png")
    print("\nPlot saved as 'mandelbrot_precision_comparison.png'")