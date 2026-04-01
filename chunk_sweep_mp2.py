import numpy as np
from numba import njit
from multiprocessing import Pool
import time, os, statistics

@njit
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        zr2 = z_real*z_real
        zi2 = z_imag*z_imag
        if zr2 + zi2 > 4.0: return i
        z_imag = 2.0*z_real*z_imag + c_imag
        z_real = zr2 - zi2 + c_real
    return max_iter

@njit
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / (N - 1)
    dy = (y_max - y_min) / (N - 1)
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col*dx, c_imag, max_iter)
    return out

def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)

def worker(args):
    return mandelbrot_chunk(*args)

def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter, n_workers, n_chunks=None, pool=None):
    if n_chunks is None:
        n_chunks = n_workers

    chunks = []
    rows_per_chunk = N // n_chunks
    for i in range(n_chunks):
        row_start = i * rows_per_chunk
        row_end = (i + 1) * rows_per_chunk if i < n_chunks - 1 else N
        chunks.append((row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter))

    if pool is None:
        with Pool(processes=n_workers) as pool:
            parts = pool.map(worker, chunks)
    else:
        parts = pool.map(worker, chunks)

    return np.vstack(parts)

if __name__ == '__main__':
    N = 512
    x_min, x_max, y_min, y_max, max_iter = -2, 1, -1.5, 1.5, 100

    # Get serial baseline
    serial_result = mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
    times = []
    for _ in range(5):  # More runs for better statistics
        t0 = time.perf_counter()
        _ = mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)
    print(f"Serial: {t_serial:.3f}s")

    # Fix n_workers at L04 optimum (10 workers based on previous n_chunks=n_workers results)
    n_workers_opt = 10
    print(f"\nFixed n_workers = {n_workers_opt} (L04 optimum)")
    print("n chunks | time (s) | vs. 1× | LIF")
    print("-" * 35)

    # Sweep n_chunks as multiples of n_workers
    chunk_multipliers = [1, 2, 4, 8, 16]

    with Pool(processes=n_workers_opt) as pool:
        for mult in chunk_multipliers:
            n_chunks = n_workers_opt * mult

            # Warm-up
            _ = mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter, n_workers_opt, n_chunks, pool)

            # Timed runs
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                _ = mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter, n_workers_opt, n_chunks, pool)
                times.append(time.perf_counter() - t0)
            t_p = statistics.median(times)

            # Calculate speedup and LIF
            speedup = t_serial / t_p
            lif = n_workers_opt * t_p / t_serial - 1

            print(f"{n_chunks:8d} | {t_p:8.3f} | {speedup:6.2f} | {lif:6.3f}")