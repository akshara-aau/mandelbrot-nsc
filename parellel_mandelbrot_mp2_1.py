import numpy as np
from numba import njit
from multiprocessing import Pool
import time, os, statistics, matplotlib.pyplot as plt
from pathlib import Path

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
    
    # Warm-up serial (JIT in main process)
    serial_result = mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
    
    # Benchmark serial
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        _ = mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)
    print(f"Serial: {t_serial:.3f}s")
    
    # Verify parallel output matches serial
    print("\n--- Verification: n_chunks = n_workers ---")
    for p in range(1, min(5, os.cpu_count() + 1)):
        parallel_result = mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter, p, n_chunks=p)
        match = np.array_equal(serial_result, parallel_result)
        print(f"{p} workers: Output matches serial = {match}")
    
    # Benchmark parallel for each n_workers with default n_chunks = n_workers
    print("\n--- Benchmark: n_chunks = n_workers ---")
    for p in range(1, os.cpu_count() + 1):
        with Pool(processes=p) as pool:
            # Warm-up: trigger JIT in workers
            _ = mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter, p, n_chunks=p, pool=pool)
            
            # Timed runs
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                _ = mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter, p, n_chunks=p, pool=pool)
                times.append(time.perf_counter() - t0)
        t_p = statistics.median(times)
        Sp = t_serial / t_p
        Ep = Sp / p
        print(f"{p:2d} workers: {t_p:.3f}s speedup={Sp:.2f} efficiency={Ep:.2f}")
    
    # Benchmark parallel with n_chunks = 4 * n_workers
    print("\n--- Benchmark: n_chunks = 4 × n_workers ---")
    for p in range(1, os.cpu_count() + 1):
        n_chunks = 4 * p
        with Pool(processes=p) as pool:
            # Warm-up: trigger JIT in workers
            _ = mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter, p, n_chunks=n_chunks, pool=pool)
            
            # Timed runs
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                _ = mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter, p, n_chunks=n_chunks, pool=pool)
                times.append(time.perf_counter() - t0)
        t_p = statistics.median(times)
        Sp = t_serial / t_p
        Ep = Sp / p
        print(f"{p:2d} workers, {n_chunks:2d} chunks: {t_p:.3f}s speedup={Sp:.2f} efficiency={Ep:.2f}")