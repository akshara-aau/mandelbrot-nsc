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

def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter, n_workers):
    chunks = []
    rows_per_worker = N // n_workers
    for i in range(n_workers):
        row_start = i * rows_per_worker
        row_end = (i + 1) * rows_per_worker if i < n_workers - 1 else N
        chunks.append((row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter))
    with Pool(processes=n_workers) as pool:
        parts = pool.map(worker, chunks)
    return np.vstack(parts)

if __name__ == '__main__':
    from mandelbrot_functions import mandelbrot_numba
    N = 512
    result_serial = mandelbrot_serial(N, -2, 1, -1.5, 1.5, 100)
    result_old = mandelbrot_numba(-2, 1, -1.5, 1.5, N, N, 100)
    print("Serial vs L03 equal:", np.array_equal(result_serial, result_old))
    
    n_workers = 4
    result_parallel = mandelbrot_parallel(N, -2, 1, -1.5, 1.5, 100, n_workers)
    print("Parallel vs Serial equal:", np.array_equal(result_parallel, result_serial))