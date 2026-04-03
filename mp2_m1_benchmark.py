import time, statistics
import numpy as np
import pandas as pd
from dask import delayed, compute
from distributed import Client
from numba import njit
import warnings
warnings.filterwarnings('ignore')

# Implementation of Mandelbrot calculation
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

@njit
def mandelbrot_serial_numba(N, x_min, x_max, y_min, y_max, max_iter):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)

def mandelbrot_dask_distributed(N, x_min, x_max, y_min, y_max, max_iter, n_chunks):
    rows_per_chunk = N // n_chunks
    chunk_args = []
    for i in range(n_chunks):
        row_start = i * rows_per_chunk
        row_end = (i + 1) * rows_per_chunk if i < n_chunks - 1 else N
        chunk_args.append((row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter))

    delayed_tasks = [delayed(mandelbrot_chunk)(*args) for args in chunk_args]
    parts = compute(*delayed_tasks)
    return np.vstack(parts)

if __name__ == '__main__':
    # Configuration
    SCHEDULER_URL = "tcp://10.92.0.225:8786"
    N = 4096
    x_min, x_max, y_min, y_max = -2.5, 1.0, -1.25, 1.25
    max_iter = 100
    
    print("="*60)
    print(f"MP2 M1 BENCHMARK: DISTRIBUTED DASK (STRATO CLUSTER)")
    print("="*60)
    print(f"Grid size N: {N}")
    print(f"Target Scheduler: {SCHEDULER_URL}")
    
    # 1. Serial Numba Baseline (Local on the Mac)
    print("\nComputing Serial Numba Baseline (Warm-up)...")
    mandelbrot_serial_numba(8, x_min, x_max, y_min, y_max, max_iter)
    t0 = time.perf_counter()
    _ = mandelbrot_serial_numba(N, x_min, x_max, y_min, y_max, max_iter)
    t_serial = time.perf_counter() - t0
    print(f"Serial Numba time: {t_serial:.4f}s")
    
    # 2. Connect to Cluster
    print(f"\nConnecting to Dask Cluster at {SCHEDULER_URL}...")
    try:
        with Client(SCHEDULER_URL) as client:
            print(f"✓ Connected! Workers detected: {len(client.scheduler_info()['workers'])}")
            
            # Warm up JIT on all workers
            print("Warming up Numba on remote workers...")
            # Use a small chunk for warmup
            client.run(lambda: mandelbrot_chunk(0, 8, 8, x_min, x_max, y_min, y_max, max_iter))
            
            # Experiment: Chunk Size Sweep
            n_chunks_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
            results = []
            
            print("\n--- Running Chunk Size Sweep ---")
            for n_chunks in n_chunks_values:
                times = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    _ = mandelbrot_dask_distributed(N, x_min, x_max, y_min, y_max, max_iter, n_chunks)
                    times.append(time.perf_counter() - t0)
                
                t_median = statistics.median(times)
                results.append({'n_chunks': n_chunks, 'wall_time': t_median})
                print(f"Chunks: {n_chunks:4d} | Median Wall Time: {t_median:7.4f}s")
            
            df = pd.DataFrame(results)
            df['speedup'] = t_serial / df['wall_time']
            df.to_csv('mp2_m1_exp_distributed.csv', index=False)
            
            best_row = df.loc[df['wall_time'].idxmin()]
            print("\n" + "="*60)
            print("FINAL SUMMARY")
            print("="*60)
            print(f"Numba Serial Baseline: {t_serial:.4f}s")
            print(f"Best Distributed Time: {best_row['wall_time']:.4f}s")
            print(f"Max Speedup Achieved:  {best_row['speedup']:.2f}x")
            print(f"Optimal n_chunks:      {int(best_row['n_chunks'])}")
            print("\n✓ Results saved to mp2_m1_exp_distributed.csv")

    except Exception as e:
        print(f"\n❌ Error connecting to cluster: {e}")
        print("Make sure dask-scheduler is running on the Head Node and you are on the VPN.")
