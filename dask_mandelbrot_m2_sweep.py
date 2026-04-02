import time, statistics
import numpy as np
import matplotlib.pyplot as plt
from dask import delayed, compute
from distributed import Client
from chunk_sweep_mp2 import mandelbrot_chunk, mandelbrot_serial


def mandelbrot_dask_local(N, x_min, x_max, y_min, y_max, max_iter, n_chunks):
    """
    Dask local implementation with n_chunks tasks (one per row-band).
    """
    if n_chunks <= 0:
        raise ValueError("n_chunks must be > 0")

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
    N = 1024
    x_min, x_max, y_min, y_max = -2.5, 1.0, -1.25, 1.25
    max_iter = 100
    n_workers = 8

    # Start local cluster once, keep open throughout sweep
    client = Client(n_workers=n_workers, threads_per_worker=1, processes=True)
    print(f"Dask cluster started: {client.dashboard_link}")

    # Warm up JIT locally and across workers
    print("Warming up serial and Dask JIT...")
    mandelbrot_chunk(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)
    client.run(lambda: mandelbrot_chunk(0, 8, 8, x_min, x_max, y_min, y_max, max_iter))

    # Get serial baseline
    print("Computing serial baseline...")
    serial_result = mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)

    # Verify correctness once with first configuration
    print("Verifying correctness...")
    dask_result = mandelbrot_dask_local(N, x_min, x_max, y_min, y_max, max_iter, n_workers)
    assert np.array_equal(serial_result, dask_result), "Dask result differs from serial!"
    print("✓ Correctness verified\n")

    # Chunk size sweep
    n_chunks_values = [1, 2, 4, 8, 16, 32, 64]
    results = {
        'n_chunks': [],
        'times': [],
        'lif': [],
    }

    # Compute serial time once
    times_serial = []
    for _ in range(3):
        t0 = time.perf_counter()
        _ = mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)
        times_serial.append(time.perf_counter() - t0)
    t_serial = statistics.median(times_serial)
    print(f"Serial baseline: {t_serial:.4f}s\n")

    print(f"{'n_chunks':>8} | {'time (s)':>10} | {'LIF':>8}")
    print("-" * 35)

    for n_chunks in n_chunks_values:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = mandelbrot_dask_local(N, x_min, x_max, y_min, y_max, max_iter, n_chunks)
            times.append(time.perf_counter() - t0)

        t_median = statistics.median(times)
        # LIF = p * T_p / T_1 - 1
        lif = n_workers * t_median / t_serial - 1

        results['n_chunks'].append(n_chunks)
        results['times'].append(t_median)
        results['lif'].append(lif)

        print(f"{n_chunks:8d} | {t_median:10.4f} | {lif:8.3f}")

    # Find optimal (minimum time)
    min_idx = np.argmin(results['times'])
    optimal_n_chunks = results['n_chunks'][min_idx]
    min_time = results['times'][min_idx]
    min_lif = results['lif'][min_idx]

    print("-" * 35)
    print(f"Optimal n_chunks: {optimal_n_chunks}")
    print(f"Minimum time: {min_time:.4f}s")
    print(f"Minimum LIF: {min_lif:.3f}\n")

    # Plot wall time vs n_chunks on log scale
    plt.figure(figsize=(10, 6))
    plt.loglog(results['n_chunks'], results['times'], 'bo-', linewidth=2, markersize=8, label='Dask local')
    plt.axhline(t_serial, color='r', linestyle='--', linewidth=2, label=f'Serial: {t_serial:.4f}s')
    plt.xlabel('Number of Chunks (n_chunks)', fontsize=12)
    plt.ylabel('Wall Time (seconds)', fontsize=12)
    plt.title(f'MP2 M2: Dask Chunk Size Sweep (N={N}, workers={n_workers})', fontsize=14)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('dask_m2_sweep.png', dpi=150)
    print("Plot saved as dask_m2_sweep.png")

    client.close()
    print("Dask cluster closed.")
