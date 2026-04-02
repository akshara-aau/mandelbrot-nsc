import time, statistics
import numpy as np
from dask import delayed, compute
from distributed import Client
from chunk_sweep_mp2 import mandelbrot_chunk, mandelbrot_serial


def mandelbrot_dask_local(N, x_min, x_max, y_min, y_max, max_iter, n_chunks):
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
    n_chunks = 8  # same as workers for this milestone

    # Start local cluster client
    client = Client(n_workers=n_workers, threads_per_worker=1, processes=True)
    print(f"Dask dashboard: {client.dashboard_link}")

    # Warm up JIT both locally and across workers
    print("Warm up serial and Dask JIT...")
    mandelbrot_chunk(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)
    client.run(lambda: mandelbrot_chunk(0, 8, 8, x_min, x_max, y_min, y_max, max_iter))

    # Serial baseline to compare
    serial = mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter)

    # Test correctness with single Dask run
    dask_result = mandelbrot_dask_local(N, x_min, x_max, y_min, y_max, max_iter, n_chunks)

    equal = np.array_equal(serial, dask_result)
    print(f"Correctness np.array_equal(serial, dask) = {equal}")
    if not equal:
        raise RuntimeError("Dask result differs from serial result")

    # Timing 3 runs and median
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        _ = mandelbrot_dask_local(N, x_min, x_max, y_min, y_max, max_iter, n_chunks)
        times.append(time.perf_counter() - t0)

    t_median = statistics.median(times)
    print(f"Dask local (n_workers={n_workers}, n_chunks={n_chunks}): median time = {t_median:.3f}s")

    client.close()
