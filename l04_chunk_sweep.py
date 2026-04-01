import time, statistics
from multiprocessing import Pool
from chunk_sweep_mp2 import mandelbrot_chunk, mandelbrot_parallel, worker as _worker

if __name__ == '__main__':
    N, max_iter = 1024, 100
    n_workers = 8  # adjust to your L04 optimum
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25

    # Warm up JIT
    mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)

    # Serial baseline
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_chunk(0, N, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)
    print(f"Serial: {t_serial:.3f}s")

    # Chunk-count sweep (M2): one Pool per config
    tiny = [(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)]

    for mult in [1, 2, 4, 8, 16]:
        n_chunks = mult * n_workers
        with Pool(processes=n_workers) as pool:
            pool.map(_worker, tiny)  # warm-up: load JIT in workers

            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                mandelbrot_parallel(
                    N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter,
                    n_workers=n_workers, n_chunks=n_chunks, pool=pool
                )
                times.append(time.perf_counter() - t0)
            t_par = statistics.median(times)

        lif = n_workers * t_par / t_serial - 1
        print(f"{n_chunks:4d} chunks {t_par:.3f}s {t_serial/t_par:.1f}x LIF={lif:.2f}")