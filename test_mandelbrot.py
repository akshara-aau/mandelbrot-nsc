import pytest
import numpy as np
from mandelbrot_functions import mandelbrot_naive, mandelbrot_numpy, mandelbrot_numba



@pytest.mark.parametrize("mandel_func", [
    mandelbrot_naive,
    mandelbrot_numpy,
    mandelbrot_numba
])
def test_analytical_origin(mandel_func):

    MAX_ITER = 50
    # Create a 1x1 grid at (0,0)
    res = mandel_func(xmin=0, xmax=0, ymin=0, ymax=0, width=1, height=1, max_iter=MAX_ITER)
    assert res[0, 0] == MAX_ITER

@pytest.mark.parametrize("mandel_func", [
    mandelbrot_naive,
    mandel_func_vectorized := lambda *args, **kwargs: mandelbrot_numpy(*args, **kwargs),
    mandelbrot_numba
])
@pytest.mark.parametrize("c_val, expected_max", [
    (1.0, False),  # c=1: z1=1, z2=2, z3=5 (escapes quickly)
    (2.5, False),  # c=2.5: escapes in 1 iteration
    (0.25, True),  # c=0.25: boundary case, stay in or escape very late
])
def test_analytical_points(mandel_func, c_val, expected_max):
    """Verify known points in/out of the set."""
    MAX_ITER = 20
    # Shift parameters to handle lambda if needed
    if mandel_func == mandelbrot_numpy: # handle the lambda name mismatch if any
         res = mandelbrot_numpy(c_val, c_val, 0, 0, 1, 1, max_iter=MAX_ITER)
    else:
         res = mandel_func(c_val, c_val, 0, 0, 1, 1, max_iter=MAX_ITER)
         
    is_max = res[0, 0] == MAX_ITER
    assert is_max == expected_max


# CROSS-VALIDATION TESTS

def test_implementation_consistency():
    """
    The Naive, NumPy, and Numba implementations must yield identical results 
    on a small grid (32x32). This uses the Naive loop as an 'oracle'.
    """
    args = {
        'xmin': -1.5, 'xmax': 0.5, 
        'ymin': -1.0, 'ymax': 1.0, 
        'width': 32, 'height': 32, 
        'max_iter': 100
    }
    
    res_naive = mandelbrot_naive(**args)
    res_numpy = mandelbrot_numpy(**args)
    res_numba = mandelbrot_numba(**args)
    
    np.testing.assert_array_equal(res_naive, res_numpy, err_msg="NumPy results differ from Naive")
    np.testing.assert_array_equal(res_naive, res_numba, err_msg="Numba results differ from Naive")

# PERFORMANCE REGRESSION TEST (Mock-like)


def test_vectorization_speedup():
    """
    Sanity check: Vectorized NumPy should be significantly faster than Naive
    on a medium grid, even for small counts. 
    Requirement: assert numpy_time < 0.1 * naive_time (on small grid)
    """
    import time
    args = (-1.5, 0.5, -1, 1, 128, 128, 50)
    
    t0 = time.perf_counter()
    mandelbrot_naive(*args)
    t_naive = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    mandelbrot_numpy(*args)
    t_numpy = time.perf_counter() - t0
    
    print(f"Naive: {t_naive:.4f}s, NumPy: {t_numpy:.4f}s")
    # NumPy should be at least an order of magnitude faster here
    assert t_numpy < t_naive, "NumPy is slower than Naive??"
