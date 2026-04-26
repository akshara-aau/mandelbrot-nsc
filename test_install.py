import numpy
import matplotlib
import scipy
import numba
import pytest
try:
    import dask
    HAS_DASK = True
except ImportError:
    HAS_DASK = False


print("Environment Test Successful!")
print(f"NumPy version: {numpy.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"SciPy version: {scipy.__version__}")
print(f"Numba version: {numba.__version__}")
if HAS_DASK:
    print(f"Dask version: {dask.__version__}")
else:
    print("Dask version: Not installed")

