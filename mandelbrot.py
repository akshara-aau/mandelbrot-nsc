import numpy as np
import matplotlib.pyplot as plt
import time

from numba import njit

#numpy version
# complex grid with 1024x1024 points linespace
x = np.linspace(-2, 1, 1024)
y = np.linspace(-1.5, 1.5, 1024)
X, Y = np.meshgrid(x, y)
C = X + 1j * Y

#verify the shape of C
# print(f"Shape of C: {C.shape}")
# print(f"Type  : {C.dtype}")

#  Initialize Z (current values) and M (iteration counts or mask)
Z = np.zeros_like(C)
M = np.zeros(C.shape, dtype=int)
max_iter = 100

# print("Starting vectorized timing...")
start = time.time()

#  The Vectorized Loop (Loop 3 only)
for n in range(max_iter):
    #  True for points that are still inside
    mask = np.abs(Z) <= 2
    
    # Update only the points that haven't escaped
    Z[mask] = Z[mask]**2 + C[mask]
    
    # Increment the iteration count for those points
    M[mask] += 1

elapsed = time.time() - start
# print(f"Finished! Vectorized computation took {elapsed:.3f} seconds")

#  drwaw the Mandelbrot set
# plt.figure(figsize=(10, 10))
# plt.imshow(M, extent=[-2, 1, -1.5, 1.5], cmap="magma")
# plt.colorbar(label="Iterations until escape")
# plt.title(f"Vectorized Mandelbrot Set ({elapsed:.3f}s)")
# plt.show()

#---------Milestone 3: Memory Access Patterns


N = 10000
A = np.random.rand(N, N) # Default is C-style (Row-major)

# 2. Row sums (Looping over rows)
def row_sums(arr):
    for i in range(N):
        s = np.sum(arr[i, :]) # Accessing a full row at once

# 3. Column sums (Looping over columns)
def col_sums(arr):
    for j in range(N):
        s = np.sum(arr[:, j]) # Accessing a full column at once

# 4. Timing the C-style array
print(f"Testing C-style array (Row-major)...")
start = time.time()
row_sums(A)
print(f"Row traversal: {time.time() - start:.4f}s")

start = time.time()
col_sums(A)
print(f"Column traversal: {time.time() - start:.4f}s")

# 5. Testing Fortran-style array (Column-major)
print(f"\nTesting Fortran-style array (Column-major)...")
A_f = np.asfortranarray(A)

start = time.time()
row_sums(A_f)
print(f"Row traversal (Fortran): {time.time() - start:.4f}s")

start = time.time()
col_sums(A_f)
print(f"Column traversal (Fortran): {time.time() - start:.4f}s")

#---------Milestone 4: Problem Size Scaling - mandelbrot again

def timed_mandelbrot(size, max_iter=100):
    x = np.linspace(-2, 1, size)
    y = np.linspace(-1.5, 1.5, size)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=int)
    
    start = time.time()
    for n in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        M[mask] += 1
    return time.time() - start

sizes = [256, 512, 1024, 2048, 4096]
runtimes = []

for s in sizes:
    t = timed_mandelbrot(s)
    runtimes.append(t)
    print(f"Size {s}x{s} ({s**2} pixels): {t:.4f}s")

# 3. Plotting the results
plt.figure(figsize=(8, 5))
plt.plot(sizes, runtimes, 'o-', label='Actual Runtime')
plt.xlabel('Grid Side Length (N)')
plt.ylabel('Time (seconds)')
plt.title('Mandelbrot Scaling: Grid Size vs Runtime')
plt.grid(True)
plt.show()



#numba JIT
@njit
def mandelbrot_numba(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    x_vals = np.linspace(xmin, xmax, width)
    y_vals = np.linspace(ymin, ymax, height)

    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):

            c = x_vals[j] + 1j * y_vals[i]
            z = 0j
            n = 0

            while n < max_iter and (z.real*z.real + z.imag*z.imag) <= 4.0:
                z = z*z + c
                n += 1

            result[i, j] = n

    return result


start = time.time()

result = mandelbrot_numba(-2, 1, -1.5, 1.5, 512, 512)

elapsed = time.time() - start
print(f"Numba computation took {elapsed:.3f} seconds")