import numpy as np
import matplotlib.pyplot as plt
import time
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
