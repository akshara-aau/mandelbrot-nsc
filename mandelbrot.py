import numpy as np
import matplotlib.pyplot as plt
import time

# complex grid with 1024x1024 points linespace
x = np.linspace(-2, 1, 1024)
y = np.linspace(-1.5, 1.5, 1024)
X, Y = np.meshgrid(x, y)
C = X + 1j * Y

#verify the shape of C
print(f"Shape of C: {C.shape}")
print(f"Type  : {C.dtype}")

#  Initialize Z (current values) and M (iteration counts or mask)
Z = np.zeros_like(C)
M = np.zeros(C.shape, dtype=int)
max_iter = 100

print("Starting vectorized timing...")
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
print(f"Finished! Vectorized computation took {elapsed:.3f} seconds")

#  drwaw the Mandelbrot set
plt.figure(figsize=(10, 10))
plt.imshow(M, extent=[-2, 1, -1.5, 1.5], cmap="magma")
plt.colorbar(label="Iterations until escape")
plt.title(f"Vectorized Mandelbrot Set ({elapsed:.3f}s)")
plt.show()