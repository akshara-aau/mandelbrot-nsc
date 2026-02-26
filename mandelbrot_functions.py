import numpy as np
import time

# ---  NAIVE VERSION  ---
@profile # Add this decorator
def mandelbrot_naive(x_min, x_max, y_min, y_max, width, height, max_iter=100):
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    m = np.zeros((height, width), dtype=int)
    
    for i in range(height):
        for j in range(width):
            c = complex(x[j], y[i])
            z = 0j
            for n in range(max_iter):
                if abs(z) <= 2:
                    z = z*z + c
                    m[i, j] += 1
                else:
                    break
    return m

# ---  NUMPY ---
def mandelbrot_numpy(x_min, x_max, y_min, y_max, width, height, max_iter=100):
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=int)
    
    for n in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        M[mask] += 1
    return M

# Add this at the very bottom of your script
if __name__ == "__main__":
    mandelbrot_naive(-2, 1, -1.5, 1.5, 512, 512)