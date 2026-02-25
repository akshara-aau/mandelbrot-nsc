import numpy as np
import matplotlib.pyplot as plt


xmin, xmax = -2.0, 1.0
ymin, ymax = -1.5, 1.5
width, height = 100, 100  # small grid for testing

x_vals = np.linspace(xmin, xmax, width)
y_vals = np.linspace(ymin, ymax, height)
#iteration function for a single point
def mandelbrot_point(c, max_iter=100):
    z = 0
    for n in range(max_iter):
        z = z*z + c
        if abs(z) > 2:
            return n
    return max_iter


print(mandelbrot_point(0, max_iter=100))
print(mandelbrot_point(2, max_iter=100))
print(mandelbrot_point(-0.75 + 0.1j, max_iter=100))



# Create a grid of complex numbers
def mandelbrot_grid(x_vals, y_vals, max_iter=100):
    height = len(y_vals)
    width = len(x_vals)

    result = np.zeros((height, width), dtype=int)

    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            c = x + 1j*y
            result[i, j] = mandelbrot_point(c, max_iter)

    return result


escape_counts = mandelbrot_grid(x_vals, y_vals, max_iter=100)
escape_counts.shape

plt.imshow(escape_counts, cmap="inferno")
plt.colorbar()
plt.title("Naive Mandelbrot (100x100)")
plt.show()



