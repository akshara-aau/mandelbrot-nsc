import cProfile, pstats
from mandelbrot_functions import mandelbrot_naive, mandelbrot_numpy

# Run profile for Naive version
cProfile.run('mandelbrot_naive(-2, 1, -1.5, 1.5, 512, 512)', 'naive_profile.prof')

# Run profile for NumPy version
cProfile.run('mandelbrot_numpy(-2, 1, -1.5, 1.5, 512, 512)', 'numpy_profile.prof')

for name in ('naive_profile.prof', 'numpy_profile.prof'):
    print(f"\n--- Profile for {name} ---")
    stats = pstats.Stats(name)
    stats.sort_stats('cumulative') # Sort by total time taken
    stats.print_stats(10)          # Print the top 10 most expensive functions