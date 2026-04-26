import matplotlib.pyplot as plt
import numpy as np


#  mp1,2,3, 
# N=1024, max_iter=1000
results = {
    "Naive Python": 2.261,
    "NumPy": 0.418,
    "Numba (f64)": 0.042,
    "Numba (f32)": 0.046,      
    "GPU (f32)": 0.0033,       
    "Multiprocessing": 0.550,   
    "Dask Local": 0.320,
    "Dask Cluster": 0.150      
}

def plot_benchmarks():
    sorted_items = sorted(results.items(), key=lambda x: x[1], reverse=True)
    names, times = zip(*sorted_items)

    colors = ['#ff7f0e' if 'Naive' in n else '#1f77b4' for n in names]
    colors = ['#2ca02c' if 'GPU' in n else c for n, c in zip(names, colors)]
    plt.figure(figsize=(12, 7))
    bars = plt.bar(names, times, color=colors, log=True)
    plt.ylabel("Execution Time (seconds, log-scale)", fontsize=12)
    plt.title("Mandelbrot Performance: Pipeline Evolution (N=1024)", fontsize=14, fontweight='bold')
    plt.xticks(rotation=30, ha="right")
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                 f'{height:.4f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig("benchmark_mp3_final.png", dpi=150)
    print("Final benchmark chart saved as 'benchmark_mp3_final.png'")

if __name__ == "__main__":
    plot_benchmarks()
    
    print("\n--- Analysis ---")
    gpu_speedup = results["Naive Python"] / results["GPU (f32)"]
    numba_speedup = results["Naive Python"] / results["Numba (f64)"]
    print(f"GPU f32 Speedup relative to Naive: {gpu_speedup:.1f}x")
    print(f"GPU f32 Speedup relative to Numba: {results['Numba (f64)'] / results['GPU (f32)']:.1f}x")
