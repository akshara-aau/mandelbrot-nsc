import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Parameters
N, MAX_ITER = 512, 1000
X_RANGE = [-0.7530, -0.7490]
Y_RANGE = [0.0990, 0.1030]

def escape_count(C, max_iter):
    """Computes the escape count for a grid of complex numbers."""
    z = np.zeros_like(C)
    cnt = np.full(C.shape, max_iter, dtype=np.int32)
    esc = np.zeros(C.shape, dtype=bool)
    
    for k in range(max_iter):
        # Only update points that haven't escaped yet
        mask = ~esc
        if not np.any(mask):
            break
            
        z[mask] = z[mask]**2 + C[mask]
        newly_escaped = mask & (np.abs(z) > 2.0)
        cnt[newly_escaped] = k
        esc[newly_escaped] = True
        
    return cnt

def run_sensitivity_analysis():
    print(f"Running Sensitivity Analysis (N={N}, MAX_ITER={MAX_ITER})")
    
    x = np.linspace(X_RANGE[0], X_RANGE[1], N)
    y = np.linspace(Y_RANGE[0], Y_RANGE[1], N)
    C = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)
    
    eps32 = float(np.finfo(np.float32).eps)
    delta = np.maximum(eps32 * np.abs(C), 1e-10)
    
    print("Computing base escape counts")
    n_base = escape_count(C, MAX_ITER).astype(float)
    
    print("Computing perturbed escape counts")
    n_perturb = escape_count(C + delta, MAX_ITER).astype(float)
    
    dn = np.abs(n_base - n_perturb)
    kappa = np.where(n_base > 0, dn / (eps32 * n_base), np.nan)
    
    print("\n--- Observations ---")
    max_kappa = np.nanmax(kappa)
    print(f"Maximum detected condition number kappa: {max_kappa:.2e}")
    
    interior_mask = n_base == MAX_ITER
    kappa_interior = np.nanmean(kappa[interior_mask]) if np.any(interior_mask) else 0
    print(f"Average kappa for interior pixels (n=MAX_ITER): {kappa_interior:.2f}")

    cmap_k = plt.cm.hot.copy()
    cmap_k.set_bad('0.25') # Dark grey for NaNs
    
    vmax = np.nanpercentile(kappa, 99.5)
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(kappa, cmap=cmap_k, origin='lower',
                    extent=[X_RANGE[0], X_RANGE[1], Y_RANGE[0], Y_RANGE[1]],
                    norm=LogNorm(vmin=1, vmax=vmax))
    
    plt.colorbar(im, label=r'$\kappa(c)$ (log scale, $\kappa \geq 1$)')
    plt.title(r'Mandelbrot Sensitivity Map: $\kappa(c) \approx \frac{|\Delta n|}{\epsilon_{32} \cdot n(c)}$')
    plt.xlabel('Re(c)')
    plt.ylabel('Im(c)')
    
    output_file = 'mp3_m2_sensitivity_map.png'
    plt.savefig(output_file, dpi=150)
    print(f"\nSaved sensitivity map to {output_file}")
    plt.show()

if __name__ == "__main__":
    run_sensitivity_analysis()
