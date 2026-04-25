import numpy as np
import matplotlib.pyplot as plt
import os

N = 512
MAX_ITER = 1000
TAU = 0.01

X_RANGE = [-0.7530, -0.7490]
Y_RANGE = [0.0990, 0.1030]

def compute_divergence():
    print(f"Computing trajectory divergence (N={N}, MAX_ITER={MAX_ITER}, TAU={TAU})")
    
    x = np.linspace(X_RANGE[0], X_RANGE[1], N)
    y = np.linspace(Y_RANGE[0], Y_RANGE[1], N)
    X, Y = np.meshgrid(x, y)
    C64 = (X + 1j * Y).astype(np.complex128)
    C32 = C64.astype(np.complex64)

    z32 = np.zeros_like(C32)
    z64 = np.zeros_like(C64)
    diverge_iter = np.full((N, N), MAX_ITER, dtype=np.int32)
    active_div = np.ones((N, N), dtype=bool)

    z_esc = np.zeros_like(C64)
    escape_count = np.full((N, N), MAX_ITER, dtype=np.int32)
    active_esc = np.ones((N, N), dtype=bool)

    for k in range(MAX_ITER):
        # Update Escape Count
        if active_esc.any():
            z_esc[active_esc] = z_esc[active_esc]**2 + C64[active_esc]
            escaped = active_esc & (np.abs(z_esc) > 2)
            escape_count[escaped] = k
            active_esc[escaped] = False

        # Update Trajectory Divergence
        if active_div.any():
            z32[active_div] = z32[active_div]**2 + C32[active_div]
            z64[active_div] = z64[active_div]**2 + C64[active_div]
            
            # Difference metrics
            diff = (np.abs(z32.real.astype(np.float64) - z64.real)
                    + np.abs(z32.imag.astype(np.float64) - z64.imag))
            
            newly_diverged = active_div & (diff > TAU)
            diverge_iter[newly_diverged] = k
            active_div[newly_diverged] = False
        
        if not active_esc.any() and not active_div.any():
            break

    return diverge_iter, escape_count

def make_observations(diverge_iter, escape_count):
    diverged_mask = diverge_iter < MAX_ITER
    fraction_diverged = np.mean(diverged_mask)
    
    print("\n--- Observations ---")
    print(f"Fraction of pixels diverging before max_iter: {fraction_diverged:.2%}")
    
    correlation = np.corrcoef(diverge_iter.flatten(), escape_count.flatten())[0, 1]
    print(f"Correlation between divergence iter and escape count: {correlation:.4f}")
    
    return fraction_diverged

def plot_results(diverge_iter, escape_count):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    im1 = axes[0].imshow(diverge_iter, cmap='plasma', origin='lower',
                         extent=[X_RANGE[0], X_RANGE[1], Y_RANGE[0], Y_RANGE[1]])
    axes[0].set_title(f'Trajectory Divergence (tau={TAU})')
    fig.colorbar(im1, ax=axes[0], label='Divergence Iteration')
    
    im2 = axes[1].imshow(escape_count, cmap='magma', origin='lower',
                         extent=[X_RANGE[0], X_RANGE[1], Y_RANGE[0], Y_RANGE[1]])
    axes[1].set_title('Standard Escape Count (float64)')
    fig.colorbar(im2, ax=axes[1], label='Escape Iteration')
    
    plt.tight_layout()
    plt.savefig('mp3_m1_divergence_comparison.png', dpi=150)
    print("\nSaved comparison plot to mp3_m1_divergence_comparison.png")
    plt.show()

if __name__ == "__main__":
    diverge_data, escape_data = compute_divergence()
    make_observations(diverge_data, escape_data)
    plot_results(diverge_data, escape_data)
