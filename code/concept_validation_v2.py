"""
Improved concept validation: Spectral derivatives + filtering for PDE discovery.
Key improvement: spectral filtering to suppress noise amplification.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
import json
import time

np.random.seed(42)

# ============================================================
# 1. Generate Burgers equation data (nu=0.1 for clearer u_xx term)
# ============================================================
def generate_burgers_data(N=256, L=2*np.pi, T=0.5, nt=50, nu=0.1):
    x = np.linspace(0, L, N, endpoint=False)
    k = 2*np.pi*np.fft.fftfreq(N, d=L/N)
    u0 = np.sin(x) + 0.5*np.cos(2*x)
    
    def rhs(t, u):
        u_hat = np.fft.fft(u)
        ux = np.real(np.fft.ifft(1j*k*u_hat))
        uxx = np.real(np.fft.ifft(-k**2*u_hat))
        return -u*ux + nu*uxx
    
    t_eval = np.linspace(0, T, nt)
    sol = solve_ivp(rhs, [0, T], u0, t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12)
    return x, sol.t, sol.y.T

print("Generating Burgers equation data (nu=0.1)...")
x, t, U = generate_burgers_data(N=256, T=0.5, nt=50, nu=0.1)
print(f"  Data shape: {U.shape}")

# ============================================================
# 2. Derivative methods
# ============================================================
def finite_diff_derivatives(u, x, time_arr):
    dx = x[1] - x[0]
    dt = time_arr[1] - time_arr[0]
    u_t = np.zeros_like(u)
    u_t[1:-1] = (u[2:] - u[:-2]) / (2*dt)
    u_t[0] = (u[1] - u[0]) / dt
    u_t[-1] = (u[-1] - u[-2]) / dt
    u_x = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2*dx)
    u_xx = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dx**2
    return u_t, u_x, u_xx

def spectral_derivatives_filtered(u, x, time_arr, filter_order=8, cutoff=0.7):
    """Spectral derivatives with exponential filtering to suppress noise."""
    N = len(x)
    L = x[-1] - x[0] + (x[1]-x[0])
    k = 2*np.pi*np.fft.fftfreq(N, d=L/N)
    k_max = np.max(np.abs(k))
    alpha = -np.log(1e-12)
    filter_factors = np.ones_like(k)
    high = np.abs(k) > cutoff * k_max
    filter_factors[high] = np.exp(-alpha * (np.abs(k[high]) / k_max)**filter_order)
    
    dt = time_arr[1] - time_arr[0]
    u_t = np.zeros_like(u)
    u_t[1:-1] = (u[2:] - u[:-2]) / (2*dt)
    u_t[0] = (u[1] - u[0]) / dt
    u_t[-1] = (u[-1] - u[-2]) / dt
    
    u_x = np.zeros_like(u)
    u_xx = np.zeros_like(u)
    for i in range(u.shape[0]):
        u_hat = np.fft.fft(u[i])
        u_hat_filt = u_hat * filter_factors
        u_x[i] = np.real(np.fft.ifft(1j*k*u_hat_filt))
        u_xx[i] = np.real(np.fft.ifft(-k**2*u_hat_filt))
    return u_t, u_x, u_xx

def savgol_derivatives(u, x, time_arr, window=11, order=3):
    """Savitzky-Golay filtered derivatives (MDBench-style preprocessing)."""
    dx = x[1] - x[0]
    dt = time_arr[1] - time_arr[0]
    u_t = np.zeros_like(u)
    u_t[1:-1] = (u[2:] - u[:-2]) / (2*dt)
    u_t[0] = (u[1] - u[0]) / dt
    u_t[-1] = (u[-1] - u[-2]) / dt
    
    # Smooth then differentiate
    u_smooth = savgol_filter(u, window, order, axis=1, mode='wrap')
    u_x = (np.roll(u_smooth, -1, axis=1) - np.roll(u_smooth, 1, axis=1)) / (2*dx)
    u_xx = (np.roll(u_smooth, -1, axis=1) - 2*u_smooth + np.roll(u_smooth, 1, axis=1)) / dx**2
    return u_t, u_x, u_xx

# ============================================================
# 3. STLSQ with adaptive threshold
# ============================================================
def build_library(u, u_x, u_xx):
    features = {}
    features['1'] = np.ones_like(u)
    features['u'] = u
    features['u^2'] = u**2
    features['u^3'] = u**3
    features['u_x'] = u_x
    features['u*u_x'] = u * u_x
    features['u^2*u_x'] = u**2 * u_x
    features['u_xx'] = u_xx
    features['u*u_xx'] = u * u_xx
    return features

def stlsq(Theta, y, threshold=0.02, max_iter=20):
    xi, _, _, _ = np.linalg.lstsq(Theta, y, rcond=None)
    for _ in range(max_iter):
        small = np.abs(xi) < threshold
        xi[small] = 0
        big = ~small
        if np.sum(big) == 0:
            break
        xi[big], _, _, _ = np.linalg.lstsq(Theta[:, big], y, rcond=None)
    return xi

# ============================================================
# 4. Run experiments
# ============================================================
print("\n" + "="*70)
print("PDE Discovery: Spectral+Filter vs Finite Diff vs Savitzky-Golay")
print("="*70)

noise_levels = [0.0, 0.01, 0.05, 0.10]
true_terms = {'u*u_x': -1.0, 'u_xx': 0.1}
results = {}

for noise in noise_levels:
    print(f"\n--- Noise: {noise*100:.0f}% ---")
    
    if noise > 0:
        noise_std = noise * np.std(U)
        U_noisy = U + np.random.normal(0, noise_std, U.shape)
    else:
        U_noisy = U.copy()
    
    methods = {
        'Finite Difference': ('fd', {}),
        'Savitzky-Golay': ('sg', {'window': 11, 'order': 3}),
        'Spectral+Filter (DSEM)': ('sp', {'filter_order': 8, 'cutoff': 0.6}),
    }
    
    method_results = {}
    for name, (method_type, kwargs) in methods.items():
        t0 = time.time()
        if method_type == 'fd':
            u_t, u_x, u_xx = finite_diff_derivatives(U_noisy, x, t)
        elif method_type == 'sg':
            u_t, u_x, u_xx = savgol_derivatives(U_noisy, x, t, **kwargs)
        else:
            u_t, u_x, u_xx = spectral_derivatives_filtered(U_noisy, x, t, **kwargs)
        lib = build_library(U_noisy, u_x, u_xx)
        Theta = np.column_stack([lib[key].flatten() for key in lib])
        y = u_t.flatten()
        xi = stlsq(Theta, y, threshold=0.02)
        elapsed = time.time() - t0
        
        term_names = list(lib.keys())
        recovered = {term_names[i]: float(xi[i]) for i in range(len(xi)) if abs(xi[i]) > 1e-6}
        true_found = sum(1 for t in true_terms if t in recovered)
        false_pos = [t for t in recovered if t not in true_terms]
        exact = (true_found == len(true_terms)) and (len(false_pos) == 0)
        
        # Coefficient errors
        coeff_err = {}
        for term, true_c in true_terms.items():
            if term in recovered:
                coeff_err[term] = abs(recovered[term] - true_c) / abs(true_c)
            else:
                coeff_err[term] = None
        
        method_results[name] = {
            'recovered': recovered,
            'true_found': true_found,
            'false_positives': false_pos,
            'exact_recovery': exact,
            'coeff_errors': coeff_err,
            'time_ms': elapsed*1000
        }
        
        print(f"  {name}:")
        print(f"    Recovered: {recovered}")
        print(f"    True: {true_found}/2, False+: {false_pos}, Exact: {exact}")
    
    results[f'noise_{noise}'] = method_results

# ============================================================
# 5. Summary table
# ============================================================
print("\n" + "="*70)
print("SUMMARY: Exact PDE Recovery Rate")
print("="*70)
print(f"{'Noise':>8} | {'Finite Diff':>14} | {'Savitzky-Golay':>16} | {'Spectral+Filter':>18}")
print("-"*70)
for noise in noise_levels:
    r = results[f'noise_{noise}']
    fd = '✓' if r['Finite Difference']['exact_recovery'] else '✗'
    sg = '✓' if r['Savitzky-Golay']['exact_recovery'] else '✗'
    sp = '✓' if r['Spectral+Filter (DSEM)']['exact_recovery'] else '✗'
    print(f"{noise*100:>7.0f}% | {fd:>14} | {sg:>16} | {sp:>18}")

print("\nTrue terms recovered (out of 2):")
print(f"{'Noise':>8} | {'Finite Diff':>14} | {'Savitzky-Golay':>16} | {'Spectral+Filter':>18}")
print("-"*70)
for noise in noise_levels:
    r = results[f'noise_{noise}']
    print(f"{noise*100:>7.0f}% | {r['Finite Difference']['true_found']:>14} | {r['Savitzky-Golay']['true_found']:>16} | {r['Spectral+Filter (DSEM)']['true_found']:>18}")

with open('pde_discovery_results_v2.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to pde_discovery_results_v2.json")
