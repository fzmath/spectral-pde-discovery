"""
Concept validation: Spectral derivatives for PDE discovery (improving MDBench).
Compares spectral derivatives vs finite differences in STLSQ PDE recovery.
"""
import numpy as np
from scipy.integrate import solve_ivp
import json
import time

np.random.seed(42)

# ============================================================
# 1. Generate Burgers equation data
# u_t = -u*u_x + nu*u_xx, periodic BC
# ============================================================
def generate_burgers_data(N=256, L=2*np.pi, T=1.0, nt=100, nu=0.01):
    """Generate Burgers equation solution using Fourier pseudospectral method."""
    x = np.linspace(0, L, N, endpoint=False)
    k = 2*np.pi*np.fft.fftfreq(N, d=L/N)
    
    # Initial condition: smooth wave
    u0 = np.sin(x) + 0.5*np.cos(2*x)
    
    def rhs(t, u):
        u_hat = np.fft.fft(u)
        ux = np.real(np.fft.ifft(1j*k*u_hat))
        uxx = np.real(np.fft.ifft(-k**2*u_hat))
        return -u*ux + nu*uxx
    
    t_span = [0, T]
    t_eval = np.linspace(0, T, nt)
    sol = solve_ivp(rhs, t_span, u0, t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12)
    
    return x, sol.t, sol.y.T  # u[t, x]

print("Generating Burgers equation data...")
x, t, U = generate_burgers_data(N=256, T=1.0, nt=50)
print(f"  Data shape: U{t, x} = {U.shape}")
print(f"  x range: [{x[0]:.3f}, {x[-1]:.3f}], N={len(x)}")
print(f"  t range: [{t[0]:.3f}, {t[-1]:.3f}], nt={len(t)}")

# ============================================================
# 2. Derivative computation methods
# ============================================================
def finite_diff_derivatives(u, x, t):
    """Compute u_t, u_x, u_xx using 2nd-order central differences."""
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    nt, N = u.shape
    
    u_t = np.zeros_like(u)
    u_x = np.zeros_like(u)
    u_xx = np.zeros_like(u)
    
    # Time derivative (central, with forward/backward at edges)
    u_t[1:-1] = (u[2:] - u[:-2]) / (2*dt)
    u_t[0] = (u[1] - u[0]) / dt
    u_t[-1] = (u[-1] - u[-2]) / dt
    
    # Space derivatives (periodic central difference)
    u_x = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2*dx)
    u_xx = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dx**2
    
    return u_t, u_x, u_xx

def spectral_derivatives(u, x, t):
    """Compute u_t, u_x, u_xx using FFT spectral differentiation (DSEM on uniform grid)."""
    N = len(x)
    L = x[-1] - x[0] + (x[1]-x[0])
    k = 2*np.pi*np.fft.fftfreq(N, d=L/N)
    dt = t[1] - t[0]
    nt = u.shape[0]
    
    u_t = np.zeros_like(u)
    u_x = np.zeros_like(u)
    u_xx = np.zeros_like(u)
    
    # Time derivative (finite diff in time, spectral in space)
    u_t[1:-1] = (u[2:] - u[:-2]) / (2*dt)
    u_t[0] = (u[1] - u[0]) / dt
    u_t[-1] = (u[-1] - u[-2]) / dt
    
    # Space derivatives via FFT (spectral accuracy)
    for i in range(nt):
        u_hat = np.fft.fft(u[i])
        u_x[i] = np.real(np.fft.ifft(1j*k*u_hat))
        u_xx[i] = np.real(np.fft.ifft(-k**2*u_hat))
    
    return u_t, u_x, u_xx

# ============================================================
# 3. STLSQ (Sequentially Thresholded Least Squares)
# ============================================================
def build_library(u, u_x, u_xx, u_xxx=None):
    """Build candidate function library for PDE discovery."""
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
    if u_xxx is not None:
        features['u_xxx'] = u_xxx
        features['u*u_xxx'] = u * u_xxx
    return features

def stlsq(Theta, y, threshold=0.1, max_iter=20):
    """Sequentially Thresholded Least Squares (SINDy algorithm)."""
    # Initial least squares
    xi, _, _, _ = np.linalg.lstsq(Theta, y, rcond=None)
    
    for _ in range(max_iter):
        # Threshold: set small coefficients to zero
        small = np.abs(xi) < threshold
        xi[small] = 0
        
        # Re-solve with remaining terms
        big = ~small
        if np.sum(big) == 0:
            break
        xi[big], _, _, _ = np.linalg.lstsq(Theta[:, big], y, rcond=None)
    
    return xi

def compute_u_xxx(u, x, method='spectral'):
    """Compute third derivative."""
    N = len(x)
    L = x[-1] - x[0] + (x[1]-x[0])
    k = 2*np.pi*np.fft.fftfreq(N, d=L/N)
    u_xxx = np.zeros_like(u)
    for i in range(u.shape[0]):
        u_hat = np.fft.fft(u[i])
        u_xxx[i] = np.real(np.fft.ifft((1j*k)**3 * u_hat))
    return u_xxx

# ============================================================
# 4. Run experiments at different noise levels
# ============================================================
print("\n" + "="*60)
print("Running PDE discovery experiments")
print("="*60)

noise_levels = [0.0, 0.01, 0.05, 0.10]
results = {}

# True Burgers equation: u_t = -u*u_x + nu*u_xx
# True coefficients: u*u_x = -1.0, u_xx = 0.01
true_terms = {'u*u_x': -1.0, 'u_xx': 0.01}

for noise in noise_levels:
    print(f"\n--- Noise level: {noise*100:.0f}% ---")
    
    # Add noise
    if noise > 0:
        noise_std = noise * np.std(U)
        U_noisy = U + np.random.normal(0, noise_std, U.shape)
    else:
        U_noisy = U.copy()
    
    # Method 1: Finite differences
    t0 = time.time()
    u_t_fd, u_x_fd, u_xx_fd = finite_diff_derivatives(U_noisy, x, t)
    u_xxx_fd = compute_u_xxx(U_noisy, x, method='fd') if False else None
    lib_fd = build_library(U_noisy, u_x_fd, u_xx_fd)
    Theta_fd = np.column_stack([lib_fd[key].flatten() for key in lib_fd])
    y_fd = u_t_fd.flatten()
    xi_fd = stlsq(Theta_fd, y_fd, threshold=0.05)
    fd_time = time.time() - t0
    
    # Method 2: Spectral (DSEM)
    t0 = time.time()
    u_t_sp, u_x_sp, u_xx_sp = spectral_derivatives(U_noisy, x, t)
    lib_sp = build_library(U_noisy, u_x_sp, u_xx_sp)
    Theta_sp = np.column_stack([lib_sp[key].flatten() for key in lib_sp])
    y_sp = u_t_sp.flatten()
    xi_sp = stlsq(Theta_sp, y_sp, threshold=0.05)
    sp_time = time.time() - t0
    
    # Evaluate recovery
    term_names = list(lib_fd.keys())
    
    def evaluate_recovery(xi, term_names, true_terms):
        """Evaluate if true terms are recovered and false terms are excluded."""
        recovered = {}
        for i, name in enumerate(term_names):
            if abs(xi[i]) > 1e-6:
                recovered[name] = xi[i]
        
        # Check true terms
        true_found = 0
        coeff_errors = {}
        for term, true_coeff in true_terms.items():
            if term in recovered:
                true_found += 1
                coeff_errors[term] = abs(recovered[term] - true_coeff) / abs(true_coeff)
            else:
                coeff_errors[term] = None  # missed
        
        # Check false positives
        false_positives = [t for t in recovered if t not in true_terms]
        
        return {
            'recovered_terms': recovered,
            'true_found': true_found,
            'total_true': len(true_terms),
            'false_positives': false_positives,
            'coeff_errors': coeff_errors,
            'exact_recovery': true_found == len(true_terms) and len(false_positives) == 0
        }
    
    fd_eval = evaluate_recovery(xi_fd, term_names, true_terms)
    sp_eval = evaluate_recovery(xi_sp, term_names, true_terms)
    
    print(f"  Finite Difference:")
    print(f"    Recovered: {fd_eval['recovered_terms']}")
    print(f"    True found: {fd_eval['true_found']}/{fd_eval['total_true']}")
    print(f"    False positives: {fd_eval['false_positives']}")
    print(f"    Exact recovery: {fd_eval['exact_recovery']}")
    print(f"    Time: {fd_time*1000:.1f} ms")
    
    print(f"  Spectral (DSEM):")
    print(f"    Recovered: {sp_eval['recovered_terms']}")
    print(f"    True found: {sp_eval['true_found']}/{sp_eval['total_true']}")
    print(f"    False positives: {sp_eval['false_positives']}")
    print(f"    Exact recovery: {sp_eval['exact_recovery']}")
    print(f"    Time: {sp_time*1000:.1f} ms")
    
    # Derivative accuracy
    u_t_exact, u_x_exact, u_xx_exact = spectral_derivatives(U, x, t)  # noise-free spectral = ground truth
    fd_deriv_err = np.linalg.norm(u_x_fd - u_x_exact) / np.linalg.norm(u_x_exact)
    sp_deriv_err = np.linalg.norm(u_x_sp - u_x_exact) / np.linalg.norm(u_x_exact)
    print(f"  Derivative accuracy (u_x relative error):")
    print(f"    FD: {fd_deriv_err:.4e}")
    print(f"    Spectral: {sp_deriv_err:.4e}")
    
    results[f'noise_{noise}'] = {
        'noise_level': noise,
        'finite_difference': {
            'recovered_terms': {k: float(v) for k, v in fd_eval['recovered_terms'].items()},
            'true_found': fd_eval['true_found'],
            'false_positives': fd_eval['false_positives'],
            'exact_recovery': fd_eval['exact_recovery'],
            'coeff_errors': {k: float(v) if v is not None else None for k, v in fd_eval['coeff_errors'].items()},
            'deriv_error': float(fd_deriv_err),
            'time_ms': float(fd_time*1000)
        },
        'spectral': {
            'recovered_terms': {k: float(v) for k, v in sp_eval['recovered_terms'].items()},
            'true_found': sp_eval['true_found'],
            'false_positives': sp_eval['false_positives'],
            'exact_recovery': sp_eval['exact_recovery'],
            'coeff_errors': {k: float(v) if v is not None else None for k, v in sp_eval['coeff_errors'].items()},
            'deriv_error': float(sp_deriv_err),
            'time_ms': float(sp_time*1000)
        }
    }

# ============================================================
# 5. Summary
# ============================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Noise':>8} | {'FD exact':>10} | {'SP exact':>10} | {'FD deriv err':>14} | {'SP deriv err':>14}")
print("-"*65)
for noise in noise_levels:
    r = results[f'noise_{noise}']
    print(f"{noise*100:>7.0f}% | {str(r['finite_difference']['exact_recovery']):>10} | {str(r['spectral']['exact_recovery']):>10} | {r['finite_difference']['deriv_error']:>14.4e} | {r['spectral']['deriv_error']:>14.4e}")

with open('pde_discovery_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to pde_discovery_results.json")
