"""
Test spectral derivatives on MDBench standard PDE datasets.
Compare finite difference vs spectral+filter for PDE discovery.
"""
import numpy as np
from pathlib import Path
import json
import time

# ============================================================
# Derivative computation methods
# ============================================================

def finite_diff_derivatives_1d(u, x, t, max_order=4):
    """Finite difference spatial derivatives (MDBench baseline).
    u shape: (n_x, n_time)
    """
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    # Time derivative (along axis=1)
    u_t = np.zeros_like(u)
    u_t[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2*dt)
    u_t[:, 0] = (u[:, 1] - u[:, 0]) / dt
    u_t[:, -1] = (u[:, -1] - u[:, -2]) / dt
    # Spatial derivatives (along axis=0)
    derivs = [u.copy()]
    for order in range(1, max_order+1):
        if order == 1:
            d = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2*dx)
        else:
            prev = derivs[-1]
            d = (np.roll(prev, -1, axis=0) - np.roll(prev, 1, axis=0)) / (2*dx)
        derivs.append(d)
    return u_t, derivs

def spectral_derivatives_1d(u, x, t, max_order=4, filter_order=8, cutoff=0.6):
    """Spectral derivatives with exponential filtering.
    u shape: (n_x, n_time)
    """
    N = len(x)
    L = x[-1] - x[0] + (x[1]-x[0])
    k = 2*np.pi*np.fft.fftfreq(N, d=L/N)
    k_max = np.max(np.abs(k))
    alpha = -np.log(1e-12)
    filter_factors = np.ones_like(k)
    high = np.abs(k) > cutoff * k_max
    filter_factors[high] = np.exp(-alpha * (np.abs(k[high]) / k_max)**filter_order)
    
    dt = t[1] - t[0]
    # Time derivative (along axis=1)
    u_t = np.zeros_like(u)
    u_t[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2*dt)
    u_t[:, 0] = (u[:, 1] - u[:, 0]) / dt
    u_t[:, -1] = (u[:, -1] - u[:, -2]) / dt
    
    # Spatial derivatives via FFT (along axis=0)
    derivs = [u.copy()]
    for order in range(1, max_order+1):
        derivs.append(np.zeros_like(u))
    for nt in range(u.shape[1]):
        u_hat = np.fft.fft(u[:, nt])
        u_hat_filt = u_hat * filter_factors
        for order in range(1, max_order+1):
            d_hat = (1j*k)**order * u_hat_filt
            derivs[order][:, nt] = np.real(np.fft.ifft(d_hat))
    return u_t, derivs

# ============================================================
# Library construction and STLSQ
# ============================================================

def build_library_1d(derivs, poly_order=2):
    """Build candidate library: u, u^2, u^3, u_x, u*u_x, u^2*u_x, u_xx, ..."""
    u = derivs[0]
    features = {}
    # Polynomial terms
    for p in range(1, poly_order+1):
        features[f'u^{p}' if p > 1 else 'u'] = u**p
    # Derivative and mixed terms
    for order in range(1, len(derivs)):
        d = derivs[order]
        name = 'u_' + 'x'*order
        features[name] = d
        for p in range(1, poly_order+1):
            features[f'u^{p}*{name}' if p > 1 else f'u*{name}'] = (u**p) * d
    return features

def stlsq(Theta, target, threshold=0.02, alpha=1e-5, max_iter=20):
    """Sequentially thresholded least squares."""
    n_features = Theta.shape[1]
    # Initial ridge regression
    I = np.eye(n_features)
    xi = np.linalg.lstsq(Theta.T @ Theta + alpha*I, Theta.T @ target, rcond=None)[0]
    for _ in range(max_iter):
        small = np.abs(xi) < threshold
        xi[small] = 0
        big = ~small
        if np.sum(big) == 0:
            break
        xi[big] = np.linalg.lstsq(Theta[:, big], target, rcond=None)[0]
    return xi

# ============================================================
# Evaluation
# ============================================================

def evaluate_recovery(xi, feature_names, true_terms):
    """Evaluate PDE recovery accuracy."""
    recovered = {}
    for i, name in enumerate(feature_names):
        if abs(xi[i]) > 1e-6:
            recovered[name] = xi[i]
    true_found = sum(1 for t in true_terms if t in recovered)
    false_pos = [name for name in recovered if name not in true_terms]
    exact = (true_found == len(true_terms)) and (len(false_pos) == 0)
    coeff_err = {}
    for term, true_c in true_terms.items():
        if term in recovered:
            coeff_err[term] = abs(recovered[term] - true_c) / abs(true_c)
        else:
            coeff_err[term] = None
    return recovered, true_found, false_pos, exact, coeff_err

# ============================================================
# Main test on MDBench data
# ============================================================

DATA_DIR = Path(r"H:\2026科研\Spectral-PDE-Discovery\mdbench\data\processed\data\pde")

# True PDE terms for each dataset
TRUE_PDES = {
    'burgers': {'u*u_x': -1.0, 'u_xx': 0.1},
    'kdv': {'u*u_x': -6.0, 'u_xxx': -1.0},
    'kuramoto_sivishinky': {'u*u_x': -1.0, 'u_xx': -1.0, 'u_xxxx': -1.0},
    'advection1d': {'u_x': -0.1},
}

def test_dataset(name, noise_file=None):
    """Test one PDE dataset."""
    if noise_file:
        path = DATA_DIR / name / noise_file
    else:
        path = DATA_DIR / name / f"{name}.npz"
    
    data = np.load(path)
    t = data['t']
    x = data['x']
    u = data['u']  # shape (n_x, n_time, n_dim)
    if u.ndim == 3:
        u = u[:, :, 0]  # take first component
    
    true_terms = TRUE_PDES.get(name, {})
    max_order = 4
    
    results = {}
    for method_name, deriv_fn in [
        ('Finite Difference', finite_diff_derivatives_1d),
        ('Spectral+Filter', spectral_derivatives_1d),
    ]:
        t0 = time.time()
        u_t, derivs = deriv_fn(u, x, t, max_order=max_order)
        lib = build_library_1d(derivs, poly_order=2)
        feature_names = list(lib.keys())
        Theta = np.column_stack([lib[name].ravel() for name in feature_names])
        target = u_t.ravel()
        
        # Best threshold search
        best_exact = False
        best_result = None
        for thresh in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
            xi = stlsq(Theta, target, threshold=thresh)
            recovered, true_found, false_pos, exact, coeff_err = evaluate_recovery(
                xi, feature_names, true_terms)
            if exact and not best_exact:
                best_exact = True
                best_result = (recovered, true_found, false_pos, exact, coeff_err, thresh)
                break
            if best_result is None or (len(false_pos) < len(best_result[2]) and true_found >= best_result[1]):
                best_result = (recovered, true_found, false_pos, exact, coeff_err, thresh)
        
        elapsed = time.time() - t0
        recovered, true_found, false_pos, exact, coeff_err, thresh = best_result
        results[method_name] = {
            'recovered': {k: float(v) for k, v in recovered.items()},
            'true_found': true_found,
            'total_true': len(true_terms),
            'false_positives': false_pos,
            'exact': exact,
            'coeff_errors': {k: float(v) if v is not None else None for k, v in coeff_err.items()},
            'best_threshold': thresh,
            'time_s': elapsed,
        }
    return results

def main():
    print("=" * 70)
    print("MDBench Standard Dataset Test: Finite Diff vs Spectral+Filter")
    print("=" * 70)
    
    all_results = {}
    datasets = ['burgers', 'kdv', 'kuramoto_sivishinky', 'advection1d']
    noise_levels = [None, 'snr_40', 'snr_30', 'snr_20', 'snr_10']
    
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}")
        print(f"{'='*60}")
        all_results[ds] = {}
        
        for noise in noise_levels:
            noise_label = 'clean' if noise is None else noise
            noise_file = f"{ds}_{noise}.npz" if noise else None
            print(f"\n--- {noise_label} ---")
            try:
                res = test_dataset(ds, noise_file)
                all_results[ds][noise_label] = res
                for method, r in res.items():
                    status = "EXACT" if r['exact'] else f"{r['true_found']}/{r['total_true']} + {len(r['false_positives'])} FP"
                    print(f"  {method:20s}: {status}  (thresh={r['best_threshold']}, {r['time_s']:.2f}s)")
                    if r['coeff_errors']:
                        errs = [f"{k}={v:.2e}" if v else f"{k}=MISSING" for k, v in r['coeff_errors'].items()]
                        print(f"    coeff errors: {', '.join(errs)}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  ERROR: {e}")
                all_results[ds][noise_label] = {'error': str(e)}
    
    # Save results
    out_path = Path(r"H:\2026科研\Spectral-PDE-Discovery\code\mdbench_test_results.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Exact Recovery Rate")
    print("=" * 70)
    print(f"{'Dataset':<25} {'Noise':<10} {'Finite Diff':<15} {'Spectral+Filter':<15}")
    print("-" * 65)
    for ds in datasets:
        for noise in ['clean', 'snr_40', 'snr_30', 'snr_20', 'snr_10']:
            if noise in all_results[ds] and 'error' not in all_results[ds][noise]:
                fd = all_results[ds][noise]['Finite Difference']
                sp = all_results[ds][noise]['Spectral+Filter']
                fd_str = "EXACT" if fd['exact'] else f"{fd['true_found']}/{fd['total_true']}"
                sp_str = "EXACT" if sp['exact'] else f"{sp['true_found']}/{sp['total_true']}"
                print(f"{ds:<25} {noise:<10} {fd_str:<15} {sp_str:<15}")

if __name__ == '__main__':
    main()
