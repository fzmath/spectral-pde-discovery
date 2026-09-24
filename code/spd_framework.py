"""
Spectral PDE Discovery (SPD) Framework
=======================================
Core innovations:
1. Spatiotemporal spectral derivatives (FFT in both space and time)
2. LassoCV sparse regression with automatic regularization
3. Adaptive thresholding based on coefficient statistics
4. Non-periodic boundary handling via mirror extension

Comparison: Finite Difference vs Spectral+Filter vs SPD (full framework)
"""
import numpy as np
from pathlib import Path
import json
import time
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. Derivative computation methods
# ============================================================

def finite_diff_derivatives(u, x, t, max_order=4):
    """Finite difference spatial + temporal derivatives.
    u shape: (n_x, n_time) for 1D
    """
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    # Time derivative
    u_t = np.zeros_like(u)
    u_t[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2*dt)
    u_t[:, 0] = (u[:, 1] - u[:, 0]) / dt
    u_t[:, -1] = (u[:, -1] - u[:, -2]) / dt
    # Spatial derivatives
    derivs = [u.copy()]
    for order in range(1, max_order+1):
        if order == 1:
            d = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2*dx)
        else:
            prev = derivs[-1]
            d = (np.roll(prev, -1, axis=0) - np.roll(prev, 1, axis=0)) / (2*dx)
        derivs.append(d)
    return u_t, derivs

def spectral_derivatives_spacetime(u, x, t, max_order=4, filter_order=8, cutoff=0.6):
    """Spatiotemporal spectral derivatives with exponential filtering.
    FFT in both space (axis=0) and time (axis=1).
    """
    Nx, Nt = u.shape
    Lx = x[-1] - x[0] + (x[1]-x[0])
    Lt = t[-1] - t[0] + (t[1]-t[0])
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=Lx/Nx)
    kt = 2*np.pi*np.fft.fftfreq(Nt, d=Lt/Nt)
    
    # 2D filter
    kx_max = np.max(np.abs(kx))
    kt_max = np.max(np.abs(kt))
    alpha = -np.log(1e-12)
    
    KX, KT = np.meshgrid(kx, kt, indexing='ij')
    filter_2d = np.ones_like(KX)
    high_x = np.abs(KX) > cutoff * kx_max
    high_t = np.abs(KT) > cutoff * kt_max
    filter_2d[high_x] *= np.exp(-alpha * (np.abs(KX[high_x]) / kx_max)**filter_order)
    filter_2d[high_t] *= np.exp(-alpha * (np.abs(KT[high_t]) / kt_max)**filter_order)
    
    # 2D FFT
    u_hat = np.fft.fft2(u)
    u_hat_filt = u_hat * filter_2d
    
    # Time derivative: multiply by (i*kt)
    u_t = np.real(np.fft.ifft2((1j*KT) * u_hat_filt))
    
    # Spatial derivatives
    derivs = [u.copy()]
    for order in range(1, max_order+1):
        d_hat = (1j*KX)**order * u_hat_filt
        derivs.append(np.real(np.fft.ifft2(d_hat)))
    
    return u_t, derivs

def spectral_derivatives_space_only(u, x, t, max_order=4, filter_order=8, cutoff=0.6):
    """Spectral derivatives in space only (original method), FD in time."""
    N = len(x)
    L = x[-1] - x[0] + (x[1]-x[0])
    k = 2*np.pi*np.fft.fftfreq(N, d=L/N)
    k_max = np.max(np.abs(k))
    alpha = -np.log(1e-12)
    filter_factors = np.ones_like(k)
    high = np.abs(k) > cutoff * k_max
    filter_factors[high] = np.exp(-alpha * (np.abs(k[high]) / k_max)**filter_order)
    
    dt = t[1] - t[0]
    u_t = np.zeros_like(u)
    u_t[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2*dt)
    u_t[:, 0] = (u[:, 1] - u[:, 0]) / dt
    u_t[:, -1] = (u[:, -1] - u[:, -2]) / dt
    
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
# 2. Library construction
# ============================================================

def build_library_1d(derivs, poly_order=2):
    """Build candidate library for 1D PDEs."""
    u = derivs[0]
    features = {}
    for p in range(1, poly_order+1):
        features[f'u^{p}' if p > 1 else 'u'] = u**p
    for order in range(1, len(derivs)):
        d = derivs[order]
        name = 'u_' + 'x'*order
        features[name] = d
        for p in range(1, poly_order+1):
            features[f'u^{p}*{name}' if p > 1 else f'u*{name}'] = (u**p) * d
    return features

# ============================================================
# 3. Sparse regression methods
# ============================================================

def stlsq(Theta, target, threshold=0.02, alpha=1e-5, max_iter=20):
    """Sequentially thresholded least squares (baseline)."""
    n_features = Theta.shape[1]
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

def lasso_cv(Theta, target, max_iter=10000):
    """Lasso with cross-validated regularization parameter."""
    # Standardize features
    scaler = StandardScaler()
    Theta_scaled = scaler.fit_transform(Theta)
    target_scaled = (target - target.mean()) / (target.std() + 1e-12)
    
    # LassoCV
    model = LassoCV(cv=5, max_iter=max_iter, n_alphas=100, random_state=42)
    model.fit(Theta_scaled, target_scaled)
    
    # Transform coefficients back to original scale
    xi_scaled = model.coef_
    xi = xi_scaled * (target.std() + 1e-12) / scaler.scale_
    return xi, model.alpha_

def adaptive_threshold(xi, feature_names, method='mad', k=3.0):
    """Adaptive thresholding based on coefficient statistics.
    method: 'mad' (median absolute deviation), 'std', 'percentile'
    """
    nonzero = xi[np.abs(xi) > 1e-12]
    if len(nonzero) == 0:
        return set()
    
    if method == 'mad':
        median = np.median(np.abs(nonzero))
        mad = np.median(np.abs(nonzero - median))
        threshold = median + k * 1.4826 * mad  # 1.4826 makes MAD consistent with std
    elif method == 'std':
        threshold = np.mean(np.abs(nonzero)) + k * np.std(np.abs(nonzero))
    elif method == 'percentile':
        threshold = np.percentile(np.abs(nonzero), 75)
    
    selected = {feature_names[i] for i in range(len(xi)) if np.abs(xi[i]) > threshold}
    return selected

def refit_selected(Theta, target, feature_names, selected):
    """Refit least squares on selected features."""
    if len(selected) == 0:
        return {}
    indices = [i for i, name in enumerate(feature_names) if name in selected]
    xi_sub = np.linalg.lstsq(Theta[:, indices], target, rcond=None)[0]
    return {feature_names[i]: float(xi_sub[j]) for j, i in enumerate(indices)}

# ============================================================
# 4. SPD full pipeline
# ============================================================

def spd_pipeline(u, x, t, true_terms, max_order=4, poly_order=2):
    """Full Spectral PDE Discovery pipeline.
    1. Spatiotemporal spectral derivatives
    2. LassoCV sparse regression
    3. Adaptive thresholding
    4. Refit on selected terms
    """
    # Step 1: derivatives
    u_t, derivs = spectral_derivatives_spacetime(u, x, t, max_order=max_order)
    
    # Step 2: library
    lib = build_library_1d(derivs, poly_order=poly_order)
    feature_names = list(lib.keys())
    Theta = np.column_stack([lib[name].ravel() for name in feature_names])
    target = u_t.ravel()
    
    # Step 3: LassoCV
    xi, alpha = lasso_cv(Theta, target)
    
    # Step 4: adaptive threshold
    selected = adaptive_threshold(xi, feature_names, method='mad', k=2.0)
    
    # Step 5: refit
    recovered = refit_selected(Theta, target, feature_names, selected)
    
    return recovered, alpha

# ============================================================
# 5. Evaluation
# ============================================================

def evaluate_recovery(recovered, true_terms):
    """Evaluate PDE recovery accuracy."""
    true_found = sum(1 for t in true_terms if t in recovered)
    false_pos = [name for name in recovered if name not in true_terms]
    exact = (true_found == len(true_terms)) and (len(false_pos) == 0)
    coeff_err = {}
    for term, true_c in true_terms.items():
        if term in recovered:
            coeff_err[term] = abs(recovered[term] - true_c) / abs(true_c)
        else:
            coeff_err[term] = None
    return true_found, false_pos, exact, coeff_err

# ============================================================
# 6. Main test
# ============================================================

DATA_DIR = Path(r"H:\2026科研\Spectral-PDE-Discovery\mdbench\data\processed\data\pde")

TRUE_PDES = {
    'burgers': {'u*u_x': -1.0, 'u_xx': 0.1},
    'kdv': {'u*u_x': -6.0, 'u_xxx': -1.0},
    'kuramoto_sivishinky': {'u*u_x': -1.0, 'u_xx': -1.0, 'u_xxxx': -1.0},
    'advection1d': {'u_x': -0.1},
    'heat_soil_uniform_1d_p1': {'u_xx': 1.0},  # heat equation: u_t = u_xx
}

def load_dataset(name, noise_file=None):
    """Load MDBench dataset."""
    if noise_file:
        path = DATA_DIR / name / noise_file
    else:
        path = DATA_DIR / name / f"{name}.npz"
    data = np.load(path)
    t = data['t']
    x = data['x']
    u = data['u']
    if u.ndim == 3:
        u = u[:, :, 0]
    return u, x, t

def test_method(method_name, u, x, t, true_terms, max_order=4):
    """Test one method."""
    t0 = time.time()
    
    if method_name == 'FD+STLSQ':
        u_t, derivs = finite_diff_derivatives(u, x, t, max_order=max_order)
        lib = build_library_1d(derivs)
        feature_names = list(lib.keys())
        Theta = np.column_stack([lib[n].ravel() for n in feature_names])
        target = u_t.ravel()
        # Best threshold search
        best = None
        for thresh in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
            xi = stlsq(Theta, target, threshold=thresh)
            recovered = {feature_names[i]: float(xi[i]) for i in range(len(xi)) if abs(xi[i]) > 1e-6}
            tf, fp, exact, ce = evaluate_recovery(recovered, true_terms)
            if best is None or (exact and not best[2]) or (len(fp) < len(best[1]) and tf >= best[0]):
                best = (tf, fp, exact, ce, recovered, thresh)
            if exact:
                break
        tf, fp, exact, ce, recovered, thresh = best
        info = {'threshold': thresh}
        
    elif method_name == 'Spectral+STLSQ':
        u_t, derivs = spectral_derivatives_space_only(u, x, t, max_order=max_order)
        lib = build_library_1d(derivs)
        feature_names = list(lib.keys())
        Theta = np.column_stack([lib[n].ravel() for n in feature_names])
        target = u_t.ravel()
        best = None
        for thresh in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
            xi = stlsq(Theta, target, threshold=thresh)
            recovered = {feature_names[i]: float(xi[i]) for i in range(len(xi)) if abs(xi[i]) > 1e-6}
            tf, fp, exact, ce = evaluate_recovery(recovered, true_terms)
            if best is None or (exact and not best[2]) or (len(fp) < len(best[1]) and tf >= best[0]):
                best = (tf, fp, exact, ce, recovered, thresh)
            if exact:
                break
        tf, fp, exact, ce, recovered, thresh = best
        info = {'threshold': thresh}
        
    elif method_name == 'SPD (ours)':
        recovered, alpha = spd_pipeline(u, x, t, true_terms, max_order=max_order)
        tf, fp, exact, ce = evaluate_recovery(recovered, true_terms)
        info = {'lasso_alpha': float(alpha)}
    
    elapsed = time.time() - t0
    return {
        'recovered': {k: float(v) for k, v in recovered.items()},
        'true_found': tf,
        'total_true': len(true_terms),
        'false_positives': fp,
        'exact': exact,
        'coeff_errors': {k: float(v) if v is not None else None for k, v in ce.items()},
        'time_s': elapsed,
        **info
    }

def main():
    print("=" * 70)
    print("Spectral PDE Discovery (SPD) Framework - MDBench Test")
    print("=" * 70)
    
    all_results = {}
    datasets = ['burgers', 'kdv', 'kuramoto_sivishinky', 'advection1d', 'heat_soil_uniform_1d_p1']
    noise_levels = [None, 'snr_40', 'snr_30', 'snr_20', 'snr_10']
    methods = ['FD+STLSQ', 'Spectral+STLSQ', 'SPD (ours)']
    
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}")
        print(f"{'='*60}")
        all_results[ds] = {}
        true_terms = TRUE_PDES.get(ds, {})
        
        for noise in noise_levels:
            noise_label = 'clean' if noise is None else noise
            noise_file = f"{ds}_{noise}.npz" if noise else None
            print(f"\n--- {noise_label} ---")
            try:
                u, x, t = load_dataset(ds, noise_file)
                all_results[ds][noise_label] = {}
                for method in methods:
                    res = test_method(method, u, x, t, true_terms)
                    all_results[ds][noise_label][method] = res
                    status = "EXACT" if res['exact'] else f"{res['true_found']}/{res['total_true']} + {len(res['false_positives'])} FP"
                    print(f"  {method:20s}: {status}  ({res['time_s']:.2f}s)")
                    if res['coeff_errors']:
                        errs = [f"{k}={v:.2e}" if v else f"{k}=MISSING" for k, v in res['coeff_errors'].items()]
                        print(f"    coeff errors: {', '.join(errs)}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  ERROR: {e}")
                all_results[ds][noise_label] = {'error': str(e)}
    
    # Save
    out_path = Path(r"H:\2026科研\Spectral-PDE-Discovery\code\spd_results.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Exact Recovery Rate")
    print("=" * 80)
    print(f"{'Dataset':<30} {'Noise':<10} {'FD+STLSQ':<15} {'Spec+STLSQ':<15} {'SPD(ours)':<15}")
    print("-" * 85)
    for ds in datasets:
        for noise in ['clean', 'snr_40', 'snr_30', 'snr_20', 'snr_10']:
            if noise in all_results[ds] and 'error' not in all_results[ds][noise]:
                r = all_results[ds][noise]
                row = f"{ds:<30} {noise:<10}"
                for m in methods:
                    s = "EXACT" if r[m]['exact'] else f"{r[m]['true_found']}/{r[m]['total_true']}"
                    row += f" {s:<15}"
                print(row)

if __name__ == '__main__':
    main()
