"""
Spectral PDE Discovery (SPD) Framework v3
==========================================
Key improvements:
1. Feature normalization (each library term divided by its L2 norm)
2. Forward stepwise regression with BIC model selection
3. Spatiotemporal spectral derivatives
4. Coefficient refitting on selected terms
"""
import numpy as np
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Derivative methods
# ============================================================

def finite_diff_derivatives(u, x, t, max_order=4):
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    u_t = np.zeros_like(u)
    u_t[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2*dt)
    u_t[:, 0] = (u[:, 1] - u[:, 0]) / dt
    u_t[:, -1] = (u[:, -1] - u[:, -2]) / dt
    derivs = [u.copy()]
    for order in range(1, max_order+1):
        if order == 1:
            d = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2*dx)
        else:
            d = (np.roll(derivs[-1], -1, axis=0) - np.roll(derivs[-1], 1, axis=0)) / (2*dx)
        derivs.append(d)
    return u_t, derivs

def spectral_derivatives_spacetime(u, x, t, max_order=4, filter_order=8, cutoff=0.6):
    Nx, Nt = u.shape
    Lx = x[-1] - x[0] + (x[1]-x[0])
    Lt = t[-1] - t[0] + (t[1]-t[0])
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=Lx/Nx)
    kt = 2*np.pi*np.fft.fftfreq(Nt, d=Lt/Nt)
    kx_max = np.max(np.abs(kx))
    kt_max = np.max(np.abs(kt))
    alpha = -np.log(1e-12)
    KX, KT = np.meshgrid(kx, kt, indexing='ij')
    filter_2d = np.ones_like(KX)
    high_x = np.abs(KX) > cutoff * kx_max
    high_t = np.abs(KT) > cutoff * kt_max
    filter_2d[high_x] *= np.exp(-alpha * (np.abs(KX[high_x]) / kx_max)**filter_order)
    filter_2d[high_t] *= np.exp(-alpha * (np.abs(KT[high_t]) / kt_max)**filter_order)
    u_hat = np.fft.fft2(u)
    u_hat_filt = u_hat * filter_2d
    u_t = np.real(np.fft.ifft2((1j*KT) * u_hat_filt))
    derivs = [u.copy()]
    for order in range(1, max_order+1):
        d_hat = (1j*KX)**order * u_hat_filt
        derivs.append(np.real(np.fft.ifft2(d_hat)))
    return u_t, derivs

def spectral_derivatives_space_only(u, x, t, max_order=4, filter_order=8, cutoff=0.6):
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
# Library with normalization
# ============================================================

def build_library_1d(derivs, poly_order=2, normalize=True):
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
    
    # Normalize each feature to unit L2 norm
    if normalize:
        norms = {}
        for name in features:
            norm = np.linalg.norm(features[name].ravel())
            norms[name] = norm if norm > 0 else 1.0
            features[name] = features[name] / norms[name]
        return features, norms
    return features, None

# ============================================================
# Sparse regression: Forward Stepwise with BIC
# ============================================================

def compute_bic(Theta, target, selected_indices):
    """Compute BIC for a subset of features."""
    n = len(target)
    if len(selected_indices) == 0:
        residual = target
        k = 0
    else:
        X = Theta[:, selected_indices]
        beta = np.linalg.lstsq(X, target, rcond=None)[0]
        residual = target - X @ beta
        k = len(selected_indices)
    rss = np.sum(residual**2)
    if rss < 1e-15:
        rss = 1e-15
    bic = n * np.log(rss / n) + k * np.log(n)
    return bic, rss

def forward_stepwise_bic(Theta, target, max_terms=10, tol=1e-6):
    """Forward stepwise regression with BIC model selection.
    Returns selected feature indices and coefficients.
    """
    n_samples, n_features = Theta.shape
    selected = []
    remaining = list(range(n_features))
    best_bic = np.inf
    best_selected = []
    
    # Precompute correlations for efficiency
    target_norm = target / (np.linalg.norm(target) + 1e-12)
    
    for step in range(min(max_terms, n_features)):
        best_candidate = None
        best_candidate_bic = np.inf
        
        for j in remaining:
            candidate = selected + [j]
            bic, _ = compute_bic(Theta, target, candidate)
            if bic < best_candidate_bic:
                best_candidate_bic = bic
                best_candidate = j
        
        if best_candidate is None:
            break
        
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        
        if best_candidate_bic < best_bic - tol:
            best_bic = best_candidate_bic
            best_selected = selected.copy()
        # If BIC starts increasing, stop (but allow one more step for safety)
        elif step > 0 and best_candidate_bic > best_bic + 0.1 * np.log(n_samples):
            break
    
    # Refit on best selected features
    if len(best_selected) > 0:
        X = Theta[:, best_selected]
        beta = np.linalg.lstsq(X, target, rcond=None)[0]
    else:
        beta = np.array([])
    
    return best_selected, beta, best_bic

def stlsq(Theta, target, threshold=0.02, alpha=1e-5, max_iter=20):
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

# ============================================================
# SPD pipeline v3
# ============================================================

def spd_pipeline_v3(u, x, t, true_terms, max_order=4, poly_order=2):
    """SPD v3: spatiotemporal spectral + normalized library + forward stepwise BIC."""
    # Step 1: spatiotemporal spectral derivatives
    u_t, derivs = spectral_derivatives_spacetime(u, x, t, max_order=max_order)
    
    # Step 2: normalized library
    lib, norms = build_library_1d(derivs, poly_order=poly_order, normalize=True)
    feature_names = list(lib.keys())
    Theta = np.column_stack([lib[name].ravel() for name in feature_names])
    target = u_t.ravel()
    
    # Step 3: forward stepwise with BIC
    selected, beta, bic = forward_stepwise_bic(Theta, target, max_terms=8)
    
    # Step 4: recover coefficients in original scale
    recovered = {}
    for i, idx in enumerate(selected):
        name = feature_names[idx]
        # Undo normalization: coefficient in original scale = beta / norm
        coeff = beta[i] / norms[name] if norms else beta[i]
        recovered[name] = float(coeff)
    
    return recovered, len(selected), bic

# ============================================================
# Evaluation
# ============================================================

def evaluate_recovery(recovered, true_terms):
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
# Main
# ============================================================

DATA_DIR = Path(r"H:\2026科研\Spectral-PDE-Discovery\mdbench\data\processed\data\pde")

TRUE_PDES = {
    'burgers': {'u*u_x': -1.0, 'u_xx': 0.1},
    'kdv': {'u*u_x': -6.0, 'u_xxx': -1.0},
    'kuramoto_sivishinky': {'u*u_x': -1.0, 'u_xx': -1.0, 'u_xxxx': -1.0},
    'advection1d': {'u_x': -0.1},
}

def load_dataset(name, noise_file=None):
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
    t0 = time.time()
    
    if method_name == 'FD+STLSQ':
        u_t, derivs = finite_diff_derivatives(u, x, t, max_order=max_order)
        lib, _ = build_library_1d(derivs, normalize=False)
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
        
    elif method_name == 'Spectral+STLSQ':
        u_t, derivs = spectral_derivatives_space_only(u, x, t, max_order=max_order)
        lib, _ = build_library_1d(derivs, normalize=False)
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
        
    elif method_name == 'SPD v3 (ours)':
        recovered, n_terms, bic = spd_pipeline_v3(u, x, t, true_terms, max_order=max_order)
        tf, fp, exact, ce = evaluate_recovery(recovered, true_terms)
        info = {'n_terms': n_terms, 'bic': float(bic)}
    
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
    print("SPD v3 Framework - Forward Stepwise + BIC")
    print("=" * 70)
    
    all_results = {}
    datasets = ['burgers', 'kdv', 'kuramoto_sivishinky', 'advection1d']
    noise_levels = [None, 'snr_40', 'snr_30', 'snr_20', 'snr_10']
    methods = ['FD+STLSQ', 'Spectral+STLSQ', 'SPD v3 (ours)']
    
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}")
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
                all_results[ds][noise_label] = {'error': str(e)}
    
    out_path = Path(r"H:\2026科研\Spectral-PDE-Discovery\code\spd_v3_results.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nResults saved to {out_path}")
    
    print("\n" + "=" * 80)
    print("SUMMARY: Exact Recovery Rate")
    print("=" * 80)
    print(f"{'Dataset':<25} {'Noise':<10} {'FD+STLSQ':<15} {'Spec+STLSQ':<15} {'SPDv3(ours)':<15}")
    print("-" * 80)
    for ds in datasets:
        for noise in ['clean', 'snr_40', 'snr_30', 'snr_20', 'snr_10']:
            if noise in all_results[ds] and 'error' not in all_results[ds][noise]:
                r = all_results[ds][noise]
                row = f"{ds:<25} {noise:<10}"
                for m in methods:
                    s = "EXACT" if r[m]['exact'] else f"{r[m]['true_found']}/{r[m]['total_true']}"
                    row += f" {s:<15}"
                print(row)

if __name__ == '__main__':
    main()
