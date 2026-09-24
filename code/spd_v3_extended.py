"""
SPD v3 Extended: More PDEs + 2D support
Uses MDBench's exact time derivative (du field) for fair comparison.
"""
import numpy as np
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Derivative methods (spatial only; time derivative from du)
# ============================================================

def finite_diff_spatial(u, x, max_order=4):
    """Finite difference spatial derivatives. u shape: (n_x, n_time) or (nx, ny, nt)."""
    dx = x[1] - x[0]
    if u.ndim == 2:
        derivs = [u.copy()]
        for order in range(1, max_order+1):
            if order == 1:
                d = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2*dx)
            else:
                d = (np.roll(derivs[-1], -1, axis=0) - np.roll(derivs[-1], 1, axis=0)) / (2*dx)
            derivs.append(d)
        return derivs
    elif u.ndim == 3:
        # 2D: (nx, ny, nt)
        dy = None  # will be passed separately
        derivs = [u.copy()]
        # x derivatives
        for order in range(1, max_order+1):
            if order == 1:
                d = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2*dx)
            else:
                d = (np.roll(derivs[-1], -1, axis=0) - np.roll(derivs[-1], 1, axis=0)) / (2*dx)
            derivs.append(d)
        return derivs

def spectral_spatial(u, x, max_order=4, filter_order=8, cutoff=0.6):
    """Spectral spatial derivatives with filtering. u shape: (n_x, n_time)."""
    N = len(x)
    L = x[-1] - x[0] + (x[1]-x[0])
    k = 2*np.pi*np.fft.fftfreq(N, d=L/N)
    k_max = np.max(np.abs(k))
    alpha = -np.log(1e-12)
    filter_factors = np.ones_like(k)
    high = np.abs(k) > cutoff * k_max
    filter_factors[high] = np.exp(-alpha * (np.abs(k[high]) / k_max)**filter_order)
    
    derivs = [u.copy()]
    for order in range(1, max_order+1):
        derivs.append(np.zeros_like(u))
    if u.ndim == 2:
        for nt in range(u.shape[1]):
            u_hat = np.fft.fft(u[:, nt])
            u_hat_filt = u_hat * filter_factors
            for order in range(1, max_order+1):
                d_hat = (1j*k)**order * u_hat_filt
                derivs[order][:, nt] = np.real(np.fft.ifft(d_hat))
    return derivs

def spectral_spacetime(u, x, t, max_order=4, filter_order=8, cutoff=0.6):
    """Spatiotemporal spectral derivatives. u shape: (n_x, n_time)."""
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

# ============================================================
# 2D spectral derivatives
# ============================================================

def spectral_spatial_2d(u, x, y, max_order=2, filter_order=8, cutoff=0.6):
    """2D spectral spatial derivatives. u shape: (nx, ny, nt)."""
    Nx, Ny, Nt = u.shape
    Lx = x[-1] - x[0] + (x[1]-x[0])
    Ly = y[-1] - y[0] + (y[1]-y[0])
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=Lx/Nx)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=Ly/Ny)
    kx_max = np.max(np.abs(kx))
    ky_max = np.max(np.abs(ky))
    alpha = -np.log(1e-12)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    filter_2d = np.ones_like(KX)
    high_x = np.abs(KX) > cutoff * kx_max
    high_y = np.abs(KY) > cutoff * ky_max
    filter_2d[high_x] *= np.exp(-alpha * (np.abs(KX[high_x]) / kx_max)**filter_order)
    filter_2d[high_y] *= np.exp(-alpha * (np.abs(KY[high_y]) / ky_max)**filter_order)
    
    derivs = {'u': u.copy()}
    for nt in range(Nt):
        u_hat = np.fft.fft2(u[:, :, nt])
        u_hat_filt = u_hat * filter_2d
        # u_x
        if 'u_x' not in derivs:
            derivs['u_x'] = np.zeros_like(u)
            derivs['u_y'] = np.zeros_like(u)
            derivs['u_xx'] = np.zeros_like(u)
            derivs['u_yy'] = np.zeros_like(u)
        derivs['u_x'][:, :, nt] = np.real(np.fft.ifft2((1j*KX) * u_hat_filt))
        derivs['u_y'][:, :, nt] = np.real(np.fft.ifft2((1j*KY) * u_hat_filt))
        derivs['u_xx'][:, :, nt] = np.real(np.fft.ifft2((1j*KX)**2 * u_hat_filt))
        derivs['u_yy'][:, :, nt] = np.real(np.fft.ifft2((1j*KY)**2 * u_hat_filt))
    return derivs

def finite_diff_spatial_2d(u, x, y, max_order=2):
    """2D finite difference spatial derivatives. u shape: (nx, ny, nt)."""
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    derivs = {'u': u.copy()}
    derivs['u_x'] = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2*dx)
    derivs['u_y'] = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2*dy)
    derivs['u_xx'] = (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / dx**2
    derivs['u_yy'] = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dy**2
    return derivs

# ============================================================
# Library
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
    if normalize:
        norms = {}
        for name in features:
            norm = np.linalg.norm(features[name].ravel())
            norms[name] = norm if norm > 0 else 1.0
            features[name] = features[name] / norms[name]
        return features, norms
    return features, None

def build_library_2d(derivs, poly_order=1, normalize=True):
    """2D library: u, u_x, u_y, u_xx, u_yy, u*u_x, u*u_y, etc."""
    u = derivs['u']
    features = {}
    features['u'] = u
    features['u_x'] = derivs['u_x']
    features['u_y'] = derivs['u_y']
    features['u_xx'] = derivs['u_xx']
    features['u_yy'] = derivs['u_yy']
    features['u*u_x'] = u * derivs['u_x']
    features['u*u_y'] = u * derivs['u_y']
    if poly_order >= 2:
        features['u^2'] = u**2
        features['u^2*u_x'] = u**2 * derivs['u_x']
        features['u^2*u_y'] = u**2 * derivs['u_y']
    if normalize:
        norms = {}
        for name in features:
            norm = np.linalg.norm(features[name].ravel())
            norms[name] = norm if norm > 0 else 1.0
            features[name] = features[name] / norms[name]
        return features, norms
    return features, None

# ============================================================
# Sparse regression
# ============================================================

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

def compute_bic(Theta, target, selected_indices):
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
    return bic

def forward_stepwise_bic(Theta, target, max_terms=10):
    n_samples, n_features = Theta.shape
    selected = []
    remaining = list(range(n_features))
    best_bic = np.inf
    best_selected = []
    for step in range(min(max_terms, n_features)):
        best_candidate = None
        best_candidate_bic = np.inf
        for j in remaining:
            candidate = selected + [j]
            bic = compute_bic(Theta, target, candidate)
            if bic < best_candidate_bic:
                best_candidate_bic = bic
                best_candidate = j
        if best_candidate is None:
            break
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        if best_candidate_bic < best_bic - 1e-6:
            best_bic = best_candidate_bic
            best_selected = selected.copy()
        elif step > 0 and best_candidate_bic > best_bic + 0.05 * np.log(n_samples):
            break
    if len(best_selected) > 0:
        X = Theta[:, best_selected]
        beta = np.linalg.lstsq(X, target, rcond=None)[0]
    else:
        beta = np.array([])
    return best_selected, beta, best_bic

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
# Main test
# ============================================================

DATA_DIR = Path(r"H:\2026科研\Spectral-PDE-Discovery\mdbench\data\processed\data\pde")

# True PDEs: coefficient sign convention is u_t = sum(c_i * term_i)
TRUE_PDES = {
    'burgers': {'u*u_x': -1.0, 'u_xx': 0.1},
    'kdv': {'u*u_x': -6.0, 'u_xxx': -1.0},
    'kuramoto_sivishinky': {'u*u_x': -1.0, 'u_xx': -1.0, 'u_xxxx': -1.0},
    'advection1d': {'u_x': -0.1},
    'nls': {'u_xx': -1.0, 'u^3': -1.0},  # NLS: i u_t = -u_xx - |u|^2 u, for real component approx
}

TRUE_PDES_2D = {
    'advection_diffusion_2d': {'u_x': -1.0, 'u_y': -1.0, 'u_xx': 0.1, 'u_yy': 0.1},  # approximate
}

def load_1d(name, noise_file=None):
    if noise_file:
        path = DATA_DIR / name / noise_file
    else:
        path = DATA_DIR / name / f"{name}.npz"
    data = np.load(path)
    t = data['t']
    x = data['x']
    u = data['u']
    du = data['du'] if 'du' in data else None
    if u.ndim == 3:
        u = u[:, :, 0]
        if du is not None:
            du = du[:, :, 0]
    return u, du, x, t

def load_2d(name, noise_file=None):
    if noise_file:
        path = DATA_DIR / name / noise_file
    else:
        path = DATA_DIR / name / f"{name}.npz"
    data = np.load(path)
    t = data['t']
    x = data['x']
    y = data['y']
    u = data['u']
    du = data['du'] if 'du' in data else None
    if u.ndim == 4:
        u = u[:, :, :, 0]
        if du is not None:
            du = du[:, :, :, 0]
    return u, du, x, y, t

def test_1d_method(method, u, du, x, t, true_terms, max_order=4):
    """Test 1D PDE discovery."""
    t0 = time.time()
    # Use exact time derivative if available, else compute
    if du is not None:
        u_t = du
    else:
        dt = t[1] - t[0]
        u_t = np.zeros_like(u)
        u_t[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2*dt)
    
    if method == 'FD+STLSQ':
        derivs = finite_diff_spatial(u, x, max_order=max_order)
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
    
    elif method == 'Spectral+STLSQ':
        derivs = spectral_spatial(u, x, max_order=max_order)
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
    
    elif method == 'SPD v3 (ours)':
        # Spatiotemporal spectral for derivatives
        u_t_spec, derivs = spectral_spacetime(u, x, t, max_order=max_order)
        # Use exact du if available for target
        target = (du if du is not None else u_t_spec).ravel()
        lib, norms = build_library_1d(derivs, normalize=True)
        feature_names = list(lib.keys())
        Theta = np.column_stack([lib[n].ravel() for n in feature_names])
        selected, beta, bic = forward_stepwise_bic(Theta, target, max_terms=8)
        recovered = {}
        for i, idx in enumerate(selected):
            name = feature_names[idx]
            coeff = beta[i] / norms[name] if norms else beta[i]
            recovered[name] = float(coeff)
        tf, fp, exact, ce = evaluate_recovery(recovered, true_terms)
        info = {'n_terms': len(selected)}
    
    elapsed = time.time() - t0
    return {
        'recovered': {k: float(v) for k, v in recovered.items()},
        'true_found': tf, 'total_true': len(true_terms),
        'false_positives': fp, 'exact': exact,
        'coeff_errors': {k: float(v) if v is not None else None for k, v in ce.items()},
        'time_s': elapsed, **info
    }

def test_2d_method(method, u, du, x, y, t, true_terms):
    """Test 2D PDE discovery."""
    t0 = time.time()
    if du is not None:
        u_t = du
    else:
        dt = t[1] - t[0]
        u_t = np.zeros_like(u)
        u_t[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) / (2*dt)
    
    if method == 'FD+STLSQ':
        derivs = finite_diff_spatial_2d(u, x, y)
        lib, _ = build_library_2d(derivs, normalize=False)
    elif method == 'Spectral+STLSQ':
        derivs = spectral_spatial_2d(u, x, y)
        lib, _ = build_library_2d(derivs, normalize=False)
    elif method == 'SPD v3 (ours)':
        derivs = spectral_spatial_2d(u, x, y)
        lib, norms = build_library_2d(derivs, normalize=True)
    
    feature_names = list(lib.keys())
    Theta = np.column_stack([lib[n].ravel() for n in feature_names])
    target = u_t.ravel()
    
    if method in ['FD+STLSQ', 'Spectral+STLSQ']:
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
    else:
        selected, beta, bic = forward_stepwise_bic(Theta, target, max_terms=8)
        recovered = {}
        for i, idx in enumerate(selected):
            name = feature_names[idx]
            coeff = beta[i] / norms[name] if norms else beta[i]
            recovered[name] = float(coeff)
        tf, fp, exact, ce = evaluate_recovery(recovered, true_terms)
        info = {'n_terms': len(selected)}
    
    elapsed = time.time() - t0
    return {
        'recovered': {k: float(v) for k, v in recovered.items()},
        'true_found': tf, 'total_true': len(true_terms),
        'false_positives': fp, 'exact': exact,
        'coeff_errors': {k: float(v) if v is not None else None for k, v in ce.items()},
        'time_s': elapsed, **info
    }

def main():
    print("=" * 70)
    print("SPD v3 Extended: 1D + 2D PDEs")
    print("=" * 70)
    
    all_results = {'1d': {}, '2d': {}}
    methods = ['FD+STLSQ', 'Spectral+STLSQ', 'SPD v3 (ours)']
    
    # 1D PDEs
    datasets_1d = ['burgers', 'kdv', 'kuramoto_sivishinky', 'advection1d', 'nls']
    noise_levels = [None, 'snr_40', 'snr_30', 'snr_20', 'snr_10']
    
    for ds in datasets_1d:
        print(f"\n{'='*60}")
        print(f"1D Dataset: {ds}")
        all_results['1d'][ds] = {}
        true_terms = TRUE_PDES.get(ds, {})
        
        for noise in noise_levels:
            noise_label = 'clean' if noise is None else noise
            noise_file = f"{ds}_{noise}.npz" if noise else None
            print(f"\n--- {noise_label} ---")
            try:
                u, du, x, t = load_1d(ds, noise_file)
                all_results['1d'][ds][noise_label] = {}
                for method in methods:
                    res = test_1d_method(method, u, du, x, t, true_terms)
                    all_results['1d'][ds][noise_label][method] = res
                    status = "EXACT" if res['exact'] else f"{res['true_found']}/{res['total_true']} + {len(res['false_positives'])} FP"
                    print(f"  {method:20s}: {status}  ({res['time_s']:.2f}s)")
            except Exception as e:
                import traceback
                traceback.print_exc()
                all_results['1d'][ds][noise_label] = {'error': str(e)}
    
    # 2D PDEs
    datasets_2d = ['advection_diffusion_2d']
    for ds in datasets_2d:
        print(f"\n{'='*60}")
        print(f"2D Dataset: {ds}")
        all_results['2d'][ds] = {}
        true_terms = TRUE_PDES_2D.get(ds, {})
        
        for noise in [None, 'snr_40', 'snr_20']:
            noise_label = 'clean' if noise is None else noise
            noise_file = f"{ds}_{noise}.npz" if noise else None
            print(f"\n--- {noise_label} ---")
            try:
                u, du, x, y, t = load_2d(ds, noise_file)
                all_results['2d'][ds][noise_label] = {}
                for method in methods:
                    res = test_2d_method(method, u, du, x, y, t, true_terms)
                    all_results['2d'][ds][noise_label][method] = res
                    status = "EXACT" if res['exact'] else f"{res['true_found']}/{res['total_true']} + {len(res['false_positives'])} FP"
                    print(f"  {method:20s}: {status}  ({res['time_s']:.2f}s)")
            except Exception as e:
                import traceback
                traceback.print_exc()
                all_results['2d'][ds][noise_label] = {'error': str(e)}
    
    out_path = Path(r"H:\2026科研\Spectral-PDE-Discovery\code\spd_v3_extended_results.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()
