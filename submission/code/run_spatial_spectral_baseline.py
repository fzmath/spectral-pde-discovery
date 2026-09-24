"""
Run spatial-only spectral baseline (Schaeffer 2017 style) for comparison
Uses: spatial-only spectral derivatives + feature normalization + forward stepwise BIC
Compare against: spatiotemporal spectral + weak form + normalization + BIC (ours)
"""
import numpy as np
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r"H:\2026科研\Spectral-PDE-Discovery\mdbench\data\processed\data\pde")

# PDE configurations
PDES = {
    'burgers': {'folder': 'burgers', 'true_terms': ['u*u_x', 'u_xx'], 'max_order': 2},
    'kdv': {'folder': 'kdv', 'true_terms': ['u*u_x', 'u_xxx'], 'max_order': 3},
    'kuramoto_sivishinky': {'folder': 'kuramoto_sivishinky', 'true_terms': ['u*u_x', 'u_xx', 'u_xxxx'], 'max_order': 4},
    'advection1d': {'folder': 'advection1d', 'true_terms': ['u_x'], 'max_order': 1},
}

NOISE_LEVELS = ['clean', 'snr_40', 'snr_30', 'snr_20', 'snr_10']

def load_data(pde_name, noise):
    folder = PDES[pde_name]['folder']
    if noise == 'clean':
        fname = f"{folder}.npz"
    else:
        fname = f"{folder}_{noise}.npz"
    fpath = DATA_DIR / folder / fname
    data = np.load(fpath)
    u = data['u'].squeeze()  # (Nx, Nt)
    x = data['x'].squeeze()
    t = data['t'].squeeze()
    ut = data['du'].squeeze()  # exact time derivative
    return u, x, t, ut

def spectral_derivatives_space_only(u, x, max_order=4, filter_order=8, cutoff=0.6):
    """Spatial-only spectral derivatives (Schaeffer 2017 style), at each time step independently."""
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
    for nt in range(u.shape[1]):
        u_hat = np.fft.fft(u[:, nt])
        u_hat_filt = u_hat * filter_factors
        for order in range(1, max_order+1):
            d_hat = (1j*k)**order * u_hat_filt
            derivs[order][:, nt] = np.real(np.fft.ifft(d_hat))
    return derivs

def build_library(derivs, max_order):
    """Build candidate library from derivatives."""
    u = derivs[0]
    features = {}
    features['1'] = np.ones_like(u)
    features['u'] = u
    features['u^2'] = u**2
    for o in range(1, max_order+1):
        name = 'u' + 'x'*o
        features[name] = derivs[o]
    # nonlinear terms
    if max_order >= 1:
        features['u*u_x'] = u * derivs[1]
        features['u^2*u_x'] = u**2 * derivs[1]
    if max_order >= 2:
        features['u*u_xx'] = u * derivs[2]
        features['u^2*u_xx'] = u**2 * derivs[2]
    return features

def forward_stepwise_bic(Theta, target, max_terms=8, lam=2.5):
    """Forward stepwise regression with BIC."""
    N, P = Theta.shape
    # Normalize features
    norms = np.linalg.norm(Theta, axis=0)
    norms[norms == 0] = 1.0
    Theta_norm = Theta / norms
    
    selected = []
    remaining = list(range(P))
    best_bic = np.inf
    best_support = []
    
    for k in range(max_terms):
        if not remaining:
            break
        bics = []
        for j in remaining:
            support = selected + [j]
            X = Theta_norm[:, support]
            coeff, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
            resid = target - X @ coeff
            rss = np.sum(resid**2)
            bic = N * np.log(rss/N + 1e-30) + lam * len(support) * np.log(N)
            bics.append((bic, j))
        best = min(bics)
        selected.append(best[1])
        remaining.remove(best[1])
        if best[0] < best_bic - 0.05*np.log(N):
            best_bic = best[0]
            best_support = selected.copy()
        elif k > 0 and best[0] > best_bic + 0.05*np.log(N):
            break
    
    # Refit on best support
    if best_support:
        X = Theta_norm[:, best_support]
        coeff, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
        coeff = coeff / norms[best_support]
    else:
        coeff = np.array([])
    
    return best_support, coeff, norms

def evaluate(support, coeff, feature_names, true_terms, threshold=0.01):
    """Evaluate recovery."""
    found = []
    for idx, c in zip(support, coeff):
        name = feature_names[idx]
        if abs(c) > threshold:
            found.append(name)
    true_found = sum(1 for t in true_terms if t in found)
    false_pos = sum(1 for f in found if f not in true_terms)
    return true_found, len(true_terms), false_pos, found

def run_experiment(pde_name, noise):
    cfg = PDES[pde_name]
    u, x, t, ut = load_data(pde_name, noise)
    max_order = cfg['max_order']
    
    # Spatial-only spectral derivatives
    derivs = spectral_derivatives_space_only(u, x, max_order=max_order)
    
    # Build library
    features = build_library(derivs, max_order)
    feature_names = list(features.keys())
    Theta = np.column_stack([features[n].flatten() for n in feature_names])
    target = ut.flatten()
    
    # Forward stepwise BIC
    t0 = time.time()
    support, coeff, norms = forward_stepwise_bic(Theta, target, max_terms=8, lam=2.5)
    elapsed = time.time() - t0
    
    # Evaluate
    true_found, total_true, false_pos, found = evaluate(
        support, coeff, feature_names, cfg['true_terms'])
    
    return {
        'true_found': true_found,
        'total_true': total_true,
        'false_positives': false_pos,
        'found_terms': found,
        'time': elapsed
    }

# Run all experiments
results = {}
for pde_name in PDES:
    results[pde_name] = {}
    for noise in NOISE_LEVELS:
        try:
            r = run_experiment(pde_name, noise)
            results[pde_name][noise] = r
            print(f"{pde_name:25s} {noise:8s}: {r['true_found']}/{r['total_true']} + {r['false_positives']}  ({r['time']:.2f}s)")
        except Exception as e:
            print(f"{pde_name:25s} {noise:8s}: ERROR - {e}")
            results[pde_name][noise] = {'error': str(e)}

# Save results
outpath = Path(r"H:\2026科研\Spectral-PDE-Discovery\code\spatial_spectral_baseline.json")
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {outpath}")
