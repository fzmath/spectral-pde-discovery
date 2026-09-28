"""
P2 experiments:
1. Train/test generalization: identify on first 70% time, test PDE residual on last 30%
2. Coefficient error: compare recovered coefficients vs true
"""
import numpy as np
import sys, json
sys.path.insert(0, r"H:\2026科研\Spectral-PDE-Discovery\code")
from spd_v6_weak import spectral_spacetime, build_lib_1d, generate_test_functions, weak_form_system, fwbic, TRUE_PDES, load_1d

def run_generalization(name, du=None, train_frac=0.7, penalty=2.5):
    """Split data: identify on first train_frac of time, test residual on held-out time."""
    u, du_full, x, t = load_1d(name, noise_file=None)
    if du is None:
        du = du_full
    
    Nt = len(t)
    n_train = int(train_frac * Nt)
    
    # Train on first 70% time
    u_train = u[:, :n_train]
    du_train = du[:, :n_train]
    t_train = t[:n_train]
    
    # Spectral derivatives on training data
    u_t_spec, derivs_train = spectral_spacetime(u_train, x, t_train, max_order=4, filter_order=8, cutoff=0.6)
    target_train = du_train if du_train is not None else u_t_spec
    
    # Build library on training data
    lib_train = build_lib_1d(derivs_train)
    
    # Weak form on training data
    tf_train = generate_test_functions(x, t_train, n_test=100, width_ratio=0.15)
    Theta, target, names = weak_form_system(target_train, lib_train, tf_train, x, t_train)
    
    # Normalize
    norms = np.maximum(np.linalg.norm(Theta, axis=0), 1e-12)
    Theta_norm = Theta / norms
    
    # BIC
    sel, beta = fwbic(Theta_norm, target, max_terms=8, penalty=penalty)
    
    rec = {}
    for j, i in enumerate(sel):
        rec[names[i]] = float(beta[j] / norms[i])
    
    true_terms = set(TRUE_PDES[name].keys())
    tf = sum(1 for tt in true_terms if tt in rec)
    fp = [n for n in rec if n not in true_terms]
    
    # Coefficient error
    xi_true = TRUE_PDES[name]
    coef_err = 0.0
    coef_den = 0.0
    for term, val_true in xi_true.items():
        val_rec = rec.get(term, 0.0)
        coef_err += (val_rec - val_true)**2
        coef_den += val_true**2
    coef_rel_err = np.sqrt(coef_err / max(coef_den, 1e-12))
    
    # Test PDE residual on held-out 30% time
    u_test = u[:, n_train:]
    du_test = du[:, n_train:]
    t_test = t[n_train:]
    
    if len(t_test) >= 5:
        u_t_test, derivs_test = spectral_spacetime(u_test, x, t_test, max_order=4, filter_order=8, cutoff=0.6)
        lib_test = build_lib_1d(derivs_test)
        
        # Predict u_t = sum of recovered terms
        u_t_pred = np.zeros_like(u_test)
        for term, coef in rec.items():
            if term in lib_test:
                u_t_pred += coef * lib_test[term]
        
        # Use exact du_test as ground truth
        u_t_true = du_test if du_test is not None else u_t_test
        residual_rel = np.linalg.norm(u_t_pred - u_t_true) / max(np.linalg.norm(u_t_true), 1e-12)
    else:
        residual_rel = float('nan')
    
    return {
        'dataset': name,
        'true_found': tf,
        'total_true': len(true_terms),
        'false_positives': fp,
        'n_fp': len(fp),
        'coefficient_rel_error': float(coef_rel_err),
        'test_residual_rel': float(residual_rel),
        'recovered_coefs': rec
    }

def main():
    datasets = ['burgers', 'kdv', 'kuramoto_sivishinky', 'advection1d']
    results = {}
    
    print("=== P2: Train/Test Generalization + Coefficient Error ===")
    for ds in datasets:
        r = run_generalization(ds)
        results[ds] = r
        print(f"\n{ds}:")
        print(f"  True terms: {r['true_found']}/{r['total_true']}, FP: {r['n_fp']}")
        print(f"  Coef rel error: {r['coefficient_rel_error']:.4f}")
        print(f"  Test residual rel: {r['test_residual_rel']:.4f}")
        print(f"  Recovered: {r['recovered_coefs']}")
    
    out = r"H:\2026科研\Spectral-PDE-Discovery\code\p2_results.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")

if __name__ == '__main__':
    main()
