"""
P0 experiments:
1. Noisy u_t: compare exact du vs spectrally estimated u_t
2. Multi-seed: 20 runs per PDE x noise, report mean ± std
"""
import numpy as np
from pathlib import Path
import json, time, warnings, sys
warnings.filterwarnings('ignore')

# Import core functions from existing module
sys.path.insert(0, str(Path(__file__).parent))
from spd_v6_weak import spectral_spacetime, build_lib_1d, weak_form_system, fwbic, TRUE_PDES, load_1d, DATA_DIR

def generate_test_functions_seeded(x, t, n_test=100, width_ratio=0.15, seed=42):
    Nx, Nt = len(x), len(t)
    Lx = x[-1] - x[0] + (x[1]-x[0])
    Lt = t[-1] - t[0] + (t[1]-t[0])
    wx = width_ratio * Lx; wt = width_ratio * Lt
    rng = np.random.RandomState(seed)
    cx = rng.uniform(x[0], x[-1], n_test)
    ct = rng.uniform(t[0], t[-1], n_test)
    X, T = np.meshgrid(x, t, indexing='ij')
    tfuncs = np.zeros((n_test, Nx, Nt))
    for i in range(n_test):
        fx = np.maximum(0, 1 - np.abs(X - cx[i]) / wx)
        ft = np.maximum(0, 1 - np.abs(T - ct[i]) / wt)
        tfuncs[i] = fx * ft
    return tfuncs

def run_spd(u, x, t, true_terms, seed=42, use_exact_ut=True, du_exact=None,
            max_order=4, n_test=100, width_ratio=0.15, penalty=2.5, cutoff=0.6):
    u_t_spec, derivs = spectral_spacetime(u, x, t, max_order, cutoff=cutoff)
    u_t = du_exact if (use_exact_ut and du_exact is not None) else u_t_spec
    lib = build_lib_1d(derivs)
    tfuncs = generate_test_functions_seeded(x, t, n_test, width_ratio, seed)
    Theta, target, names = weak_form_system(u_t, lib, tfuncs, x, t)
    norms = np.maximum(np.linalg.norm(Theta, axis=0), 1e-12)
    Theta_n = Theta / norms
    sel, beta = fwbic(Theta_n, target, max_terms=8, penalty=penalty)
    rec = {names[i]: float(beta[j]/norms[i]) for j, i in enumerate(sel)}
    tf = sum(1 for tt in true_terms if tt in rec)
    fp = [n for n in rec if n not in true_terms]
    # Coefficient error
    true_coefs = true_terms
    common = set(rec.keys()) & set(true_coefs.keys())
    if common:
        coef_err = np.sqrt(np.mean([(rec[k]-true_coefs[k])**2 for k in common])) / \
                   (np.std(list(true_coefs.values())) + 1e-10)
    else:
        coef_err = 999.0
    # F1
    tp = tf; fp_count = len(fp); fn = len(true_terms) - tf
    f1 = 2*tp/(2*tp+fp_count+fn) if (2*tp+fp_count+fn) > 0 else 0.0
    return {
        'true_found': tf, 'total_true': len(true_terms),
        'n_fp': fp_count, 'f1': f1, 'coef_err': coef_err,
        'exact': (tf==len(true_terms)) and (fp_count==0)
    }

def main():
    datasets = ['burgers', 'kdv', 'kuramoto_sivishinky', 'advection1d']
    noises = [None, 'snr_40', 'snr_30', 'snr_20', 'snr_10']
    n_runs = 20
    seeds = list(range(n_runs))

    results = {}
    for ds in datasets:
        results[ds] = {}
        tt = TRUE_PDES[ds]
        for noise in noises:
            label = 'clean' if noise is None else noise
            nf = f"{ds}_{noise}.npz" if noise else None
            try:
                u, du, x, t = load_1d(ds, nf)
            except Exception as e:
                print(f"  {ds}/{label}: load error {e}")
                continue

            # Exact u_t mode
            exact_runs = []
            est_runs = []
            for seed in seeds:
                r_exact = run_spd(u, x, t, tt, seed=seed, use_exact_ut=True, du_exact=du)
                r_est = run_spd(u, x, t, tt, seed=seed, use_exact_ut=False, du_exact=du)
                exact_runs.append(r_exact)
                est_runs.append(r_est)

            def summarize(runs):
                tf = [r['true_found'] for r in runs]
                fp = [r['n_fp'] for r in runs]
                f1 = [r['f1'] for r in runs]
                exact_rate = np.mean([r['exact'] for r in runs])
                return {
                    'tf_mean': float(np.mean(tf)), 'tf_std': float(np.std(tf)),
                    'fp_mean': float(np.mean(fp)), 'fp_std': float(np.std(fp)),
                    'f1_mean': float(np.mean(f1)), 'f1_std': float(np.std(f1)),
                    'exact_rate': float(exact_rate),
                    'tf_values': tf, 'fp_values': fp
                }

            results[ds][label] = {
                'exact_ut': summarize(exact_runs),
                'estimated_ut': summarize(est_runs)
            }
            e = results[ds][label]['exact_ut']
            s = results[ds][label]['estimated_ut']
            print(f"{ds:25s} {label:8s} | exact_ut: TF={e['tf_mean']:.1f}±{e['tf_std']:.1f} FP={e['fp_mean']:.1f}±{e['fp_std']:.1f} F1={e['f1_mean']:.2f} Exact={e['exact_rate']:.0%} | est_ut: TF={s['tf_mean']:.1f}±{s['tf_std']:.1f} FP={s['fp_mean']:.1f}±{s['fp_std']:.1f} F1={s['f1_mean']:.2f}")

    out = Path(r"H:\2026科研\Spectral-PDE-Discovery\code\p0_experiments.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x,'item') else x)
    print(f"\nSaved to {out}")

if __name__ == '__main__':
    main()
