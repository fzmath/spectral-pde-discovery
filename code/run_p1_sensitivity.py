"""
P1 experiments:
1. Lambda sensitivity: lambda in {1, 1.5, 2, 2.5, 3, 4}
2. Gaussian width sensitivity: sigma/L in {0.05, 0.1, 0.15, 0.2, 0.3}
3. Spectrum visualization: 2D FFT heatmaps
"""
import numpy as np
from pathlib import Path
import json, warnings, sys
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
from spd_v6_weak import spectral_spacetime, build_lib_1d, weak_form_system, fwbic, TRUE_PDES, load_1d, DATA_DIR

def generate_test_functions_seeded(x, t, n_test=100, width_ratio=0.15, seed=42):
    Nx, Nt = len(x), len(t)
    Lx = x[-1]-x[0]+(x[1]-x[0]); Lt = t[-1]-t[0]+(t[1]-t[0])
    wx = width_ratio*Lx; wt = width_ratio*Lt
    rng = np.random.RandomState(seed)
    cx = rng.uniform(x[0], x[-1], n_test)
    ct = rng.uniform(t[0], t[-1], n_test)
    X, T = np.meshgrid(x, t, indexing='ij')
    tfuncs = np.zeros((n_test, Nx, Nt))
    for i in range(n_test):
        fx = np.maximum(0, 1-np.abs(X-cx[i])/wx)
        ft = np.maximum(0, 1-np.abs(T-ct[i])/wt)
        tfuncs[i] = fx*ft
    return tfuncs

def run_one(u, du, x, t, true_terms, penalty=2.5, width_ratio=0.15, seed=42, use_exact=True):
    u_t_spec, derivs = spectral_spacetime(u, x, t, max_order=4, cutoff=0.6)
    u_t = du if (use_exact and du is not None) else u_t_spec
    lib = build_lib_1d(derivs)
    tf = generate_test_functions_seeded(x, t, 100, width_ratio, seed)
    Theta, target, names = weak_form_system(u_t, lib, tf, x, t)
    norms = np.maximum(np.linalg.norm(Theta, axis=0), 1e-12)
    sel, beta = fwbic(Theta/norms, target, max_terms=8, penalty=penalty)
    rec = {names[i]: float(beta[j]/norms[i]) for j, i in enumerate(sel)}
    tf_found = sum(1 for tt in true_terms if tt in rec)
    fp = len([n for n in rec if n not in true_terms])
    tp = tf_found; fn = len(true_terms)-tf_found
    f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0.0
    return {'tf': tf_found, 'fp': fp, 'f1': f1}

def main():
    # Use Burgers snr30 as sensitivity test case
    ds = 'burgers'; noise = 'snr_30'
    nf = f"{ds}_{noise}.npz"
    u, du, x, t = load_1d(ds, nf)
    tt = TRUE_PDES[ds]

    # === Lambda sensitivity ===
    lambdas = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    print("=== Lambda sensitivity (Burgers snr30, 5 seeds) ===")
    lam_results = {}
    for lam in lambdas:
        runs = [run_one(u, du, x, t, tt, penalty=lam, seed=s) for s in range(5)]
        f1s = [r['f1'] for r in runs]; fps = [r['fp'] for r in runs]
        tfs = [r['tf'] for r in runs]
        lam_results[str(lam)] = {'f1_mean': float(np.mean(f1s)), 'fp_mean': float(np.mean(fps)), 'tf_mean': float(np.mean(tfs))}
        print(f"  lambda={lam}: F1={np.mean(f1s):.2f} TF={np.mean(tfs):.1f} FP={np.mean(fps):.1f}")

    # === Width sensitivity ===
    widths = [0.05, 0.1, 0.15, 0.2, 0.3]
    print("\n=== Gaussian width sensitivity (Burgers snr30, 5 seeds) ===")
    w_results = {}
    for w in widths:
        runs = [run_one(u, du, x, t, tt, width_ratio=w, seed=s) for s in range(5)]
        f1s = [r['f1'] for r in runs]; fps = [r['fp'] for r in runs]
        tfs = [r['tf'] for r in runs]
        w_results[str(w)] = {'f1_mean': float(np.mean(f1s)), 'fp_mean': float(np.mean(fps)), 'tf_mean': float(np.mean(tfs))}
        print(f"  width={w}: F1={np.mean(f1s):.2f} TF={np.mean(tfs):.1f} FP={np.mean(fps):.1f}")

    # === Spectrum visualization data ===
    print("\n=== Computing spectrum visualization ===")
    # Advection clean: spatial-only vs spatiotemporal spectrum
    adv_u, adv_du, adv_x, adv_t = load_1d('advection1d', None)
    Nx, Nt = adv_u.shape
    Lx = adv_x[-1]-adv_x[0]+(adv_x[1]-adv_x[0])
    Lt = adv_t[-1]-adv_t[0]+(adv_t[1]-adv_t[0])

    # Spatiotemporal spectrum
    KX, KT = np.meshgrid(np.fft.fftfreq(Nx, d=Lx/Nx)*2*np.pi,
                         np.fft.fftfreq(Nt, d=Lt/Nt)*2*np.pi, indexing='ij')
    spec_st = np.fft.fftshift(np.abs(np.fft.fft2(adv_u)))

    # Spatial-only spectrum (average over time)
    spec_sp = np.fft.fftshift(np.abs(np.fft.fft(adv_u, axis=0)), axes=0)
    spec_sp_avg = np.mean(spec_sp, axis=1)

    # Filtered spatiotemporal spectrum
    kxmax = np.max(np.abs(np.fft.fftfreq(Nx, d=Lx/Nx)*2*np.pi))
    ktmax = np.max(np.abs(np.fft.fftfreq(Nt, d=Lt/Nt)*2*np.pi))
    alpha = -np.log(1e-12); cutoff = 0.6; p = 8
    filt = np.ones_like(KX)
    mask_x = np.abs(KX) > cutoff*kxmax
    filt[mask_x] *= np.exp(-alpha*(np.abs(KX[mask_x])/kxmax)**p)
    mask_t = np.abs(KT) > cutoff*ktmax
    filt[mask_t] *= np.exp(-alpha*(np.abs(KT[mask_t])/ktmax)**p)
    spec_filt = spec_st * filt

    np.savez(Path(r"H:\2026科研\Spectral-PDE-Discovery\code\spectrum_data.npz"),
             spec_st=spec_st, spec_sp_avg=spec_sp_avg, spec_filt=spec_filt,
             KX=KX, KT=KT, adv_x=adv_x, adv_t=adv_t)
    print("Spectrum data saved")

    out = Path(r"H:\2026科研\Spectral-PDE-Discovery\code\p1_sensitivity.json")
    with open(out, 'w') as f:
        json.dump({'lambda': lam_results, 'width': w_results}, f, indent=2)
    print(f"Saved to {out}")

if __name__ == '__main__':
    main()
