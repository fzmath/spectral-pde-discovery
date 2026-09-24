"""
SPD v6: Spectral Weak Formulation
Innovation: spectral derivatives + weak-form regression (integration against test functions)
Combines spectral accuracy with noise robustness of weak formulation.
"""
import numpy as np
from pathlib import Path
import json, time, warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r"H:\2026科研\Spectral-PDE-Discovery\mdbench\data\processed\data\pde")

def spectral_spacetime(u, x, t, max_order=4, filter_order=8, cutoff=0.6):
    Nx, Nt = u.shape
    Lx = x[-1]-x[0]+(x[1]-x[0]); Lt = t[-1]-t[0]+(t[1]-t[0])
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=Lx/Nx); kt = 2*np.pi*np.fft.fftfreq(Nt, d=Lt/Nt)
    kxmax=np.max(np.abs(kx)); ktmax=np.max(np.abs(kt)); alpha=-np.log(1e-12)
    KX,KT=np.meshgrid(kx,kt,indexing='ij')
    filt=np.ones_like(KX)
    filt[np.abs(KX)>cutoff*kxmax]*=np.exp(-alpha*(np.abs(KX[np.abs(KX)>cutoff*kxmax])/kxmax)**filter_order)
    filt[np.abs(KT)>cutoff*ktmax]*=np.exp(-alpha*(np.abs(KT[np.abs(KT)>cutoff*ktmax])/ktmax)**filter_order)
    uh=np.fft.fft2(u)*filt
    u_t=np.real(np.fft.ifft2((1j*KT)*uh))
    derivs=[u.copy()]
    for o in range(1,max_order+1):
        derivs.append(np.real(np.fft.ifft2((1j*KX)**o*uh)))
    return u_t, derivs

def build_lib_1d(derivs, poly_order=2):
    """Build library WITHOUT normalization (weak form handles scaling)."""
    u=derivs[0]; features={}
    for p in range(1,poly_order+1):
        features[f'u^{p}' if p>1 else 'u']=u**p
    for o in range(1,len(derivs)):
        name='u_'+'x'*o; features[name]=derivs[o]
        for p in range(1,poly_order+1):
            features[f'u^{p}*{name}' if p>1 else f'u*{name}']=u**p*derivs[o]
    return features

def generate_test_functions(x, t, n_test=200, width_ratio=0.1):
    """Generate localized hat test functions in space-time domain."""
    Nx, Nt = len(x), len(t)
    Lx = x[-1] - x[0] + (x[1]-x[0])
    Lt = t[-1] - t[0] + (t[1]-t[0])
    wx = width_ratio * Lx
    wt = width_ratio * Lt
    
    # Random centers
    rng = np.random.RandomState(42)
    centers_x = rng.uniform(x[0], x[-1], n_test)
    centers_t = rng.uniform(t[0], t[-1], n_test)
    
    # Precompute test function values on grid (hat functions)
    X, T = np.meshgrid(x, t, indexing='ij')  # (Nx, Nt)
    test_funcs = np.zeros((n_test, Nx, Nt))
    for i in range(n_test):
        dx = np.abs(X - centers_x[i])
        dt = np.abs(T - centers_t[i])
        # Product of 1D hat functions
        phi_x = np.maximum(0, 1 - dx / wx)
        phi_t = np.maximum(0, 1 - dt / wt)
        test_funcs[i] = phi_x * phi_t
    
    return test_funcs

def weak_form_system(u_t, lib, test_funcs, x, t):
    """Build weak-form linear system: ∫u_t φ_i = Σ ξ_j ∫lib_j φ_i."""
    n_test = len(test_funcs)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    dA = dx * dt  # area element for trapezoidal rule
    
    names = list(lib.keys())
    n_features = len(names)
    
    # Target: ∫ u_t * φ_i
    target = np.zeros(n_test)
    for i in range(n_test):
        target[i] = np.sum(u_t * test_funcs[i]) * dA
    
    # Design matrix: ∫ lib_j * φ_i
    Theta = np.zeros((n_test, n_features))
    for j, name in enumerate(names):
        for i in range(n_test):
            Theta[i, j] = np.sum(lib[name] * test_funcs[i]) * dA
    
    return Theta, target, names

def fwbic(Theta, target, max_terms=8, penalty=2.0):
    n=len(target); P=Theta.shape[1]; sel=[]; rem=list(range(P)); best_bic=np.inf; best_sel=[]
    for step in range(min(max_terms,P)):
        bj=None; bb=np.inf
        for j in rem:
            cand=sel+[j]; X=Theta[:,cand]; beta=np.linalg.lstsq(X,target,rcond=None)[0]
            rss=np.sum((target-X@beta)**2); bic=n*np.log(max(rss/n,1e-15))+penalty*len(cand)*np.log(n)
            if bic<bb: bb=bic; bj=j
        if bj is None or (bb>=best_bic and step>0): break
        sel.append(bj); rem.remove(bj)
        if bb<best_bic: best_bic=bb; best_sel=sel.copy()
    final=best_sel.copy(); improved=True
    while improved and len(final)>0:
        improved=False
        Xf=Theta[:,final]; bf=np.linalg.lstsq(Xf,target,rcond=None)[0]
        cb=n*np.log(max(np.sum((target-Xf@bf)**2)/n,1e-15))+penalty*len(final)*np.log(n)
        for j in final:
            trial=[x for x in final if x!=j]
            Xt=Theta[:,trial]; bt=np.linalg.lstsq(Xt,target,rcond=None)[0]
            tb=n*np.log(max(np.sum((target-Xt@bt)**2)/n,1e-15))+penalty*len(trial)*np.log(n)
            if tb<=cb: final=trial; improved=True; break
    beta=np.linalg.lstsq(Theta[:,final],target,rcond=None)[0] if final else np.array([])
    return final, beta

def spd_v6(u, du, x, t, true_terms, max_order=4, n_test=200, width_ratio=0.1, penalty=2.0, cutoff=0.6):
    """Spectral Weak Formulation: spectral derivatives + weak-form regression."""
    t0 = time.time()
    # Step 1: spectral derivatives
    u_t_spec, derivs = spectral_spacetime(u, x, t, max_order, cutoff=cutoff)
    u_t = du if du is not None else u_t_spec
    
    # Step 2: build library
    lib = build_lib_1d(derivs)
    
    # Step 3: generate test functions
    test_funcs = generate_test_functions(x, t, n_test=n_test, width_ratio=width_ratio)
    
    # Step 4: build weak-form system
    Theta, target, names = weak_form_system(u_t, lib, test_funcs, x, t)
    
    # Step 5: normalize features in weak form (L2 norm of integrated features)
    norms = np.maximum(np.linalg.norm(Theta, axis=0), 1e-12)
    Theta_norm = Theta / norms
    
    # Step 6: FWBIC
    sel, beta = fwbic(Theta_norm, target, max_terms=8, penalty=penalty)
    
    # Recover coefficients (undo normalization)
    rec = {}
    for j, i in enumerate(sel):
        rec[names[i]] = float(beta[j] / norms[i])
    
    tf = sum(1 for tt in true_terms if tt in rec)
    fp = [n for n in rec if n not in true_terms]
    return {
        'recovered': rec, 'true_found': tf, 'total_true': len(true_terms),
        'false_positives': fp, 'exact': (tf == len(true_terms)) and (len(fp) == 0),
        'time_s': time.time() - t0
    }

TRUE_PDES = {
    'burgers': {'u*u_x': -1.0, 'u_xx': 0.1},
    'kdv': {'u*u_x': -6.0, 'u_xxx': -1.0},
    'kuramoto_sivishinky': {'u*u_x': -1.0, 'u_xx': -1.0, 'u_xxxx': -1.0},
    'advection1d': {'u_x': -0.1},
}

def load_1d(name, noise_file=None):
    path = DATA_DIR / name / (noise_file if noise_file else f"{name}.npz")
    data = np.load(path); t = data['t']; x = data['x']; u = data['u']
    du = data['du'] if 'du' in data else None
    if u.ndim == 3: u = u[:, :, 0]; du = du[:, :, 0] if du is not None else None
    return u, du, x, t

def main():
    print("=" * 80)
    print("SPD v6: Spectral Weak Formulation (spectral derivatives + weak-form regression)")
    print("=" * 80)
    
    datasets = ['burgers', 'kdv', 'kuramoto_sivishinky', 'advection1d']
    noises = [None, 'snr_40', 'snr_30', 'snr_20', 'snr_10']
    results = {}
    
    for ds in datasets:
        print(f"\n--- {ds} ---")
        results[ds] = {}
        tt = TRUE_PDES[ds]
        for noise in noises:
            label = 'clean' if noise is None else noise
            nf = f"{ds}_{noise}.npz" if noise else None
            try:
                u, du, x, t = load_1d(ds, nf)
                r = spd_v6(u, du, x, t, tt)
                results[ds][label] = r
                status = "EXACT" if r['exact'] else f"{r['true_found']}/{r['total_true']}+{len(r['false_positives'])}FP"
                print(f"  {label:8s}: {status:14s} ({r['time_s']:.1f}s)  FP={r['false_positives']}")
            except Exception as e:
                print(f"  {label}: ERROR {e}")
                import traceback; traceback.print_exc()
    
    out = Path(r"H:\2026科研\Spectral-PDE-Discovery\code\spd_v6_weak_results.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nSaved to {out}")
    
    # Compare with v4 and Weak SINDy
    print("\n" + "=" * 80)
    print("COMPARISON: v4(point) vs v6(spectral weak) vs Weak SINDy")
    print("=" * 80)
    v4_path = Path(r"H:\2026科研\Spectral-PDE-Discovery\code\spd_v4_results.json")
    v4 = {}
    if v4_path.exists():
        with open(v4_path) as f:
            v4 = json.load(f)
    
    # Weak SINDy results from previous run (hardcoded key findings)
    weak_sindy = {
        'burgers': {'clean': 'EXACT', 'snr_40': 'EXACT', 'snr_30': '2/2+2FP', 'snr_20': '2/2+1FP', 'snr_10': '2/2+3FP'},
        'kdv': {'clean': 'EXACT', 'snr_40': '2/2+1FP', 'snr_30': '2/2+1FP', 'snr_20': 'EXACT', 'snr_10': '2/2+1FP'},
        'kuramoto_sivishinky': {'clean': 'EXACT', 'snr_40': 'EXACT', 'snr_30': 'EXACT', 'snr_20': 'EXACT', 'snr_10': 'EXACT'},
        'advection1d': {'clean': 'EXACT', 'snr_40': '1/1+2FP', 'snr_30': 'EXACT', 'snr_20': '1/1+2FP', 'snr_10': '?'},
    }
    
    print(f"{'PDE':<20} {'Noise':<8} {'v4(point)':<14} {'v6(weak)':<14} {'WeakSINDy':<14}")
    for ds in datasets:
        for noise in ['clean', 'snr_40', 'snr_30', 'snr_20', 'snr_10']:
            row = f"{ds:<20} {noise:<8}"
            # v4
            if noise in v4.get('penalty_2.0', {}).get(ds, {}):
                r4 = v4['penalty_2.0'][ds][noise]['SPD v4 (ours)']
                row += f" {r4['true_found']}/{r4['total_true']}+{len(r4['false_positives']):<2d}FP  "
            else:
                row += " N/A           "
            # v6
            if noise in results.get(ds, {}):
                r6 = results[ds][noise]
                row += f" {r6['true_found']}/{r6['total_true']}+{len(r6['false_positives']):<2d}FP  "
            else:
                row += " N/A           "
            # Weak SINDy
            row += f" {weak_sindy.get(ds, {}).get(noise, '?'):<14}"
            print(row)

if __name__ == '__main__':
    main()
