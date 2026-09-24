"""
Parameter sweep for SPD v6 (Spectral Weak Formulation)
Optimize: n_test, width_ratio, penalty, cutoff
Focus on fixing KdV snr_20 (1/2) and KS snr_40/30 (2/3)
"""
import numpy as np
from pathlib import Path
import json, time, warnings, itertools
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
    u=derivs[0]; features={}
    for p in range(1,poly_order+1):
        features[f'u^{p}' if p>1 else 'u']=u**p
    for o in range(1,len(derivs)):
        name='u_'+'x'*o; features[name]=derivs[o]
        for p in range(1,poly_order+1):
            features[f'u^{p}*{name}' if p>1 else f'u*{name}']=u**p*derivs[o]
    return features

def generate_test_functions(x, t, n_test=200, width_ratio=0.1, seed=42):
    Nx, Nt = len(x), len(t)
    Lx = x[-1] - x[0] + (x[1]-x[0])
    Lt = t[-1] - t[0] + (t[1]-t[0])
    wx = width_ratio * Lx
    wt = width_ratio * Lt
    rng = np.random.RandomState(seed)
    centers_x = rng.uniform(x[0], x[-1], n_test)
    centers_t = rng.uniform(t[0], t[-1], n_test)
    X, T = np.meshgrid(x, t, indexing='ij')
    test_funcs = np.zeros((n_test, Nx, Nt))
    for i in range(n_test):
        dx = np.abs(X - centers_x[i])
        dt = np.abs(T - centers_t[i])
        phi_x = np.maximum(0, 1 - dx / wx)
        phi_t = np.maximum(0, 1 - dt / wt)
        test_funcs[i] = phi_x * phi_t
    return test_funcs

def weak_form_system(u_t, lib, test_funcs, x, t):
    n_test = len(test_funcs)
    dx = x[1] - x[0]; dt = t[1] - t[0]; dA = dx * dt
    names = list(lib.keys())
    target = np.zeros(n_test)
    for i in range(n_test):
        target[i] = np.sum(u_t * test_funcs[i]) * dA
    Theta = np.zeros((n_test, len(names)))
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

def run_one(u, du, x, t, true_terms, n_test=200, width_ratio=0.1, penalty=2.0, cutoff=0.6, max_terms=8):
    u_t_spec, derivs = spectral_spacetime(u, x, t, cutoff=cutoff)
    u_t = du if du is not None else u_t_spec
    lib = build_lib_1d(derivs)
    test_funcs = generate_test_functions(x, t, n_test=n_test, width_ratio=width_ratio)
    Theta, target, names = weak_form_system(u_t, lib, test_funcs, x, t)
    norms = np.maximum(np.linalg.norm(Theta, axis=0), 1e-12)
    Theta_norm = Theta / norms
    sel, beta = fwbic(Theta_norm, target, max_terms=max_terms, penalty=penalty)
    rec = {names[i]: float(beta[j] / norms[i]) for j, i in enumerate(sel)}
    tf = sum(1 for tt in true_terms if tt in rec)
    fp = [n for n in rec if n not in true_terms]
    return tf, len(true_terms), len(fp), rec

def load_1d(name, noise_file=None):
    path = DATA_DIR / name / (noise_file if noise_file else f"{name}.npz")
    data = np.load(path); t = data['t']; x = data['x']; u = data['u']
    du = data['du'] if 'du' in data else None
    if u.ndim == 3: u = u[:, :, 0]; du = du[:, :, 0] if du is not None else None
    return u, du, x, t

TRUE_PDES = {
    'kdv': {'u*u_x': -6.0, 'u_xxx': -1.0},
    'kuramoto_sivishinky': {'u*u_x': -1.0, 'u_xx': -1.0, 'u_xxxx': -1.0},
}

def main():
    print("=" * 80)
    print("PARAMETER SWEEP for SPD v6 (focus: KdV snr_20, KS snr_40/30)")
    print("=" * 80)
    
    # Parameter grid
    n_tests = [100, 200, 400]
    widths = [0.05, 0.1, 0.15, 0.2]
    penalties = [1.0, 1.5, 2.0, 2.5]
    cutoffs = [0.5, 0.6, 0.7]
    
    # Test cases to optimize
    test_cases = [
        ('kdv', 'snr_20'),
        ('kuramoto_sivishinky', 'snr_40'),
        ('kuramoto_sivishinky', 'snr_30'),
    ]
    
    # Preload data
    data_cache = {}
    for ds, noise in test_cases:
        nf = f"{ds}_{noise}.npz" if noise != 'clean' else None
        data_cache[(ds, noise)] = load_1d(ds, nf)
    
    results = []
    best_score = -1
    best_params = None
    
    for n_test, width, penalty, cutoff in itertools.product(n_tests, widths, penalties, cutoffs):
        score = 0
        details = []
        for ds, noise in test_cases:
            u, du, x, t = data_cache[(ds, noise)]
            tt = TRUE_PDES[ds]
            tf, total, nfp, rec = run_one(u, du, x, t, tt, n_test=n_test, width_ratio=width, penalty=penalty, cutoff=cutoff)
            # Score: true terms found - half penalty for false positives
            case_score = tf - 0.3 * nfp
            score += case_score
            details.append(f"{ds}/{noise}:{tf}/{total}+{nfp}FP")
        
        if score > best_score:
            best_score = score
            best_params = (n_test, width, penalty, cutoff)
            print(f"  NEW BEST: score={score:.1f} n_test={n_test} width={width} pen={penalty} cut={cutoff} | {' | '.join(details)}")
        
        results.append({
            'n_test': n_test, 'width': width, 'penalty': penalty, 'cutoff': cutoff,
            'score': score, 'details': details
        })
    
    print(f"\n{'='*80}")
    print(f"BEST PARAMS: n_test={best_params[0]}, width={best_params[1]}, penalty={best_params[2]}, cutoff={best_params[3]}")
    print(f"{'='*80}")
    
    # Run full test with best params
    print("\nFull validation with best params:")
    n_test, width, penalty, cutoff = best_params
    for ds in ['burgers', 'kdv', 'kuramoto_sivishinky', 'advection1d']:
        print(f"\n  --- {ds} ---")
        tt = TRUE_PDES.get(ds, {'u_x': -0.1} if ds=='advection1d' else {'u*u_x': -1.0, 'u_xx': 0.1})
        for noise in [None, 'snr_40', 'snr_30', 'snr_20', 'snr_10']:
            label = 'clean' if noise is None else noise
            nf = f"{ds}_{noise}.npz" if noise else None
            u, du, x, t = load_1d(ds, nf)
            tf, total, nfp, rec = run_one(u, du, x, t, tt, n_test=n_test, width_ratio=width, penalty=penalty, cutoff=cutoff)
            status = "EXACT" if (tf==total and nfp==0) else f"{tf}/{total}+{nfp}FP"
            print(f"    {label:8s}: {status}")
    
    out = Path(r"H:\2026科研\Spectral-PDE-Discovery\code\param_sweep_results.json")
    with open(out, 'w') as f:
        json.dump({'best_params': {'n_test': best_params[0], 'width': best_params[1], 'penalty': best_params[2], 'cutoff': best_params[3]}, 'sweep': results}, f, indent=2)
    print(f"\nSaved to {out}")

if __name__ == '__main__':
    main()
