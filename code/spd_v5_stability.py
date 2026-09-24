"""
SPD v5: Stability Selection + Forward-Backward BIC
Key improvement: subsample aggregation to reduce false positives.
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

def build_lib_1d(derivs, poly_order=2, normalize=True):
    u=derivs[0]; features={}
    for p in range(1,poly_order+1):
        features[f'u^{p}' if p>1 else 'u']=u**p
    for o in range(1,len(derivs)):
        name='u_'+'x'*o; features[name]=derivs[o]
        for p in range(1,poly_order+1):
            features[f'u^{p}*{name}' if p>1 else f'u*{name}']=u**p*derivs[o]
    if normalize:
        norms={k:max(np.linalg.norm(v.ravel()),1e-12) for k,v in features.items()}
        features={k:v/norms[k] for k,v in features.items()}
        return features,norms
    return features,None

def fwbic(Theta,target,max_terms=8,penalty=2.0):
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
    return final

def stability_selection(Theta, target, n_bootstrap=50, subsample_ratio=0.8, threshold=0.6, penalty=2.0, max_terms=8):
    """Stability selection: run FWBIC on many subsamples, keep frequently selected features."""
    n = len(target)
    P = Theta.shape[1]
    selection_count = np.zeros(P)
    n_sub = int(n * subsample_ratio)
    
    for b in range(n_bootstrap):
        idx = np.random.choice(n, size=n_sub, replace=False)
        sel = fwbic(Theta[idx], target[idx], max_terms=max_terms, penalty=penalty)
        for j in sel:
            selection_count[j] += 1
    
    # Keep features selected in > threshold fraction of subsamples
    stable_features = [j for j in range(P) if selection_count[j] / n_bootstrap >= threshold]
    
    # Final refit on full data with stable features
    if stable_features:
        beta = np.linalg.lstsq(Theta[:, stable_features], target, rcond=None)[0]
    else:
        beta = np.array([])
        stable_features = []
    
    return stable_features, beta, selection_count / n_bootstrap

def spd_v5(u, du, x, t, true_terms, max_order=4, n_bootstrap=50, threshold=0.6):
    t0 = time.time()
    u_t_spec, derivs = spectral_spacetime(u, x, t, max_order)
    target = (du if du is not None else u_t_spec).ravel()
    lib, norms = build_lib_1d(derivs, normalize=True)
    names = list(lib.keys())
    Theta = np.column_stack([lib[n].ravel() for n in names])
    
    sel, beta, freq = stability_selection(Theta, target, n_bootstrap=n_bootstrap, threshold=threshold)
    
    rec = {names[i]: float(beta[j] / norms[names[i]]) for j, i in enumerate(sel)}
    tf = sum(1 for tt in true_terms if tt in rec)
    fp = [n for n in rec if n not in true_terms]
    return {
        'recovered': rec, 'true_found': tf, 'total_true': len(true_terms),
        'false_positives': fp, 'exact': (tf == len(true_terms)) and (len(fp) == 0),
        'time_s': time.time() - t0,
        'selection_freq': {names[i]: float(freq[i]) for i in range(len(names)) if freq[i] > 0.01}
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
    print("SPD v5: Stability Selection (50 bootstrap, threshold=0.6)")
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
                r = spd_v5(u, du, x, t, tt)
                results[ds][label] = r
                status = "EXACT" if r['exact'] else f"{r['true_found']}/{r['total_true']}+{len(r['false_positives'])}FP"
                print(f"  {label:8s}: {status:14s} ({r['time_s']:.1f}s)  FP={r['false_positives']}")
            except Exception as e:
                print(f"  {label}: ERROR {e}")
                import traceback; traceback.print_exc()
    
    out = Path(r"H:\2026科研\Spectral-PDE-Discovery\code\spd_v5_results.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nSaved to {out}")
    
    # Compare with v4
    print("\n" + "=" * 80)
    print("COMPARISON: v4 (FWBIC) vs v5 (Stability Selection)")
    print("=" * 80)
    v4_path = Path(r"H:\2026科研\Spectral-PDE-Discovery\code\spd_v4_results.json")
    if v4_path.exists():
        with open(v4_path) as f:
            v4 = json.load(f)
        print(f"{'PDE':<20} {'Noise':<8} {'v4 (p=2)':<14} {'v5 (stab)':<14}")
        for ds in datasets:
            for noise in ['clean', 'snr_40', 'snr_30', 'snr_20', 'snr_10']:
                row = f"{ds:<20} {noise:<8}"
                if noise in v4.get('penalty_2.0', {}).get(ds, {}):
                    r4 = v4['penalty_2.0'][ds][noise]['SPD v4 (ours)']
                    row += f" {r4['true_found']}/{r4['total_true']}+{len(r4['false_positives']):<2d}FP  "
                else:
                    row += " N/A           "
                if noise in results.get(ds, {}):
                    r5 = results[ds][noise]
                    row += f" {r5['true_found']}/{r5['total_true']}+{len(r5['false_positives']):<2d}FP"
                else:
                    row += " N/A"
                print(row)

if __name__ == '__main__':
    main()
