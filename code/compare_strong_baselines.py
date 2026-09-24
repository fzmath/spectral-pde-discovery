"""
Comprehensive comparison: SPD vs Weak SINDy vs SR3(SINDy-PI) vs FD+STLSQ
Fixed: pysindy feature name mapping, correct input format
"""
import numpy as np
from pathlib import Path
import json, time, warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r"H:\2026科研\Spectral-PDE-Discovery\mdbench\data\processed\data\pde")

def pysindy_name_to_std(name):
    """Convert pysindy feature name to standard name."""
    # x0_1 -> u_x, x0_11 -> u_xx, x0_111 -> u_xxx, x0_1111 -> u_xxxx
    if name.startswith('x0_'):
        order = len(name) - 3  # 'x0_' is 3 chars
        return 'u_' + 'x' * order
    # ux0_1 -> u*u_x, u^2x0_1 -> u^2*u_x
    if 'x0_' in name:
        parts = name.split('x0_')
        prefix = parts[0]  # 'u' or 'u^2'
        order = len(parts[1])
        deriv = 'u_' + 'x' * order
        if prefix == 'u':
            return f'u*{deriv}'
        else:
            return f'{prefix}*{deriv}'
    return name

# ============ SPD (ours) ============
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
    beta=np.linalg.lstsq(Theta[:,final],target,rcond=None)[0] if final else np.array([])
    return final,beta

def spd_1d(u,du,x,t,true_terms,max_order=4):
    t0=time.time()
    u_t_spec,derivs=spectral_spacetime(u,x,t,max_order)
    target=(du if du is not None else u_t_spec).ravel()
    lib,norms=build_lib_1d(derivs,normalize=True)
    names=list(lib.keys()); Theta=np.column_stack([lib[n].ravel() for n in names])
    sel,beta=fwbic(Theta,target)
    rec={names[i]:float(beta[j]/norms[names[i]]) for j,i in enumerate(sel)}
    tf=sum(1 for tt in true_terms if tt in rec); fp=[n for n in rec if n not in true_terms]
    return {'recovered':rec,'true_found':tf,'total_true':len(true_terms),'false_positives':fp,
            'exact':(tf==len(true_terms)) and (len(fp)==0),'time_s':time.time()-t0}

# ============ Weak SINDy (pysindy) ============
def weak_sindy_1d(u,du,x,t,true_terms,max_order=4):
    import pysindy as ps
    t0=time.time()
    u_3d = u[:,:,np.newaxis]  # (n_x, n_t, 1)
    ST=np.stack(np.meshgrid(x,t,indexing='ij'),axis=-1)  # (n_x, n_t, 2)
    lib=ps.WeakPDELibrary(
        library_functions=[lambda x:x, lambda x:x**2],
        function_names=[lambda x:'u', lambda x:'u^2'],
        derivative_order=max_order, include_bias=False,
        spatiotemporal_grid=ST, is_uniform=True, K=2000)
    best=None
    for th in np.logspace(-6,0,13):
        opt=ps.SR3(threshold=th,max_iter=200)
        model=ps.SINDy(feature_library=lib,optimizer=opt)
        try:
            model.fit(u_3d, t=t)
            coeffs=model.coefficients()[0]
            feats=model.get_feature_names()
            rec={}
            for i in range(len(coeffs)):
                if abs(coeffs[i])>1e-10:
                    std_name = pysindy_name_to_std(feats[i])
                    rec[std_name]=float(coeffs[i])
            tf=sum(1 for tt in true_terms if tt in rec); fp=[n for n in rec if n not in true_terms]
            exact=(tf==len(true_terms)) and (len(fp)==0)
            if best is None or (exact and not best[2]) or (len(fp)<len(best[1]) and tf>=best[0]):
                best=(tf,fp,exact,rec)
            if exact: break
        except Exception as e:
            pass
    tf,fp,exact,rec=best if best else (0,[],False,{})
    return {'recovered':rec,'true_found':tf,'total_true':len(true_terms),'false_positives':fp,'exact':exact,'time_s':time.time()-t0}

# ============ SR3 (SINDy-PI style) with FD derivatives ============
def sr3_fd_1d(u,du,x,t,true_terms,max_order=4):
    import pysindy as ps
    t0=time.time()
    u_3d = u[:,:,np.newaxis]
    s=x.reshape(-1,1)
    lib=ps.PDELibrary(
        library_functions=[lambda x:x, lambda x:x**2],
        function_names=[lambda x:'u', lambda x:'u^2'],
        derivative_order=max_order, spatial_grid=s, include_bias=False)
    best=None
    for th in np.logspace(-6,0,13):
        opt=ps.SR3(threshold=th,max_iter=200)
        model=ps.SINDy(feature_library=lib,optimizer=opt)
        try:
            model.fit(u_3d, t=t)
            coeffs=model.coefficients()[0]
            feats=model.get_feature_names()
            rec={}
            for i in range(len(coeffs)):
                if abs(coeffs[i])>1e-10:
                    std_name = pysindy_name_to_std(feats[i])
                    rec[std_name]=float(coeffs[i])
            tf=sum(1 for tt in true_terms if tt in rec); fp=[n for n in rec if n not in true_terms]
            exact=(tf==len(true_terms)) and (len(fp)==0)
            if best is None or (exact and not best[2]) or (len(fp)<len(best[1]) and tf>=best[0]):
                best=(tf,fp,exact,rec)
            if exact: break
        except Exception as e:
            pass
    tf,fp,exact,rec=best if best else (0,[],False,{})
    return {'recovered':rec,'true_found':tf,'total_true':len(true_terms),'false_positives':fp,'exact':exact,'time_s':time.time()-t0}

TRUE_PDES = {
    'burgers': {'u*u_x': -1.0, 'u_xx': 0.1},
    'kdv': {'u*u_x': -6.0, 'u_xxx': -1.0},
    'kuramoto_sivishinky': {'u*u_x': -1.0, 'u_xx': -1.0, 'u_xxxx': -1.0},
    'advection1d': {'u_x': -0.1},
}

def load_1d(name, noise_file=None):
    path = DATA_DIR/name/(noise_file if noise_file else f"{name}.npz")
    data=np.load(path); t=data['t']; x=data['x']; u=data['u']
    du=data['du'] if 'du' in data else None
    if u.ndim==3: u=u[:,:,0]; du=du[:,:,0] if du is not None else None
    return u,du,x,t

def main():
    print("="*80)
    print("COMPREHENSIVE COMPARISON: SPD vs Weak SINDy vs SR3(FD)")
    print("="*80)
    
    methods = ['SR3 (FD)', 'Weak SINDy', 'SPD (ours)']
    datasets = ['burgers','kdv','kuramoto_sivishinky','advection1d']
    noises = [None,'snr_40','snr_30','snr_20','snr_10']
    
    results = {}
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"  {ds}")
        print(f"{'='*60}")
        results[ds] = {}
        tt = TRUE_PDES[ds]
        for noise in noises:
            label='clean' if noise is None else noise
            nf=f"{ds}_{noise}.npz" if noise else None
            try:
                u,du,x,t=load_1d(ds,nf)
                results[ds][label]={}
                row = f"  {label:8s}"
                for m in methods:
                    if m=='SPD (ours)': r=spd_1d(u,du,x,t,tt)
                    elif m=='Weak SINDy': r=weak_sindy_1d(u,du,x,t,tt)
                    else: r=sr3_fd_1d(u,du,x,t,tt)
                    results[ds][label][m]=r
                    status="EXACT" if r['exact'] else f"{r['true_found']}/{r['total_true']}+{len(r['false_positives'])}FP"
                    row += f" | {m:14s}:{status:12s}({r['time_s']:.1f}s)"
                print(row)
            except Exception as e:
                print(f"  {label}: ERROR {e}")
                import traceback; traceback.print_exc()
    
    out=Path(r"H:\2026科研\Spectral-PDE-Discovery\code\comparison_strong_baselines.json")
    with open(out,'w') as f:
        json.dump(results,f,indent=2,default=lambda x:float(x) if hasattr(x,'item') else x)
    print(f"\nSaved to {out}")

if __name__=='__main__':
    main()
