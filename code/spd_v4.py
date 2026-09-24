"""
SPD v4: Improved BIC with backward elimination
Key changes:
1. Stricter stopping: stop as soon as BIC increases (no tolerance)
2. Backward elimination after forward selection
3. Optional EBIC (Extended BIC) with stronger penalty
"""
import numpy as np
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r"H:\2026科研\Spectral-PDE-Discovery\mdbench\data\processed\data\pde")

def spectral_spacetime(u, x, t, max_order=4, filter_order=8, cutoff=0.6):
    Nx, Nt = u.shape
    Lx = x[-1] - x[0] + (x[1]-x[0])
    Lt = t[-1] - t[0] + (t[1]-t[0])
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=Lx/Nx)
    kt = 2*np.pi*np.fft.fftfreq(Nt, d=Lt/Nt)
    kx_max = np.max(np.abs(kx)); kt_max = np.max(np.abs(kt))
    alpha = -np.log(1e-12)
    KX, KT = np.meshgrid(kx, kt, indexing='ij')
    filt = np.ones_like(KX)
    filt[np.abs(KX)>cutoff*kx_max] *= np.exp(-alpha*(np.abs(KX[np.abs(KX)>cutoff*kx_max])/kx_max)**filter_order)
    filt[np.abs(KT)>cutoff*kt_max] *= np.exp(-alpha*(np.abs(KT[np.abs(KT)>cutoff*kt_max])/kt_max)**filter_order)
    uh = np.fft.fft2(u) * filt
    u_t = np.real(np.fft.ifft2((1j*KT)*uh))
    derivs = [u.copy()]
    for o in range(1, max_order+1):
        derivs.append(np.real(np.fft.ifft2((1j*KX)**o*uh)))
    return u_t, derivs

def finite_diff_spatial(u, x, max_order=4):
    dx = x[1]-x[0]
    derivs = [u.copy()]
    for o in range(1, max_order+1):
        if o==1: d = (np.roll(u,-1,0)-np.roll(u,1,0))/(2*dx)
        else: d = (np.roll(derivs[-1],-1,0)-np.roll(derivs[-1],1,0))/(2*dx)
        derivs.append(d)
    return derivs

def spectral_spatial(u, x, max_order=4, filter_order=8, cutoff=0.6):
    N=len(x); L=x[-1]-x[0]+(x[1]-x[0])
    k=2*np.pi*np.fft.fftfreq(N,d=L/N); kmax=np.max(np.abs(k)); alpha=-np.log(1e-12)
    f=np.ones_like(k); f[np.abs(k)>cutoff*kmax]=np.exp(-alpha*(np.abs(k[np.abs(k)>cutoff*kmax])/kmax)**filter_order)
    derivs=[u.copy()]+[np.zeros_like(u) for _ in range(max_order)]
    for nt in range(u.shape[1]):
        uh=np.fft.fft(u[:,nt])*f
        for o in range(1,max_order+1):
            derivs[o][:,nt]=np.real(np.fft.ifft((1j*k)**o*uh))
    return derivs

def build_library_1d(derivs, poly_order=2, normalize=True):
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

def stlsq(Theta,target,threshold=0.02,alpha=1e-5,max_iter=20):
    nf=Theta.shape[1]; xi=np.linalg.lstsq(Theta.T@Theta+alpha*np.eye(nf),Theta.T@target,rcond=None)[0]
    for _ in range(max_iter):
        small=np.abs(xi)<threshold; xi[small]=0; big=~small
        if big.sum()==0: break
        xi[big]=np.linalg.lstsq(Theta[:,big],target,rcond=None)[0]
    return xi

def compute_bic(Theta,target,sel,n,penalty=1.0):
    if len(sel)==0:
        rss=np.sum(target**2); k=0
    else:
        X=Theta[:,sel]; beta=np.linalg.lstsq(X,target,rcond=None)[0]
        rss=np.sum((target-X@beta)**2); k=len(sel)
    return n*np.log(max(rss/n,1e-15)) + penalty*k*np.log(n)

def forward_backward_bic(Theta, target, max_terms=8, penalty=1.0):
    """Forward stepwise with strict stopping + backward elimination."""
    n=len(target); P=Theta.shape[1]
    selected=[]; remaining=list(range(P))
    best_bic=np.inf; best_sel=[]
    
    # Forward: strict stopping (stop as soon as BIC increases)
    for step in range(min(max_terms,P)):
        best_j=None; best_b=np.inf
        for j in remaining:
            bic=compute_bic(Theta,target,selected+[j],n,penalty)
            if bic<best_b: best_b=bic; best_j=j
        if best_j is None: break
        # Strict: if BIC increases from best, stop immediately
        if best_b >= best_bic and step > 0:
            break
        selected.append(best_j); remaining.remove(best_j)
        if best_b < best_bic:
            best_bic=best_b; best_sel=selected.copy()
    
    # Backward elimination: remove terms that don't improve BIC
    final_sel = best_sel.copy()
    improved = True
    while improved and len(final_sel) > 0:
        improved = False
        current_bic = compute_bic(Theta, target, final_sel, n, penalty)
        for j in final_sel:
            trial = [x for x in final_sel if x != j]
            trial_bic = compute_bic(Theta, target, trial, n, penalty)
            if trial_bic <= current_bic:  # removing j improves or doesn't hurt
                final_sel = trial
                improved = True
                break
    
    if final_sel:
        beta=np.linalg.lstsq(Theta[:,final_sel],target,rcond=None)[0]
    else:
        beta=np.array([])
    return final_sel, beta

def evaluate(recovered, true_terms):
    tf=sum(1 for t in true_terms if t in recovered)
    fp=[n for n in recovered if n not in true_terms]
    exact=(tf==len(true_terms)) and (len(fp)==0)
    return tf,fp,exact

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

def test_method(method, u, du, x, t, true_terms, max_order=4, penalty=1.0):
    t0=time.time()
    u_t = du if du is not None else None
    if method=='FD+STLSQ':
        derivs=finite_diff_spatial(u,x,max_order)
        lib,_=build_library_1d(derivs,normalize=False)
        names=list(lib.keys()); Theta=np.column_stack([lib[n].ravel() for n in names])
        target=(u_t if u_t is not None else np.zeros_like(u)).ravel()
        best=None
        for th in [0.001,0.005,0.01,0.02,0.05,0.1]:
            xi=stlsq(Theta,target,threshold=th)
            rec={names[i]:float(xi[i]) for i in range(len(xi)) if abs(xi[i])>1e-6}
            tf,fp,exact=evaluate(rec,true_terms)
            if best is None or (exact and not best[2]) or (len(fp)<len(best[1]) and tf>=best[0]):
                best=(tf,fp,exact,rec)
            if exact: break
        tf,fp,exact,rec=best
    elif method=='Spectral+STLSQ':
        derivs=spectral_spatial(u,x,max_order)
        lib,_=build_library_1d(derivs,normalize=False)
        names=list(lib.keys()); Theta=np.column_stack([lib[n].ravel() for n in names])
        target=(u_t if u_t is not None else np.zeros_like(u)).ravel()
        best=None
        for th in [0.001,0.005,0.01,0.02,0.05,0.1]:
            xi=stlsq(Theta,target,threshold=th)
            rec={names[i]:float(xi[i]) for i in range(len(xi)) if abs(xi[i])>1e-6}
            tf,fp,exact=evaluate(rec,true_terms)
            if best is None or (exact and not best[2]) or (len(fp)<len(best[1]) and tf>=best[0]):
                best=(tf,fp,exact,rec)
            if exact: break
        tf,fp,exact,rec=best
    else: # SPD v4
        u_t_spec,derivs=spectral_spacetime(u,x,t,max_order)
        target=(u_t if u_t is not None else u_t_spec).ravel()
        lib,norms=build_library_1d(derivs,normalize=True)
        names=list(lib.keys()); Theta=np.column_stack([lib[n].ravel() for n in names])
        sel,beta=forward_backward_bic(Theta,target,max_terms=8,penalty=penalty)
        rec={names[i]:float(beta[j]/norms[names[i]]) for j,i in enumerate(sel)}
        tf,fp,exact=evaluate(rec,true_terms)
    return {'recovered':rec,'true_found':tf,'total_true':len(true_terms),
            'false_positives':fp,'exact':exact,'time_s':time.time()-t0}

def main():
    print("="*70)
    print("SPD v4: Forward-Backward BIC (reduced false positives)")
    print("="*70)
    
    results={}
    methods=['FD+STLSQ','Spectral+STLSQ','SPD v4 (ours)']
    datasets=['burgers','kdv','kuramoto_sivishinky','advection1d']
    noises=[None,'snr_40','snr_30','snr_20','snr_10']
    
    # Test different penalty values
    for penalty in [1.0, 2.0, 3.0]:
        print(f"\n{'#'*60}")
        print(f"# BIC penalty = {penalty}")
        print(f"{'#'*60}")
        results[f'penalty_{penalty}']={}
        for ds in datasets:
            print(f"\n--- {ds} ---")
            results[f'penalty_{penalty}'][ds]={}
            tt=TRUE_PDES[ds]
            for noise in noises:
                label='clean' if noise is None else noise
                nf=f"{ds}_{noise}.npz" if noise else None
                try:
                    u,du,x,t=load_1d(ds,nf)
                    results[f'penalty_{penalty}'][ds][label]={}
                    for m in methods:
                        r=test_method(m,u,du,x,t,tt,penalty=penalty)
                        results[f'penalty_{penalty}'][ds][label][m]=r
                        status="EXACT" if r['exact'] else f"{r['true_found']}/{r['total_true']}+{len(r['false_positives'])}FP"
                        if m=='SPD v4 (ours)':
                            print(f"  {label:8s} {m:20s}: {status} ({r['time_s']:.2f}s)")
                except Exception as e:
                    print(f"  {label}: ERROR {e}")
    
    out=Path(r"H:\2026科研\Spectral-PDE-Discovery\code\spd_v4_results.json")
    with open(out,'w') as f:
        json.dump(results,f,indent=2,default=lambda x:float(x) if hasattr(x,'item') else x)
    print(f"\nSaved to {out}")
    
    # Summary: compare false positives
    print("\n"+"="*80)
    print("FALSE POSITIVES COMPARISON (SPD v4)")
    print("="*80)
    print(f"{'PDE':<20} {'Noise':<8} {'p=1.0':<10} {'p=2.0':<10} {'p=3.0':<10}")
    for ds in datasets:
        for noise in ['clean','snr_40','snr_30','snr_20','snr_10']:
            row=f"{ds:<20} {noise:<8}"
            for p in [1.0,2.0,3.0]:
                key=f'penalty_{p}'
                if noise in results[key][ds]:
                    fp=len(results[key][ds][noise]['SPD v4 (ours)']['false_positives'])
                    tf=results[key][ds][noise]['SPD v4 (ours)']['true_found']
                    row+=f" {tf}/{fp}FP   "
                else: row+=" N/A       "
            print(row)

if __name__=='__main__':
    main()
