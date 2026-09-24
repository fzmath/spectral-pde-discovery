"""3D heat conduction experiment (simplified: clean + snr_40 only)."""
import numpy as np
from pathlib import Path
import json, time, warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r"H:\2026科研\Spectral-PDE-Discovery\mdbench\data\processed\data\pde")

def spectral_3d(u, x, y, z, filter_order=8, cutoff=0.6):
    """3D spectral spatial derivatives."""
    Nx,Ny,Nz,Nt = u.shape
    Lx=x[-1]-x[0]+(x[1]-x[0]); Ly=y[-1]-y[0]+(y[1]-y[0]); Lz=z[-1]-z[0]+(z[1]-z[0])
    kx=2*np.pi*np.fft.fftfreq(Nx,d=Lx/Nx); ky=2*np.pi*np.fft.fftfreq(Ny,d=Ly/Ny); kz=2*np.pi*np.fft.fftfreq(Nz,d=Lz/Nz)
    kxmax=np.max(np.abs(kx)); kymax=np.max(np.abs(ky)); kzmax=np.max(np.abs(kz)); alpha=-np.log(1e-12)
    KX,KY,KZ=np.meshgrid(kx,ky,kz,indexing='ij')
    filt=np.ones_like(KX)
    filt[np.abs(KX)>cutoff*kxmax]*=np.exp(-alpha*(np.abs(KX[np.abs(KX)>cutoff*kxmax])/kxmax)**filter_order)
    filt[np.abs(KY)>cutoff*kymax]*=np.exp(-alpha*(np.abs(KY[np.abs(KY)>cutoff*kymax])/kymax)**filter_order)
    filt[np.abs(KZ)>cutoff*kzmax]*=np.exp(-alpha*(np.abs(KZ[np.abs(KZ)>cutoff*kzmax])/kzmax)**filter_order)
    derivs={'u':u.copy(),'u_xx':np.zeros_like(u),'u_yy':np.zeros_like(u),'u_zz':np.zeros_like(u)}
    for nt in range(Nt):
        uh=np.fft.fftn(u[:,:,:,nt])*filt
        derivs['u_xx'][:,:,:,nt]=np.real(np.fft.ifftn((1j*KX)**2*uh))
        derivs['u_yy'][:,:,:,nt]=np.real(np.fft.ifftn((1j*KY)**2*uh))
        derivs['u_zz'][:,:,:,nt]=np.real(np.fft.ifftn((1j*KZ)**2*uh))
    return derivs

def fd_3d(u,x,y,z):
    dx=x[1]-x[0]; dy=y[1]-y[0]; dz=z[1]-z[0]
    return {'u':u.copy(),
            'u_xx':(np.roll(u,-1,0)-2*u+np.roll(u,1,0))/dx**2,
            'u_yy':(np.roll(u,-1,1)-2*u+np.roll(u,1,1))/dy**2,
            'u_zz':(np.roll(u,-1,2)-2*u+np.roll(u,1,2))/dz**2}

def build_lib(derivs, normalize=True):
    f={'u':derivs['u'],'u_xx':derivs['u_xx'],'u_yy':derivs['u_yy'],'u_zz':derivs['u_zz'],
       'u^2':derivs['u']**2,'u^3':derivs['u']**3}
    if normalize:
        norms={k:max(np.linalg.norm(v.ravel()),1e-12) for k,v in f.items()}
        f={k:v/norms[k] for k,v in f.items()}
        return f,norms
    return f,None

def stlsq(Theta,target,threshold=0.02,alpha=1e-5,max_iter=20):
    nf=Theta.shape[1]; xi=np.linalg.lstsq(Theta.T@Theta+alpha*np.eye(nf),Theta.T@target,rcond=None)[0]
    for _ in range(max_iter):
        s=np.abs(xi)<threshold; xi[s]=0; b=~s
        if b.sum()==0: break
        xi[b]=np.linalg.lstsq(Theta[:,b],target,rcond=None)[0]
    return xi

def fwbic(Theta,target,max_terms=6,penalty=2.0):
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
    # backward
    final=best_sel.copy(); improved=True
    while improved and len(final)>0:
        improved=False; cb=compute_bic(Theta,target,final,n,penalty)
        for j in final:
            trial=[x for x in final if x!=j]; tb=compute_bic(Theta,target,trial,n,penalty)
            if tb<=cb: final=trial; improved=True; break
    beta=np.linalg.lstsq(Theta[:,final],target,rcond=None)[0] if final else np.array([])
    return final,beta

def compute_bic(Theta,target,sel,n,penalty):
    if len(sel)==0: rss=np.sum(target**2); k=0
    else: X=Theta[:,sel]; beta=np.linalg.lstsq(X,target,rcond=None)[0]; rss=np.sum((target-X@beta)**2); k=len(sel)
    return n*np.log(max(rss/n,1e-15))+penalty*k*np.log(n)

def evaluate(rec,true):
    tf=sum(1 for t in true if t in rec); fp=[n for n in rec if n not in true]
    return tf,fp,(tf==len(true)) and (len(fp)==0)

def main():
    ds='heat_soil_uniform_3d_p1'
    print("="*60)
    print(f"3D {ds}")
    data=np.load(DATA_DIR/ds/f"{ds}.npz")
    u=data['u'][:,:,:,:,0]; du=data['du'][:,:,:,:,0]
    x,y,z,t=data['x'],data['y'],data['z'],data['t']
    print(f"  Shape: {u.shape}, range: [{u.min():.2f}, {u.max():.2f}]")
    true={'u_xx':1.0,'u_yy':1.0,'u_zz':1.0}
    results={}
    for noise in [None,'snr_40']:
        label='clean' if noise is None else noise
        nf=f"{ds}_{noise}.npz" if noise else None
        if nf:
            d2=np.load(DATA_DIR/ds/nf); u2=d2['u'][:,:,:,:,0]; du2=d2['du'][:,:,:,:,0]
        else: u2,du2=u,du
        results[label]={}
        print(f"\n--- {label} ---")
        for method in ['FD+STLSQ','Spectral+STLSQ','SPD (ours)']:
            t0=time.time()
            if method=='FD+STLSQ': derivs=fd_3d(u2,x,y,z); lib,_=build_lib(derivs,normalize=False)
            elif method=='Spectral+STLSQ': derivs=spectral_3d(u2,x,y,z); lib,_=build_lib(derivs,normalize=False)
            else: derivs=spectral_3d(u2,x,y,z); lib,norms=build_lib(derivs,normalize=True)
            names=list(lib.keys()); Theta=np.column_stack([lib[n].ravel() for n in names]); target=du2.ravel()
            if method in ['FD+STLSQ','Spectral+STLSQ']:
                best=None
                for th in [0.001,0.005,0.01,0.02,0.05,0.1]:
                    xi=stlsq(Theta,target,threshold=th)
                    rec={names[i]:float(xi[i]) for i in range(len(xi)) if abs(xi[i])>1e-6}
                    tf,fp,exact=evaluate(rec,true)
                    if best is None or (exact and not best[2]) or (len(fp)<len(best[1]) and tf>=best[0]): best=(tf,fp,exact,rec)
                    if exact: break
                tf,fp,exact,rec=best
            else:
                sel,beta=fwbic(Theta,target)
                rec={names[i]:float(beta[j]/norms[names[i]]) for j,i in enumerate(sel)}
                tf,fp,exact=evaluate(rec,true)
            elapsed=time.time()-t0
            results[label][method]={'true_found':tf,'total_true':3,'false_positives':fp,'exact':exact,'time_s':elapsed}
            status="EXACT" if exact else f"{tf}/3+{len(fp)}FP"
            print(f"  {method:20s}: {status} ({elapsed:.1f}s)")
    out=Path(r"H:\2026科研\Spectral-PDE-Discovery\code\spd_3d_results.json")
    with open(out,'w') as f: json.dump(results,f,indent=2)
    print(f"\nSaved to {out}")

if __name__=='__main__':
    main()
