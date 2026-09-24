"""
Extended 2D experiments: reaction_diffusion_2d + heat_soil_2d
"""
import numpy as np
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r"H:\2026科研\Spectral-PDE-Discovery\mdbench\data\processed\data\pde")

def spectral_spatial_2d(u, x, y, max_order=2, filter_order=8, cutoff=0.6):
    Nx, Ny, Nt = u.shape
    Lx = x[-1] - x[0] + (x[1]-x[0])
    Ly = y[-1] - y[0] + (y[1]-y[0])
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=Lx/Nx)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=Ly/Ny)
    kx_max = np.max(np.abs(kx))
    ky_max = np.max(np.abs(ky))
    alpha = -np.log(1e-12)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    filt = np.ones_like(KX)
    filt[np.abs(KX) > cutoff*kx_max] *= np.exp(-alpha*(np.abs(KX[np.abs(KX)>cutoff*kx_max])/kx_max)**filter_order)
    filt[np.abs(KY) > cutoff*ky_max] *= np.exp(-alpha*(np.abs(KY[np.abs(KY)>cutoff*ky_max])/ky_max)**filter_order)
    derivs = {'u': u.copy(), 'u_x': np.zeros_like(u), 'u_y': np.zeros_like(u),
              'u_xx': np.zeros_like(u), 'u_yy': np.zeros_like(u)}
    for nt in range(Nt):
        uh = np.fft.fft2(u[:,:,nt]) * filt
        derivs['u_x'][:,:,nt] = np.real(np.fft.ifft2((1j*KX)*uh))
        derivs['u_y'][:,:,nt] = np.real(np.fft.ifft2((1j*KY)*uh))
        derivs['u_xx'][:,:,nt] = np.real(np.fft.ifft2((1j*KX)**2*uh))
        derivs['u_yy'][:,:,nt] = np.real(np.fft.ifft2((1j*KY)**2*uh))
    return derivs

def finite_diff_spatial_2d(u, x, y):
    dx = x[1]-x[0]; dy = y[1]-y[0]
    return {'u': u.copy(),
            'u_x': (np.roll(u,-1,0)-np.roll(u,1,0))/(2*dx),
            'u_y': (np.roll(u,-1,1)-np.roll(u,1,1))/(2*dy),
            'u_xx': (np.roll(u,-1,0)-2*u+np.roll(u,1,0))/dx**2,
            'u_yy': (np.roll(u,-1,1)-2*u+np.roll(u,1,1))/dy**2}

def build_library_2d(derivs, poly_order=2, normalize=True):
    u = derivs['u']
    features = {'u': u, 'u_x': derivs['u_x'], 'u_y': derivs['u_y'],
                'u_xx': derivs['u_xx'], 'u_yy': derivs['u_yy'],
                'u*u_x': u*derivs['u_x'], 'u*u_y': u*derivs['u_y']}
    if poly_order >= 2:
        features.update({'u^2': u**2, 'u^3': u**3, 'u^2*u_x': u**2*derivs['u_x'], 'u^2*u_y': u**2*derivs['u_y']})
    if normalize:
        norms = {k: max(np.linalg.norm(v.ravel()), 1e-12) for k,v in features.items()}
        features = {k: v/norms[k] for k,v in features.items()}
        return features, norms
    return features, None

def stlsq(Theta, target, threshold=0.02, alpha=1e-5, max_iter=20):
    nf = Theta.shape[1]
    xi = np.linalg.lstsq(Theta.T@Theta + alpha*np.eye(nf), Theta.T@target, rcond=None)[0]
    for _ in range(max_iter):
        small = np.abs(xi) < threshold
        xi[small] = 0
        big = ~small
        if big.sum() == 0: break
        xi[big] = np.linalg.lstsq(Theta[:,big], target, rcond=None)[0]
    return xi

def forward_stepwise_bic(Theta, target, max_terms=8):
    n = len(target)
    selected, remaining = [], list(range(Theta.shape[1]))
    best_bic, best_sel = np.inf, []
    for step in range(min(max_terms, Theta.shape[1])):
        best_j, best_b = None, np.inf
        for j in remaining:
            cand = selected + [j]
            X = Theta[:, cand]
            beta = np.linalg.lstsq(X, target, rcond=None)[0]
            rss = np.sum((target - X@beta)**2)
            bic = n*np.log(max(rss/n, 1e-15)) + len(cand)*np.log(n)
            if bic < best_b: best_b, best_j = bic, j
        if best_j is None: break
        selected.append(best_j); remaining.remove(best_j)
        if best_b < best_bic - 1e-6:
            best_bic, best_sel = best_b, selected.copy()
        elif step > 0 and best_b > best_bic + 0.05*np.log(n):
            break
    if best_sel:
        beta = np.linalg.lstsq(Theta[:,best_sel], target, rcond=None)[0]
    else:
        beta = np.array([])
    return best_sel, beta

def evaluate(recovered, true_terms):
    tf = sum(1 for t in true_terms if t in recovered)
    fp = [n for n in recovered if n not in true_terms]
    exact = (tf == len(true_terms)) and (len(fp) == 0)
    return tf, fp, exact

def test_2d(method, u, du, x, y, true_terms):
    t0 = time.time()
    if method == 'FD+STLSQ':
        derivs = finite_diff_spatial_2d(u, x, y)
        lib, _ = build_library_2d(derivs, normalize=False)
    elif method == 'Spectral+STLSQ':
        derivs = spectral_spatial_2d(u, x, y)
        lib, _ = build_library_2d(derivs, normalize=False)
    else:
        derivs = spectral_spatial_2d(u, x, y)
        lib, norms = build_library_2d(derivs, normalize=True)
    names = list(lib.keys())
    Theta = np.column_stack([lib[n].ravel() for n in names])
    target = du.ravel()
    if method in ['FD+STLSQ', 'Spectral+STLSQ']:
        best = None
        for th in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
            xi = stlsq(Theta, target, threshold=th)
            rec = {names[i]: float(xi[i]) for i in range(len(xi)) if abs(xi[i]) > 1e-6}
            tf, fp, exact = evaluate(rec, true_terms)
            if best is None or (exact and not best[2]) or (len(fp) < len(best[1]) and tf >= best[0]):
                best = (tf, fp, exact, rec)
            if exact: break
        tf, fp, exact, rec = best
    else:
        sel, beta = forward_stepwise_bic(Theta, target)
        rec = {names[i]: float(beta[j]/norms[names[i]]) for j,i in enumerate(sel)}
        tf, fp, exact = evaluate(rec, true_terms)
    return {'recovered': rec, 'true_found': tf, 'total_true': len(true_terms),
            'false_positives': fp, 'exact': exact, 'time_s': time.time()-t0}

def main():
    results = {}
    methods = ['FD+STLSQ', 'Spectral+STLSQ', 'SPD (ours)']
    
    # Reaction-diffusion 2D (component 0)
    # Gray-Scott: u_t = D_u (u_xx+u_yy) - u*v^2 + F(1-u)
    # For component 0 alone, true terms: u_xx, u_yy, u, u^2*v? (v not available)
    # Simplified: u_t = D_u (u_xx + u_yy) + reaction(u)
    print("="*60)
    print("2D Reaction-Diffusion (component 0)")
    ds = 'reaction_diffusion_2d'
    data = np.load(DATA_DIR / ds / f"{ds}.npz")
    u = data['u'][:,:,:,0]  # component 0
    du = data['du'][:,:,:,0]
    x, y, t = data['x'], data['y'], data['t']
    # True: u_t = D_u (u_xx + u_yy) - u*v^2 + F(1-u)
    # With only u component, we expect diffusion + reaction terms
    true_terms = {'u_xx': 1.0, 'u_yy': 1.0, 'u': -1.0}  # approximate
    results[ds] = {}
    for noise in [None, 'snr_40', 'snr_20']:
        label = 'clean' if noise is None else noise
        nf = f"{ds}_{noise}.npz" if noise else None
        if nf:
            d2 = np.load(DATA_DIR / ds / nf)
            u2 = d2['u'][:,:,:,0]; du2 = d2['du'][:,:,:,0]
        else:
            u2, du2 = u, du
        results[ds][label] = {}
        print(f"\n--- {label} ---")
        for m in methods:
            r = test_2d(m, u2, du2, x, y, true_terms)
            results[ds][label][m] = r
            status = "EXACT" if r['exact'] else f"{r['true_found']}/{r['total_true']}+{len(r['false_positives'])}FP"
            print(f"  {m:20s}: {status} ({r['time_s']:.2f}s)")
    
    # Heat soil 2D
    print("\n" + "="*60)
    print("2D Heat Soil")
    ds = 'heat_soil_uniform_2d_p1'
    data = np.load(DATA_DIR / ds / f"{ds}.npz")
    u = data['u'][:,:,:,0]
    du = data['du'][:,:,:,0]
    x, y = data['x'], data['y']
    # Heat equation: u_t = alpha (u_xx + u_yy) + source
    true_terms = {'u_xx': 1.0, 'u_yy': 1.0}
    results[ds] = {}
    for noise in [None, 'snr_40', 'snr_20']:
        label = 'clean' if noise is None else noise
        nf = f"{ds}_{noise}.npz" if noise else None
        if nf:
            d2 = np.load(DATA_DIR / ds / nf)
            u2 = d2['u'][:,:,:,0]; du2 = d2['du'][:,:,:,0]
        else:
            u2, du2 = u, du
        results[ds][label] = {}
        print(f"\n--- {label} ---")
        for m in methods:
            r = test_2d(m, u2, du2, x, y, true_terms)
            results[ds][label][m] = r
            status = "EXACT" if r['exact'] else f"{r['true_found']}/{r['total_true']}+{len(r['false_positives'])}FP"
            print(f"  {m:20s}: {status} ({r['time_s']:.2f}s)")
    
    out = Path(r"H:\2026科研\Spectral-PDE-Discovery\code\spd_2d_extended.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x,'item') else x)
    print(f"\nSaved to {out}")

if __name__ == '__main__':
    main()
