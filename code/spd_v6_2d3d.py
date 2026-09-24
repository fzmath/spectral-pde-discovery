"""
SPD v6 for 2D/3D PDEs: Spectral derivatives + Weak formulation + FWBIC
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
    mask_x = np.abs(KX) > cutoff*kx_max
    mask_y = np.abs(KY) > cutoff*ky_max
    filt[mask_x] *= np.exp(-alpha*(np.abs(KX[mask_x])/kx_max)**filter_order)
    filt[mask_y] *= np.exp(-alpha*(np.abs(KY[mask_y])/ky_max)**filter_order)
    derivs = {'u': u.copy(), 'u_x': np.zeros_like(u), 'u_y': np.zeros_like(u),
              'u_xx': np.zeros_like(u), 'u_yy': np.zeros_like(u)}
    for nt in range(Nt):
        uh = np.fft.fft2(u[:,:,nt]) * filt
        derivs['u_x'][:,:,nt] = np.real(np.fft.ifft2((1j*KX)*uh))
        derivs['u_y'][:,:,nt] = np.real(np.fft.ifft2((1j*KY)*uh))
        derivs['u_xx'][:,:,nt] = np.real(np.fft.ifft2((1j*KX)**2*uh))
        derivs['u_yy'][:,:,nt] = np.real(np.fft.ifft2((1j*KY)**2*uh))
    return derivs

def spectral_spatial_3d(u, x, y, z, filter_order=8, cutoff=0.6):
    Nx, Ny, Nz, Nt = u.shape
    Lx = x[-1] - x[0] + (x[1]-x[0])
    Ly = y[-1] - y[0] + (y[1]-y[0])
    Lz = z[-1] - z[0] + (z[1]-z[0])
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=Lx/Nx)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=Ly/Ny)
    kz = 2*np.pi*np.fft.fftfreq(Nz, d=Lz/Nz)
    kx_max = np.max(np.abs(kx)); ky_max = np.max(np.abs(ky)); kz_max = np.max(np.abs(kz))
    alpha = -np.log(1e-12)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    filt = np.ones_like(KX)
    for K, kmax in [(KX, kx_max), (KY, ky_max), (KZ, kz_max)]:
        mask = np.abs(K) > cutoff*kmax
        filt[mask] *= np.exp(-alpha*(np.abs(K[mask])/kmax)**filter_order)
    derivs = {'u': u.copy(), 'u_xx': np.zeros_like(u), 'u_yy': np.zeros_like(u), 'u_zz': np.zeros_like(u)}
    for nt in range(Nt):
        uh = np.fft.fftn(u[:,:,:,nt]) * filt
        derivs['u_xx'][:,:,:,nt] = np.real(np.fft.ifftn((1j*KX)**2*uh))
        derivs['u_yy'][:,:,:,nt] = np.real(np.fft.ifftn((1j*KY)**2*uh))
        derivs['u_zz'][:,:,:,nt] = np.real(np.fft.ifftn((1j*KZ)**2*uh))
    return derivs

def build_library_2d(derivs, poly_order=2):
    u = derivs['u']
    features = {'u': u, 'u_x': derivs['u_x'], 'u_y': derivs['u_y'],
                'u_xx': derivs['u_xx'], 'u_yy': derivs['u_yy'],
                'u*u_x': u*derivs['u_x'], 'u*u_y': u*derivs['u_y']}
    if poly_order >= 2:
        features.update({'u^2': u**2, 'u^2*u_x': u**2*derivs['u_x'], 'u^2*u_y': u**2*derivs['u_y']})
    return features

def build_library_3d(derivs):
    return {'u': derivs['u'], 'u_xx': derivs['u_xx'], 'u_yy': derivs['u_yy'], 'u_zz': derivs['u_zz']}

def weak_integrate(field, test_funcs, dx, dy, dt):
    """Integrate field against each test function (trapezoidal)."""
    n_test = len(test_funcs)
    result = np.zeros(n_test)
    for i, phi in enumerate(test_funcs):
        result[i] = np.sum(field * phi) * dx * dy * dt
    return result

def weak_integrate_3d(field, test_funcs, dx, dy, dz, dt):
    n_test = len(test_funcs)
    result = np.zeros(n_test)
    for i, phi in enumerate(test_funcs):
        result[i] = np.sum(field * phi) * dx * dy * dz * dt
    return result

def generate_test_funcs_2d(x, y, t, n_test=50, width_ratio=0.15, seed=42):
    Nx, Ny, Nt = len(x), len(y), len(t)
    Lx = x[-1]-x[0]+(x[1]-x[0]); Ly = y[-1]-y[0]+(y[1]-y[0]); Lt = t[-1]-t[0]+(t[1]-t[0])
    sx = width_ratio*Lx/2.355; sy = width_ratio*Ly/2.355; st = width_ratio*Lt/2.355
    rng = np.random.RandomState(seed)
    cx = rng.uniform(x[0], x[-1], n_test)
    cy = rng.uniform(y[0], y[-1], n_test)
    ct = rng.uniform(t[0], t[-1], n_test)
    X, Y, T = np.meshgrid(x, y, t, indexing='ij')
    funcs = []
    for i in range(n_test):
        phi = np.exp(-((X-cx[i])**2/(2*sx**2) + (Y-cy[i])**2/(2*sy**2) + (T-ct[i])**2/(2*st**2)))
        funcs.append(phi)
    return funcs

def generate_test_funcs_3d(x, y, z, t, n_test=30, width_ratio=0.15, seed=42):
    Nx, Ny, Nz, Nt = len(x), len(y), len(z), len(t)
    Lx = x[-1]-x[0]+(x[1]-x[0]); Ly = y[-1]-y[0]+(y[1]-y[0])
    Lz = z[-1]-z[0]+(z[1]-z[0]); Lt = t[-1]-t[0]+(t[1]-t[0])
    sx = width_ratio*Lx/2.355; sy = width_ratio*Ly/2.355; sz = width_ratio*Lz/2.355; st = width_ratio*Lt/2.355
    rng = np.random.RandomState(seed)
    cx = rng.uniform(x[0], x[-1], n_test)
    cy = rng.uniform(y[0], y[-1], n_test)
    cz = rng.uniform(z[0], z[-1], n_test)
    ct = rng.uniform(t[0], t[-1], n_test)
    X, Y, Z, T = np.meshgrid(x, y, z, t, indexing='ij')
    funcs = []
    for i in range(n_test):
        phi = np.exp(-((X-cx[i])**2/(2*sx**2) + (Y-cy[i])**2/(2*sy**2) + (Z-cz[i])**2/(2*sz**2) + (T-ct[i])**2/(2*st**2)))
        funcs.append(phi)
    return funcs

def fwbic(Theta, target, penalty=2.5, max_terms=8, tol=1e-10):
    """Forward-backward stepwise BIC."""
    N, P = Theta.shape
    selected = []
    remaining = list(range(P))
    best_bic = np.inf
    best_sel = []
    
    for _ in range(max_terms):
        best_j = None; best_j_bic = np.inf
        for j in remaining:
            cand = selected + [j]
            Xsub = Theta[:, cand]
            coef, res, _, _ = np.linalg.lstsq(Xsub, target, rcond=None)
            rss = np.sum((target - Xsub@coef)**2)
            bic = N*np.log(rss/N + 1e-30) + penalty*len(cand)*np.log(N)
            if bic < best_j_bic:
                best_j_bic = bic; best_j = j
        if best_j is None: break
        selected.append(best_j); remaining.remove(best_j)
        if best_j_bic < best_bic:
            best_bic = best_j_bic; best_sel = selected.copy()
        # Backward: try removing each selected term
        improved = True
        while improved and len(selected) > 1:
            improved = False
            for idx, j in enumerate(selected):
                cand = [s for s in selected if s != j]
                Xsub = Theta[:, cand]
                coef, _, _, _ = np.linalg.lstsq(Xsub, target, rcond=None)
                rss = np.sum((target - Xsub@coef)**2)
                bic = N*np.log(rss/N + 1e-30) + penalty*len(cand)*np.log(N)
                if bic < best_bic - tol:
                    best_bic = bic; best_sel = cand.copy()
                    selected = cand.copy()
                    improved = True
                    break
    
    coef = np.zeros(P)
    if best_sel:
        Xsub = Theta[:, best_sel]
        c, _, _, _ = np.linalg.lstsq(Xsub, target, rcond=None)
        coef[best_sel] = c
    return coef

def evaluate(coef, names, true_terms, tol=0.3):
    found = [names[i] for i in range(len(names)) if abs(coef[i]) > tol]
    true_found = sum(1 for t in true_terms if t in found)
    false_pos = sum(1 for f in found if f not in true_terms)
    return true_found, false_pos, found

def run_2d(pde_name, true_terms, noise_levels, n_test=50, penalty=2.5):
    print(f"\n--- {pde_name} ---")
    results = {}
    for nl in noise_levels:
        if nl == 'clean':
            fname = f"{pde_name}.npz"
        else:
            fname = f"{pde_name}_{nl}.npz"
        data = np.load(DATA_DIR / pde_name / fname, allow_pickle=True)
        u = np.squeeze(data['u']); x = data['x']; y = data['y']; t = data['t']
        ut = np.squeeze(data['du'])
        dx = x[1]-x[0]; dy = y[1]-y[0]; dt = t[1]-t[0]
        
        t0 = time.time()
        derivs = spectral_spatial_2d(u, x, y)
        features = build_library_2d(derivs)
        names = list(features.keys())
        
        # Weak formulation
        test_funcs = generate_test_funcs_2d(x, y, t, n_test=n_test)
        P = len(names)
        A = np.zeros((n_test, P))
        b = weak_integrate(ut, test_funcs, dx, dy, dt)
        for j, name in enumerate(names):
            A[:, j] = weak_integrate(features[name], test_funcs, dx, dy, dt)
        
        # L2 normalize
        norms = np.maximum(np.linalg.norm(A, axis=0), 1e-12)
        A_norm = A / norms
        b_norm = b / max(np.linalg.norm(b), 1e-12)
        coef_norm = fwbic(A_norm, b_norm, penalty=penalty)
        coef = coef_norm / norms * np.linalg.norm(b)
        
        elapsed = time.time() - t0
        tf, fp, found = evaluate(coef, names, true_terms)
        status = "EXACT" if (tf == len(true_terms) and fp == 0) else f"{tf}/{len(true_terms)}+{fp}FP"
        print(f"  {nl:8s}: {status:20s} ({elapsed:.1f}s)  found={found}")
        results[nl] = {'true_found': tf, 'total': len(true_terms), 'false_pos': fp, 'time': elapsed, 'found': found}
    return results

def run_3d(pde_name, true_terms, noise_levels, n_test=30, penalty=2.5):
    print(f"\n--- {pde_name} (3D) ---")
    results = {}
    for nl in noise_levels:
        if nl == 'clean':
            fname = f"{pde_name}.npz"
        else:
            fname = f"{pde_name}_{nl}.npz"
        data = np.load(DATA_DIR / pde_name / fname, allow_pickle=True)
        u = np.squeeze(data['u']); x = data['x']; y = data['y']; z = data['z']; t = data['t']
        ut = np.squeeze(data['du'])
        dx = x[1]-x[0]; dy = y[1]-y[0]; dz = z[1]-z[0]; dt = t[1]-t[0]
        
        t0 = time.time()
        derivs = spectral_spatial_3d(u, x, y, z)
        features = build_library_3d(derivs)
        names = list(features.keys())
        
        test_funcs = generate_test_funcs_3d(x, y, z, t, n_test=n_test)
        P = len(names)
        A = np.zeros((n_test, P))
        b = weak_integrate_3d(ut, test_funcs, dx, dy, dz, dt)
        for j, name in enumerate(names):
            A[:, j] = weak_integrate_3d(features[name], test_funcs, dx, dy, dz, dt)
        
        norms = np.maximum(np.linalg.norm(A, axis=0), 1e-12)
        A_norm = A / norms
        b_norm = b / max(np.linalg.norm(b), 1e-12)
        coef_norm = fwbic(A_norm, b_norm, penalty=penalty)
        coef = coef_norm / norms * np.linalg.norm(b)
        
        elapsed = time.time() - t0
        tf, fp, found = evaluate(coef, names, true_terms)
        status = "EXACT" if (tf == len(true_terms) and fp == 0) else f"{tf}/{len(true_terms)}+{fp}FP"
        print(f"  {nl:8s}: {status:20s} ({elapsed:.1f}s)  found={found}")
        results[nl] = {'true_found': tf, 'total': len(true_terms), 'false_pos': fp, 'time': elapsed, 'found': found}
    return results

if __name__ == '__main__':
    all_results = {}
    
    # 2D advection-diffusion
    all_results['advection_diffusion_2d'] = run_2d(
        'advection_diffusion_2d', 
        ['u_x', 'u_y', 'u_xx', 'u_yy'],
        ['clean', 'snr_40', 'snr_20'])
    
    # 2D heat soil (non-periodic)
    all_results['heat_soil_uniform_2d_p1'] = run_2d(
        'heat_soil_uniform_2d_p1',
        ['u_xx', 'u_yy'],
        ['clean', 'snr_40', 'snr_20'])
    
    # 3D heat soil
    all_results['heat_soil_uniform_3d_p1'] = run_3d(
        'heat_soil_uniform_3d_p1',
        ['u_xx', 'u_yy', 'u_zz'],
        ['clean', 'snr_40', 'snr_20'])
    
    with open(r"H:\2026科研\Spectral-PDE-Discovery\code\spd_v6_2d3d_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to spd_v6_2d3d_results.json")
