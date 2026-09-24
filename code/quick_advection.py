"""Quick Advection-only comparison (non-periodic, the key test case)."""
import numpy as np, sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, r'H:\2026科研\Spectral-PDE-Discovery\code')
from compare_strong_baselines import spd_1d, weak_sindy_1d, sr3_fd_1d, load_1d

DATA_DIR = r"H:\2026科研\Spectral-PDE-Discovery\mdbench\data\processed\data\pde"
tt = {'u_x': -0.1}

for noise in [None, 'snr_40', 'snr_30', 'snr_20', 'snr_10']:
    label = 'clean' if noise is None else noise
    nf = f"advection1d_{noise}.npz" if noise else None
    u,du,x,t = load_1d('advection1d', nf)
    print(f"\n--- {label} ---")
    for name, fn in [('SR3(FD)', sr3_fd_1d), ('Weak SINDy', weak_sindy_1d), ('SPD(ours)', spd_1d)]:
        r = fn(u,du,x,t,tt)
        status = "EXACT" if r['exact'] else f"{r['true_found']}/{r['total_true']}+{len(r['false_positives'])}FP"
        print(f"  {name:14s}: {status:12s} ({r['time_s']:.1f}s)  rec={list(r['recovered'].keys())[:5]}")
